"""
Warm Start DROID-SLAM

Provides modified DROID-SLAM components that support:
1. Forced keyframe selection (bypassing MotionFilter's optical flow threshold)
2. Warm start initialization (injecting pre-computed poses from a previous SLAM run)
3. Disabled keyframe removal in frontend (preserving all forced keyframes)

These components inherit from the original DROID-SLAM classes to avoid modifying
the third-party code directly.
"""

import sys
sys.path.insert(0, 'thirdparty/DROID-SLAM/droid_slam')
sys.path.insert(0, 'thirdparty/DROID-SLAM')

import torch
import lietorch
import numpy as np
import cv2
import json
import os
from tqdm import tqdm
from glob import glob
from PIL import Image
from collections import OrderedDict

from droid import Droid
from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller
from factor_graph import FactorGraph
from lietorch import SE3

import geom.projective_ops as pops
from modules.corr import CorrBlock

from .slam_utils import slam_args, parser, image_stream, est_calib, get_dimention, preprocess_masks
from .est_scale import est_scale_hybrid
from ..utils.rotation_conversions import quaternion_to_matrix


class WarmstartMotionFilter(MotionFilter):
    """
    Modified MotionFilter that forces keyframe selection based on a pre-determined
    set of frame timestamps, instead of using optical flow magnitude thresholds.
    
    Optionally injects pre-computed SE3 poses for warm start initialization.
    """

    def __init__(self, net, video, forced_keyframes, initial_poses_se3=None,
                 initial_disps=None, thresh=2.5, device="cuda:0"):
        """
        Args:
            net: DroidNet
            video: DepthVideo
            forced_keyframes: set of int timestamps that must be keyframes
            initial_poses_se3: dict mapping tstamp (int) -> numpy array [7] (SE3 format)
            initial_disps: dict mapping tstamp (int) -> numpy array [h//8, w//8]
            thresh: fallback threshold (not used when forced_keyframes is provided)
            device: cuda device
        """
        super().__init__(net, video, thresh=thresh, device=device)
        self.forced_keyframes = set(forced_keyframes)
        self.initial_poses_se3 = initial_poses_se3 or {}
        self.initial_disps = initial_disps or {}

    @torch.amp.autocast('cuda', enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth=None, intrinsics=None, mask=None):
        """
        Modified track: only adds frames that are in forced_keyframes.
        Injects pre-computed pose if available.
        """

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8

        # normalize images
        inputs = image[None, :, [2, 1, 0]].to(self.device) / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features (always needed for keyframes)
        gmap = self._MotionFilter__feature_encoder(inputs)
        if mask is None:
            mask = torch.zeros([gmap.shape[-2], gmap.shape[-1]]).to(gmap)

        # Check if this frame should be a keyframe
        is_forced = int(tstamp) in self.forced_keyframes

        ### always add first frame ###
        if self.video.counter.value == 0:
            if not is_forced:
                # First forced keyframe hasn't arrived yet, skip
                return

            net, inp = self._MotionFilter__context_encoder(inputs[:, [0]])
            self.net, self.inp, self.fmap = net, inp, gmap

            # Use warm start pose if available, otherwise identity
            init_pose = Id
            init_disp = 1.0
            ts = int(tstamp)
            if ts in self.initial_poses_se3:
                init_pose = torch.as_tensor(
                    self.initial_poses_se3[ts], dtype=torch.float, device="cuda"
                )
            if ts in self.initial_disps:
                init_disp = torch.as_tensor(
                    self.initial_disps[ts], dtype=torch.float, device="cuda"
                )

            self.video.append(
                tstamp, image[0], init_pose, init_disp, depth,
                intrinsics / 8.0, gmap, net[0, 0], inp[0, 0], mask
            )

            if self._record_filter:
                self.filter_log.append({
                    'tstamp': int(tstamp),
                    'accepted': True,
                    'reason': 'first_frame_forced',
                    'flow_magnitude': None,
                    'keyframe_idx': 0,
                })

        ### for subsequent frames, only add if forced ###
        elif is_forced:
            self.count = 0
            net, inp = self._MotionFilter__context_encoder(inputs[:, [0]])
            self.net, self.inp, self.fmap = net, inp, gmap

            # Use warm start pose if available
            init_pose = None  # None = use frontend's prediction (poses[t1-1])
            init_disp = None
            ts = int(tstamp)
            if ts in self.initial_poses_se3:
                init_pose = torch.as_tensor(
                    self.initial_poses_se3[ts], dtype=torch.float, device="cuda"
                )
            if ts in self.initial_disps:
                init_disp = torch.as_tensor(
                    self.initial_disps[ts], dtype=torch.float, device="cuda"
                )

            self.video.append(
                tstamp, image[0], init_pose, init_disp, depth,
                intrinsics / 8.0, gmap, net[0], inp[0], mask
            )

            if self._record_filter:
                self.filter_log.append({
                    'tstamp': int(tstamp),
                    'accepted': True,
                    'reason': 'forced_keyframe',
                    'flow_magnitude': None,
                    'keyframe_idx': self.video.counter.value - 1,
                })

        else:
            # Not a forced keyframe, skip entirely
            self.count += 1

            if self._record_filter:
                self.filter_log.append({
                    'tstamp': int(tstamp),
                    'accepted': False,
                    'reason': 'not_forced_keyframe',
                    'flow_magnitude': None,
                })


class WarmstartDroidFrontend(DroidFrontend):
    """
    Modified DroidFrontend that disables keyframe removal.
    
    Since we force specific keyframes, the frontend should not remove any of them
    based on the distance metric.
    """

    def __init__(self, net, video, args):
        super().__init__(net, video, args)
        # Disable keyframe removal by setting threshold to 0
        # (distance is always >= 0, so d < 0 is never True)
        self.keyframe_thresh = 0.0

    # Note: __update uses name mangling (_DroidFrontend__update).
    # By setting keyframe_thresh=0.0 in __init__, the removal branch
    # in the parent's __update will never trigger, so we don't need
    # to override __update or __call__.


class WarmstartDroid(Droid):
    """
    Modified Droid that uses WarmstartMotionFilter and WarmstartDroidFrontend.
    """

    def __init__(self, args, forced_keyframes, initial_poses_se3=None, initial_disps=None):
        """
        Args:
            args: SLAM arguments (same as original Droid)
            forced_keyframes: set/list of int timestamps for forced keyframes
            initial_poses_se3: dict mapping tstamp -> SE3 pose array [7]
            initial_disps: dict mapping tstamp -> disparity array [h//8, w//8]
        """
        # We can't call super().__init__ because it creates MotionFilter/DroidFrontend
        # that we want to replace. So we replicate the init logic.
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # Use warmstart motion filter
        self.filterx = WarmstartMotionFilter(
            self.net, self.video,
            forced_keyframes=forced_keyframes,
            initial_poses_se3=initial_poses_se3,
            initial_disps=initial_disps,
            thresh=args.filter_thresh
        )

        # Use warmstart frontend (no keyframe removal)
        self.frontend = WarmstartDroidFrontend(self.net, self.video, self.args)

        # Standard backend
        self.backend = DroidBackend(self.net, self.video, self.args)

        # Visualizer
        if not self.disable_vis:
            from vis_headless import droid_visualization
            print('Using headless ...')
            from torch.multiprocessing import Process
            self.visualizer = Process(target=droid_visualization, args=(self.video, '.'))
            self.visualizer.start()

        # Post processor
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)


def _load_image_for_slam(imfile, calib):
    """Load and preprocess a single image for DROID-SLAM (resize + crop to multiple of 8)."""
    fx, fy, cx, cy = calib[:4]
    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image = cv2.imread(imfile)
    if len(calib) > 4:
        image = cv2.undistort(image, K, calib[4:])

    h0, w0, _ = image.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

    image = cv2.resize(image, (w1, h1))
    image = image[:h1-h1%8, :w1-w1%8]
    image = torch.as_tensor(image).permute(2, 0, 1)

    intrinsics = torch.as_tensor([fx, fy, cx, cy])
    intrinsics[0::2] *= (w1 / w0)
    intrinsics[1::2] *= (h1 / h0)

    return image[None], intrinsics


def run_slam_warmstart(imagedir, forced_keyframes, initial_poses_se3=None,
                       initial_disps=None, masks=None, calib=None,
                       depth=None, record_debug=True, original_imagedir=None,
                       use_rendered_for_keyframes=False):
    """
    Run DROID-SLAM with forced keyframes and warm start initialization.
    
    Args:
        imagedir: directory containing rendered .jpg files (adjacent_smpl/ images)
        forced_keyframes: set/list of int frame indices to force as keyframes
        initial_poses_se3: dict mapping frame_idx -> SE3 pose [7]
        initial_disps: dict mapping frame_idx -> disparity [h//8, w//8]
        masks: optional human masks for masking
        calib: camera intrinsics [fx, fy, cx, cy]
        depth: optional depth input
        record_debug: whether to record debug info
        original_imagedir: directory containing original (unmodified) images.
        use_rendered_for_keyframes: if True, keyframes use rendered images from
            imagedir (with SMPL mesh), non-keyframes use original images + mask.
            If False, all frames use original images + mask (baseline behavior).
    
    Returns:
        droid: WarmstartDroid object
        traj: full trajectory [N_total, 7] for all input frames
    """
    droid = None

    # Determine base image dir (for calib, mask sizing, traj_filler)
    base_imagedir = original_imagedir if original_imagedir else imagedir

    if calib is None:
        calib = est_calib(base_imagedir)

    if masks is not None:
        img_msks, conf_msks = preprocess_masks(base_imagedir, masks)

    forced_set = set(int(k) for k in forced_keyframes)

    # Pre-load rendered image file list if needed
    rendered_image_list = None
    if use_rendered_for_keyframes and original_imagedir:
        rendered_image_list = sorted(glob(os.path.join(imagedir, '*.jpg')))
        print(f"[Warmstart] Keyframes use rendered images from {imagedir}")
        print(f"[Warmstart] Non-keyframes use original images from {original_imagedir}")

    for (t, image, intrinsics) in tqdm(image_stream(base_imagedir, calib)):
        if droid is None:
            slam_args.image_size = [image.shape[2], image.shape[3]]
            droid = WarmstartDroid(
                slam_args,
                forced_keyframes=forced_set,
                initial_poses_se3=initial_poses_se3,
                initial_disps=initial_disps
            )
            if record_debug:
                droid.enable_recording(True)

        # Decide which image to use for this frame
        is_keyframe = int(t) in forced_set
        if use_rendered_for_keyframes and is_keyframe and rendered_image_list is not None:
            # Keyframe: use rendered image (with SMPL mesh from adjacent_smpl/)
            if t < len(rendered_image_list):
                rendered_image, _ = _load_image_for_slam(rendered_image_list[t], calib)
                track_image = rendered_image
            else:
                track_image = image
            # Keyframe still uses mask for correlation weighting:
            # SMPL mesh provides visual context for feature extraction,
            # but mask prevents false matches on non-static human regions
            if masks is not None:
                conf_msk = conf_msks[t]
                droid.track(t, track_image, intrinsics=intrinsics, depth=depth, mask=conf_msk)
            else:
                droid.track(t, track_image, intrinsics=intrinsics, depth=depth, mask=None)
        else:
            # Non-keyframe (or baseline mode): use original image + mask
            if masks is not None:
                img_msk = img_msks[t]
                conf_msk = conf_msks[t]
                image = image * (img_msk < 0.5)
                droid.track(t, image, intrinsics=intrinsics, depth=depth, mask=conf_msk)
            else:
                droid.track(t, image, intrinsics=intrinsics, depth=depth, mask=None)

    if droid is None:
        return None, None

    # save debug info before terminate
    debug_info = None
    if record_debug:
        debug_info = droid.get_debug_info()

    # traj_filler uses original images (unmodified)
    traj = droid.terminate(image_stream(base_imagedir, calib))

    # save debug info
    if debug_info is not None:
        n = droid.video.counter.value
        debug_info['num_keyframes_after_backend'] = n
        debug_info['keyframe_tstamps_after_backend'] = droid.video.tstamp.cpu().numpy()[:n].tolist()
        debug_info['warmstart'] = True
        debug_info['forced_keyframes'] = sorted(list(forced_set))
        debug_info['use_rendered_for_keyframes'] = use_rendered_for_keyframes

        debug_dir = os.path.join(imagedir, '..', 'droid_debug')
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, 'droid_frontend_debug.json')
        with open(debug_path, 'w') as f:
            json.dump(debug_info, f, indent=2)
        print(f"[DROID Warmstart Debug] Saved debug info to {debug_path}")

    return droid, traj


def run_metric_slam_warmstart(img_folder, forced_keyframes, initial_poses_se3=None,
                              initial_disps=None, masks=None, calib=None,
                              is_static=False, original_image_dir=None,
                              use_rendered_for_keyframes=False):
    """
    Run metric-scale SLAM with forced keyframes and warm start.
    
    Returns full-frame camera trajectory (no interpolation needed).
    
    Args:
        img_folder: directory containing .jpg files (rendered images with SMPL mesh)
        forced_keyframes: set/list of int frame indices to force as keyframes
        initial_poses_se3: dict mapping frame_idx -> SE3 pose [7]
        initial_disps: dict mapping frame_idx -> disparity [h//8, w//8]
        masks: optional human masks
        calib: camera intrinsics [fx, fy, cx, cy]
        is_static: if True, return identity camera motion
        original_image_dir: directory containing original images
        use_rendered_for_keyframes: if True, keyframes use rendered images,
            non-keyframes use original images + mask
    
    Returns:
        pred_cam_r: rotation matrices [N, 3, 3]
        pred_cam_t: translation vectors [N, 3]
    """
    # Use original images directory for frame counting and depth estimation
    base_img_folder = original_image_dir if original_image_dir else img_folder
    imgfiles = sorted(glob(f'{base_img_folder}/*.jpg'))
    if len(imgfiles) == 0:
        # Fallback to img_folder
        base_img_folder = img_folder
        imgfiles = sorted(glob(f'{img_folder}/*.jpg'))

    if is_static:
        pred_cam_t = torch.zeros([len(imgfiles), 3])
        pred_cam_r = torch.eye(3).expand(len(imgfiles), 3, 3)
        return pred_cam_r, pred_cam_t

    # Run warmstart SLAM
    droid, traj = run_slam_warmstart(
        img_folder,
        forced_keyframes=forced_keyframes,
        initial_poses_se3=initial_poses_se3,
        initial_disps=initial_disps,
        masks=masks,
        calib=calib,
        original_imagedir=original_image_dir,
        use_rendered_for_keyframes=use_rendered_for_keyframes,
    )

    if droid is None:
        raise RuntimeError(
            f"Warmstart DROID-SLAM failed. No images found or processed in {img_folder}."
        )

    n = droid.video.counter.value
    tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
    disps = droid.video.disps_up.cpu().numpy()[:n]
    del droid
    torch.cuda.empty_cache()

    # Estimate metric depth using ZoeDepth on original images
    zoe_imgfiles = imgfiles
    print(f"[ZoeDepth] Using images from {base_img_folder}")

    repo = "isl-org/ZoeDepth"
    model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)
    _ = model_zoe_n.eval()
    model_zoe_n = model_zoe_n.to('cuda')

    pred_depths = []
    H, W = get_dimention(base_img_folder)
    for t in tqdm(tstamp):
        img = cv2.imread(zoe_imgfiles[t])[:, :, ::-1]
        img = cv2.resize(img, (W, H))
        img_pil = Image.fromarray(img)
        pred_depth = model_zoe_n.infer_pil(img_pil)
        pred_depths.append(pred_depth)

    # Estimate metric scale
    scales_ = []
    for i in tqdm(range(len(tstamp))):
        t = tstamp[i]
        disp = disps[i]
        pred_depth = pred_depths[i]
        slam_depth = 1 / disp

        if masks is None:
            msk = None
        else:
            msk = masks[t].numpy()

        scale = est_scale_hybrid(slam_depth, pred_depth, msk=msk)
        scales_.append(scale)
    scale = np.median(scales_)

    del model_zoe_n
    del pred_depths
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Convert to metric-scale camera extrinsics
    pred_cam_t = torch.tensor(traj[:, :3]) * scale
    pred_cam_q = torch.tensor(traj[:, 3:])
    pred_cam_r = quaternion_to_matrix(pred_cam_q[:, [3, 0, 1, 2]])

    return pred_cam_r, pred_cam_t
