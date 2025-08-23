import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, ToTensor, Compose
import numpy as np
import cv2
from os.path import join
import joblib
import logging
from glob import glob

from data_config import JOINT_REGRESSOR_H36M, PASCAL_OCCLUDERS, ROOT
from lib.core import constants, config
from lib.utils.imutils import crop, crop_crop, flip_img, flip_pose, flip_kp, transform, rot_aa
from lib.utils import rotation_conversions as geo
from lib.utils.geometry import perspective_projection, estimate_translation

from .coco_occlusion import occlude_with_pascal_objects
from lib.models.smpl import SMPL
from skimage.util.shape import view_as_windows


smpl = SMPL()
smpls = {g:SMPL(gender=g) for g in ['neutral', 'male', 'female']}

logger = logging.getLogger(__name__)


class VideoDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, dataset, ignore_3d=False, use_augmentation=True, is_train=True,
                normalization=False, cropped=False, crop_size=224, seqlen=16, stride=16, subset=True, need_warp=False):
        super(VideoDataset, self).__init__()
        
        self.is_train = is_train
        self.need_warp = need_warp
        self.dataset = dataset
        self.data = np.load(config.DATASET_FILES[is_train][dataset])
        self.extra_data = np.load(config.EXTRA_DATASET_FILES[is_train][dataset])
        self.img_dir = config.DATASET_FOLDERS[dataset]
        self.imgname = self.data['imgname'].astype(np.string_)
        self.normalization = normalization
        self.normalize_img = Compose([
                            ToTensor(),
                            Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
                        ])


        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.crop_size = crop_size

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']
        self.length = self.scale.shape[0]
        self.sc = 1.0
        
        # If False, do not do augmentation
        self.use_augmentation = use_augmentation
        self.occluders = joblib.load(PASCAL_OCCLUDERS)
        self.cropped = cropped

        # Get camera intrinsic, if available
        try:    
            self.img_focal = self.data['img_focal']
            self.img_center = self.data['img_center']
            self.has_camcalib = True

            print(dataset, 'has camera intrinsics')
        except KeyError:
            self.has_camcalib = False

        # Get camera intrinsic, if available
        try:
                
            self.orig_shape = self.data['orig_shape']
            print(dataset, 'has original image shape')
        except KeyError:
            self.orig_shape = None

        # Get camera extrinsic, if available
        try:
            self.camera = self.extra_data['camera']
            self.w2c_camera = self.extra_data['w2c_camera']
            self.has_extrinsic = True
            print(dataset, 'has camera extrinsic')
        except KeyError:
            self.has_extrinsic = False
        
        # Get GT camera, if available
        try:
            self.gt_camera = self.extra_data['GT_camera']
            self.gt_w2c_camera = self.extra_data['gt_w2c_camera']
            print(dataset, 'has GT camera')
        except KeyError:
            self.gt_camera = None
            self.gt_w2c_camera = None        
        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(float)
            self.betas = self.data['shape'].astype(float)
            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.imgname))
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname))
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname))

        
        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = 1
            #print(self.pose_3d[0]) #和我的一致，看看pose_3d做了哪些处理
            print(dataset, 'has pose_3d')
        except KeyError:
            self.has_pose_3d = 0
        if ignore_3d:
            self.has_pose_3d = 0

        if 'coco' in dataset or 'mpii' in dataset:
            self.has_pose_3d = 0
            print('Not using pose3d for', dataset)
        
        # Get 2D keypoints
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))

        self.keypoints = keypoints_gt

        # Get gender data, if available
        try:
            self.gender = self.data['gender']
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)
        self.gender = [str(g) for g in self.gender]

        ######### Convert to Video dataset #########
        if 'h36m' in self.dataset:
            seqname = [s[:-11] for s in self.imgname]
            self.seqname = np.array(seqname)
        
        if '3dpw' in self.dataset:
            self.seqname = self.data['seqname']

        if 'bedlam' in self.dataset:
            self.seqname = self.data['seqname']
        
        if 'emdb' in self.dataset:
            seqname = [s[:-17] for s in self.imgname.astype(str)]
            self.seqname = np.array(seqname)

        # Video
        self.seqlen = seqlen
        self.stride = stride

        if 'coco' not in self.dataset:
            
            self.seq_idx, self.group = self.split_into_chunks(self.seqname, seqlen, stride=stride)
            print(f'Found {len(self.seq_idx)} sequences in {self.dataset}.') #Found 1347 16 sequences in 3dpw_vid.
            if (not is_train) and subset:
                np.random.seed(0)
                self.seq_idx = np.random.permutation(self.seq_idx)
                self.seq_idx = self.seq_idx[:300].tolist()
                print(f'Using a subset of {self.dataset}')

            seqs = []
            for seq in self.seq_idx: #(N, 2)-->(N, seq)
                seqs.append(list(range(seq[0], seq[1]+1)))
            self.seq_idx = seqs
        else:
            seqs = []
            for i in range(len(self.imgname)):
                seqs.append([i] * self.seqlen)
            self.seq_idx = seqs        

        # Pre-cropped images
        cropdirs = {'h36m_vid': f'{ROOT}/h36m/crops',
                    '3dpw_vid': f'{ROOT}/3dpw/crops',
                    'bedlam_vid': f'{ROOT}/bedlam_30fps/crops',
                    'emdb_1':  f'{ROOT}/emdb/crops_1',
                    '3dpw_vid_test': f'{ROOT}/3dpw/crops_test'}
        
        if self.need_warp:
            warp_cropdirs = {'3dpw_vid': f'{ROOT}/3dpw/warped', 'emdb_1': f'{ROOT}/emdb/warped_1'}
            self.warp_dir = warp_cropdirs[self.dataset]

        self.crop_dir = cropdirs[self.dataset]
        self.crop_files = sorted(glob(f'{self.crop_dir}/*.jpg'))


    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)     # per channel pixel-noise
        rot = 0             # rotation
        sc = self.sc        # scaling
        occ = 0             # synthetic occlusion
        # if self.is_train and self.use_augmentation:
        if self.use_augmentation:
            OCCLUDE_PROB = 0.30
            NOISE_FACTOR = 0.20
            
            FLIP_PROB = 0.5
            ROT_FACTOR = 30
            SCALE_FACTOR = 0.20

            if np.random.uniform() <= OCCLUDE_PROB:
                occ = 1

            if np.random.uniform() <= FLIP_PROB:
                flip = 1

            if np.random.uniform() <= 0.6:
                rot = 0
            else:
                rot = min(2*ROT_FACTOR,
                      max(-2*ROT_FACTOR, np.random.randn()*ROT_FACTOR))
                
            sc = min(1+SCALE_FACTOR,
                 max(1-SCALE_FACTOR, np.random.randn()*SCALE_FACTOR+1))
            
            pn = np.random.uniform(1-NOISE_FACTOR, 1+NOISE_FACTOR, 3)

        return flip, pn, rot, sc, occ

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn, occ, sc):
        """Process rgb image and do augmentation."""
        if not self.cropped:
            rgb_img = crop(rgb_img, center, scale, 
                        [self.crop_size, self.crop_size], rot=rot)
        else:
            center = [128, 128]
            scale = 256/200 * sc #?
            rgb_img = crop_crop(rgb_img, center, scale, 
                        [self.crop_size, self.crop_size], rot=rot)
            
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)

        # occlusion augmentation: PARE uses this.
        if occ:
            rgb_img = occlude_with_pascal_objects(rgb_img, self.occluders)
            
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        
        return rgb_img.astype('uint8')

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                 [self.crop_size, self.crop_size], rot=r)

        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/self.crop_size - 1.

        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def center_processing(self, center, rot, flip, orig_shape):
        WH = orig_shape[::-1]

        if flip:
            rot = -rot
            center = center - WH/2
            center[0] = -center[0]
            center = center + WH/2
            
        R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot))],
                      [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot))]])

        aug_center = R @ (center - WH/2) + WH/2

        return aug_center


    def est_pose_3d(self, item, gender):
        shape = item['betas'][np.newaxis]
        pose = item['pose'].reshape(24, 3)[np.newaxis]
        pose = geo.axis_angle_to_matrix(pose)

        if gender in ['m', 'male']:
            gender = 'male'
        elif gender in ['f', 'female']:
            gender = 'female'
        else:
            gender = 'neutral'
        
        if '3dpw' in self.dataset:
            out = smpls[gender](global_orient=pose[:, [0]], body_pose=pose[:, 1:], betas=shape, 
                                pose2rot=False, default_smpl=False)
            vertices = out.vertices[0]
            gt_keypoints_3d = torch.matmul(self.J_regressor, vertices)

            gt_pelvis = gt_keypoints_3d[[0], :].clone()
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
            gt_verts = vertices - gt_pelvis

        elif 'emdb' in self.dataset:
            out = smpls[gender](global_orient=pose[:, [0]], body_pose=pose[:, 1:], betas=shape, 
                                pose2rot=False, default_smpl=True)
            vertices = out.vertices[0]
            gt_keypoints_3d = out.joints[0, :24]

            gt_pelvis = gt_keypoints_3d[[1,2], :].mean(dim=0, keepdims=True).clone()
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
            gt_verts = vertices - gt_pelvis


        conf = torch.ones([len(gt_keypoints_3d), 1])
        gt_keypoints_3d = torch.concat([gt_keypoints_3d, conf], axis=-1)

        return gt_keypoints_3d.numpy(), gt_verts.numpy()


    def __getitem__(self, index):
        augs = self.augm_params()
       
        indices = self.seq_idx[index] #取出一个序列
        items = []
        for idx in indices:
            item = self.get_frame(idx, augs)
            items.append(item)

        keys = [k for k in item]
        batch = {k:torch.stack([item[k] for item in items]) for k in keys}

        return batch
    
    def get_frame(self, index, augs):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters 数据增强
        flip, pn, rot, sc, occ = augs
        _, _, _, _, occ = self.augm_params()
        
        
        # Load image
        imgname = str(self.imgname[index], encoding='utf-8')
        imgname = join(self.img_dir, imgname)
        item['img_idx'] = torch.tensor([index]).long()

        if self.cropped:
            # cropfile = self.crop_files[index]
            crop_dir = self.crop_dir
            cropfile = f'{crop_dir}/{index:08d}.jpg'
            if self.need_warp:
                warp_cropdirs = self.warp_dir
                warp_cropdirs = f'{warp_cropdirs}/{index:08d}.jpg'
                #print("loading", warp_cropdirs)
            try:
                img = cv2.imread(cropfile)[:,:,::-1].copy().astype(float)
                if self.need_warp:
                    warp_img = cv2.imread(warp_cropdirs)[:,:,::-1].copy().astype(float)
                    cur_img = warp_img[:, 512:, :]
                    pre_img = warp_img[:, 256:512, :]
                    prepre_img = warp_img[:, :256, :]
                    warp_bags = [prepre_img, pre_img, cur_img]
                    #print("warp_bags", warp_bags[0].shape, warp_bags[1].shape, warp_bags[2].shape)
                    #print("warp_img", warp_img.shape)
            except Exception:
                print(f'Cropfile unavailable: {cropfile}')
                img = np.zeros([256, 256, 3]).astype(float)
                cur_img = np.zeros([256, 256, 3]).astype(float)
                prepre_img = np.zeros([256, 256, 3]).astype(float)
                pre_img = np.zeros([256, 256, 3]).astype(float)
        else:
            img = cv2.imread(imgname)[:,:,::-1].copy().astype(float)

        if self.orig_shape is not None:
            orig_shape = self.orig_shape[index]
        else:
            orig_shape = np.array(img.shape)[:2]

        # Get camera intrinsics
        if self.has_camcalib:
            item['img_focal'] = self.img_focal[index]
            item['img_center'] = self.img_center[index]
        else:
            item['img_focal'] = self.est_focal(orig_shape)
            item['img_center'] = self.est_center(orig_shape)


        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = torch.zeros(72).float()
            betas = torch.zeros(10).float()

        #Process image

        try:    
            img = self.rgb_processing(img, center, sc*scale, rot, flip, pn, occ, sc)
            if self.need_warp:
                warp_bags = [self.rgb_processing(bag, center, sc*scale, rot, flip, pn, occ, sc) for bag in warp_bags]
        except:
            img = np.zeros([self.crop_size, self.crop_size, 3])

        if self.normalization:
            img = self.normalize_img(img)
            if self.need_warp:
                warp_bags = [self.normalize_img(bag) for bag in warp_bags]
        else:
            img = torch.from_numpy(img)
            if self.need_warp:
                warp_img = torch.from_numpy(warp_img)

        # Store unnormalize image
        item['img'] = img
        if self.need_warp:
            item['img_cur'] = warp_bags[2]
            item['img_pre'] = warp_bags[1]
            item['img_prepre'] = warp_bags[0]
        item['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        item['betas'] = torch.from_numpy(betas).float()

        # Get 3D joints for training, if available
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()
            item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()
            
        else:
            item['pose_3d'] = torch.zeros(24,4, dtype=torch.float32)


        # Get SMPL 3D joints for evaluation
        if self.is_train == False:
            S, gt_verts = self.est_pose_3d(item, self.gender[index])
            item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()
            item['gt_verts'] = torch.from_numpy(gt_verts)


        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.keypoints[index].copy()
        keypoints = torch.from_numpy(self.j2d_processing(keypoints, center, sc*scale, rot, flip)).float()
        openpose = torch.zeros((25, 3))
        item['keypoints'] = torch.concatenate([openpose, keypoints], dim=0)
    
        # Apply augmentation transforms to bbox center
        center = self.center_processing(center, rot, flip, orig_shape)
        
        item['pred_camera'] = torch.from_numpy(self.camera[index].copy())
        item['pred_w2c_camera'] = torch.from_numpy(self.w2c_camera[index].copy())
        
        if self.is_train:
            item['gt_camera'] = torch.from_numpy(self.gt_camera[index].copy())
            item['gt_w2c_camera'] = torch.from_numpy(self.gt_w2c_camera[index].copy())
        
        item['scale'] = torch.tensor(sc * scale).float()
        item['center'] = torch.from_numpy(center).float()
        item['img_focal'] = torch.tensor(item['img_focal']).float()
        item['img_center'] = torch.from_numpy(item['img_center']).float()
        item['has_smpl'] = torch.tensor(self.has_smpl[index]).long()
        item['has_pose_3d'] = torch.tensor(self.has_pose_3d).long()
        
        
        return item


    def __len__(self):
        return len(self.seq_idx)


    def est_focal(self, orig_shape):
        h, w = orig_shape
        focal = np.sqrt(h**2 + w**2)
        return focal


    def est_center(self, orig_shape):
        h, w = orig_shape
        center = np.array([w/2., h/2.])
        return center


    def split_into_chunks(self, vid_names, seqlen, stride):
        print('Splitting into chunks')
        #print(vid_names) #seq_names 是有非常多的重复的,这是一个二键值的数组,要验证分出来的小序列是在一个视频里
        video_names, group = np.unique(vid_names, return_index=True)
        perm = np.argsort(group)
        video_names, group = video_names[perm], group[perm] #去重和排序

        indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])
        #print(indices) #(N, length) N 为视频的数量, length 为每个视频的长度
        if '3dpw' in self.dataset:
            invalid = self.detect_invalid_section()
        elif 'emdb' in self.dataset:
            invalid = self.data['invalid']
        elif 'mpi' in self.dataset:
            invalid = self.data['invalid']
        elif 'bedlam_vid' in self.dataset:
            invalid = self.data['invalid']
            self.invalid = invalid
        else:
            invalid = np.zeros(len(self.imgname)) # bool (N)

        video_start_end_indices = []

        for idx in range(len(video_names)): 
            indexes = indices[idx] #枚举每一个视频
            if indexes.shape[0] < seqlen:
                continue

            if idx == len(video_names)-1:
                indexes_invalid = invalid[group[idx]:]
            else:
                indexes_invalid = invalid[group[idx]:group[idx+1]]

            chunks = view_as_windows(indexes, (seqlen,), step=stride)
            #print("video length", indexes.shape[0]) # length*seqlen == N
            #print(chunks.shape) # (length, seqlen) length为序列数量 序列是不重复的
            chunks_invalid = view_as_windows(indexes_invalid, (seqlen,), step=stride)
            
            chunks_valid = chunks[chunks_invalid.sum(axis=-1)==0] #去除不合法的序列 (length, seqlen) = int 图像数组的index
            
            start_finish = chunks_valid[:, (0, -1)].tolist() # (length, 2) 取出每个序列的开始和结束的index
            video_start_end_indices += start_finish

        return video_start_end_indices, group
            

    def detect_invalid_section(self,):
        center = self.center
        size = self.scale * 200
        shape = self.data['img_shape']

        xy1 = center - size[:,None]/2
        xy2 = center + size[:,None]/2
        bbox = np.concatenate([xy1, xy2], axis=1)
        invalid = (bbox[:,2]<0) + (bbox[:,3]<0) + (bbox[:,0]>shape[:,0]) + (bbox[:,1]>shape[:,1])
        invalid = invalid + (self.data['valid']!=1)
        return invalid


