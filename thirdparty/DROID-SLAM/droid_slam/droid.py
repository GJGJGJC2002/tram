import torch
import lietorch
import numpy as np

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process


class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        
        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            # from visualization import droid_visualization
            from vis_headless import droid_visualization
            print('Using headless ...')
            self.visualizer = Process(target=droid_visualization, args=(self.video, '.'))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)

    def enable_recording(self, enabled=True):
        """Enable/disable recording of keyframe events and edge history"""
        self.filterx._record_filter = enabled
        self.frontend._record_keyframes = enabled
        self.frontend.graph._record_edges = enabled

    def get_debug_info(self):
        """Collect all recorded debug information"""
        n = self.video.counter.value
        tstamps = self.video.tstamp.cpu().numpy()[:n].tolist()

        info = {
            'num_keyframes': n,
            'keyframe_tstamps': tstamps,
            'motion_filter_log': self.filterx.filter_log,
            'keyframe_log': self.frontend.keyframe_log,
            'edge_history': self.frontend.graph.edge_history,
            'final_edge_snapshot': self.frontend.graph.get_edge_snapshot(),
        }
        return info


    def load_weights(self, weights):
        """ load trained model weights """

        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def track(self, tstamp, image, depth=None, intrinsics=None, mask=None):
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics, mask)

            # local bundle adjustment
            self.frontend()

            # global bundle adjustment
            # self.backend()

    def terminate(self, stream=None, backend=True):
        """ terminate the visualization process, return poses [t, q] """

        del self.frontend

        if backend:
            torch.cuda.empty_cache()
            # print("#" * 32)
            self.backend(7)

            torch.cuda.empty_cache()
            # print("#" * 32)
            self.backend(12)

        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy()
    
    def compute_error(self):
        """ compute slam reprojection error """

        del self.frontend

        torch.cuda.empty_cache()
        self.backend(12)

        return self.backend.errors[-1]
        

