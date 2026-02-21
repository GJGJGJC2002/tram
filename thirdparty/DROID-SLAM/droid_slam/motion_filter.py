import cv2
import torch
import lietorch

from collections import OrderedDict
from droid_net import DroidNet

import geom.projective_ops as pops
from modules.corr import CorrBlock


class MotionFilter:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self, net, video, thresh=2.5, device="cuda:0"):
        
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # hook: record which frames are accepted/rejected as keyframes
        self.filter_log = []  # list of dicts
        self._record_filter = False  # set True to enable

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
    @torch.amp.autocast('cuda', enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.amp.autocast('cuda', enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)

    @torch.amp.autocast('cuda', enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth=None, intrinsics=None, mask=None):
        """ main update operation - run on every frame in video """

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8

        # normalize images
        inputs = image[None, :, [2,1,0]].to(self.device) / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs) # [1, 128, gh, gw]
        if mask is None:
            mask = torch.zeros([gmap.shape[-2], gmap.shape[-1]]).to(gmap)
        # if mask is not None:
        #     # bias = self.fnet.conv2.bias.detach().clone().half()
        #     # gmap[:,:,mask>0.0] = bias[:, None].repeat(1, (mask>0.0).sum())
        #     gmap[:,:,mask>0.0] = 0

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            net, inp = self.__context_encoder(inputs[:,[0]])
            self.net, self.inp, self.fmap = net, inp, gmap
            self.video.append(tstamp, image[0], Id, 1.0, depth, intrinsics / 8.0, gmap, net[0,0], inp[0,0], mask)

            if self._record_filter:
                self.filter_log.append({
                    'tstamp': int(tstamp),
                    'accepted': True,
                    'reason': 'first_frame',
                    'flow_magnitude': None,
                    'keyframe_idx': 0,
                })

        ### only add new frame if there is enough motion ###
        else:                
            # index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)

            flow_mag = delta.norm(dim=-1).mean().item()

            # check motion magnitue / add new frame to video
            if flow_mag > self.thresh:
                self.count = 0
                net, inp = self.__context_encoder(inputs[:,[0]])
                self.net, self.inp, self.fmap = net, inp, gmap
                self.video.append(tstamp, image[0], None, None, depth, intrinsics / 8.0, gmap, net[0], inp[0], mask)

                if self._record_filter:
                    self.filter_log.append({
                        'tstamp': int(tstamp),
                        'accepted': True,
                        'reason': 'motion_above_thresh',
                        'flow_magnitude': flow_mag,
                        'threshold': self.thresh,
                        'keyframe_idx': self.video.counter.value - 1,
                    })

            else:
                self.count += 1

                if self._record_filter:
                    self.filter_log.append({
                        'tstamp': int(tstamp),
                        'accepted': False,
                        'reason': 'motion_below_thresh',
                        'flow_magnitude': flow_mag,
                        'threshold': self.thresh,
                    })
