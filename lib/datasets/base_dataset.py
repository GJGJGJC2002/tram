import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from os.path import join
import logging
import pickle
from lib.core import constants, config
from lib.utils.imutils import crop, transform, rot_aa

logger = logging.getLogger(__name__)

def world2cam(Rt):
    new_Rt = np.eye(4)[np.newaxis].repeat(repeats=Rt.shape[0], axis=0)
    new_Rt[:, :3, :3] = Rt[:, :3, :3]
    #new_Rt[:, :3, 1:3] *= -1.0
    new_Rt[:, :3, :3] = np.transpose(new_Rt[:, :3, :3], axes=(0, 2, 1))
    new_Rt[:, :3, 3:] = -new_Rt[:, :3, :3] @ Rt[:, :3, 3:]
    return new_Rt

def perspective_projection(points, focal_length, camera_center):
    """
    Project 3D points to 2D using perspective projection.
    
    Args:
        points (np.ndarray): 3D points of shape (N, 3).
        focal_length (float): Focal length of the camera.
        camera_center (np.ndarray): Camera center of shape (2,).
        
    Returns:
        np.ndarray: Projected 2D points of shape (N, 2).
    """
    x = points[:, 0] / points[:, 2] * focal_length + camera_center[0]
    y = points[:, 1] / points[:, 2] * focal_length + camera_center[1]
    #将nan inf替换为0
    x[np.isnan(x) | np.isinf(x)] = 0
    y[np.isnan(y) | np.isinf(y)] = 0
    return np.stack((x, y), axis=-1)

def visual_j2d(img, j2d, color=(0, 255, 0), radius=2):
    """
    Visualize 2D joints on an image.
    
    Args:
        img (np.ndarray): Input image.
        j2d (np.ndarray): 2D joints of shape (N, 2).
        color (tuple): Color for the joints.
        radius (int): Radius of the joints.
        
    Returns:
        np.ndarray: Image with visualized joints.
    """
    for joint in j2d:
        cv2.circle(img, tuple(joint.astype(int)), radius, color, -1)
    return img

class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in /config.py.
    """

    def __init__(self, dataset, is_train=True, crop_size=256):
        super(BaseDataset, self).__init__()
        
        self.is_train = is_train
        self.append_camera = False
        self.dataset = dataset
        print(f'Loading {dataset} dataset ...', config.DATASET_FILES[is_train][dataset])
        print(f'Loading {dataset} extra_dataset ...', config.EXTRA_DATASET_FILES[is_train][dataset])
        self.data = np.load(config.DATASET_FILES[is_train][dataset])
        self.extra_data = np.load(config.EXTRA_DATASET_FILES[is_train][dataset])
        self.img_dir = config.DATASET_FOLDERS[dataset]
        self.imgname = self.data['imgname'].astype(np.string_) #(23543)

        print(f'Loading {dataset} dataset ...', self.imgname.shape)
        #print(self.imgname[:100])
        self.crop_size = crop_size

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']
        self.sc = 1.0

        
        # Get camera intrinsic, if available
        try:    
            self.img_focal = self.data['img_focal']
            self.img_center = self.data['img_center']
            N = self.img_focal.shape[0]
            self.intrinsics = np.zeros((N, 3, 3))
            self.intrinsics[:, 0, 0] = self.img_focal
            self.intrinsics[:, 1, 1] = self.img_focal
            self.intrinsics[:, 0, 2] = self.img_center[:, 0]
            self.intrinsics[:, 1, 2] = self.img_center[:, 1]
            self.intrinsics[:, 2, 2] = 1
            self.has_camcalib = True
            print(dataset, 'has camera intrinsics')
        except KeyError:
            self.has_camcalib = False
        
        if self.append_camera:
            # Get camera extrinsic, if available
            if dataset == '3dpw_vid':
                tram_root = "/home/gejunchen/Work/2024-11/Baseline/tram/results"
                GT_root = "/home/gejunchen/Work/2024-7/Dataset/3DPW/sequenceFiles/train"
            if dataset == 'emdb_1':
                tram_root = "/home/gejunchen/Work/2024-11/Baseline/tram/tram_results"
            try:
                self.camera = []
                self.gt_camera = []
                self.img_seq = []
                self.img_num = []
                now_open_seq = ''
                for img_path in self.data['imgname']:
                    print("loading", img_path)
                    if dataset == '3dpw_vid':
                        img_name = img_path.split('/')[-1]
                        seq_name = img_path.split('/')[1]
                        img_num = img_name[6:-4]
                    else:
                        img_name = img_path.split('/')[-1]
                        seq_name = img_path.split('/')[0]+'_'+img_path.split('/')[1]+'_I'
                        img_num = img_name[:5]
                    print("getting", img_name, seq_name, img_num)
                    if seq_name != now_open_seq:
                        cam_R = np.load(join(tram_root, seq_name, 'camera.npy'), allow_pickle=True).item()['pred_cam_R']
                        cam_T = np.load(join(tram_root, seq_name, 'camera.npy'), allow_pickle=True).item()['pred_cam_T']
                    self.img_num.append(img_num)
                    self.img_seq.append(seq_name)
                    camera = np.eye(4)
                    camera[:3, :3] = cam_R[int(img_num)]
                    camera[:3, 3] = cam_T[int(img_num)]
                    camera[3, 3] = 1
                    self.camera.append(camera)
                    if 1: #get GT
                        if seq_name != now_open_seq:
                            with open(join(GT_root, seq_name+'.pkl'), 'rb') as f:
                                data = pickle.load(f, encoding='latin1')
                            print(data.keys())
                            print(data['cam_poses'].shape)
                        self.gt_camera.append(data['cam_poses'][int(img_num)])
                    now_open_seq = seq_name
                
                self.img_seq = np.array(self.img_seq)
                self.img_num = np.array(self.img_num)
                self.camera = np.stack(self.camera, axis=0) #(23543, 4, 4)
                self.gt_camera = np.stack(self.gt_camera, axis=0) #(23543, 4, 4)
                self.w2c_camera = world2cam(self.camera) #(23543, 4, 4)
                self.gt_w2c_camera = world2cam(self.gt_camera) #(23543, 4, 4)
                #保存至config.EXTRA_DATASET_FILES[is_train][dataset]
                save_dict = {'camera': self.camera, 'GT_camera':self.gt_camera, 'w2c_camera': self.w2c_camera, 'gt_w2c_camera': self.gt_w2c_camera, 'img_seq': self.img_seq, 'img_num': self.img_num}
                print("saving in", config.EXTRA_DATASET_FILES[is_train][dataset])
                np.savez(config.EXTRA_DATASET_FILES[is_train][dataset], **save_dict)

                self.has_camextrinsic = True
                print(dataset, 'has camera extrinsics')
            except KeyError:
                self.has_camextrinsic = False
        else:
            self.camera = self.extra_data['camera']
            self.gt_camera = self.extra_data['GT_camera']
            self.w2c_camera = self.extra_data['w2c_camera']
            self.gt_w2c_camera = self.extra_data['gt_w2c_camera']
            self.img_seq = self.extra_data['img_seq']

        self.length = self.scale.shape[0]

        if 'mpi3d_vid' in dataset:
            self.invalid = self.data['invalid']
        elif '3dpw' in dataset:
            self.invalid = self.detect_invalid_section()
        elif 'bedlam' in dataset:
            self.invalid = self.data['invalid']
        else:
            self.invalid = np.zeros(len(self))
        
        try:
            self.pose = self.data['pose'].astype(float)
            self.betas = self.data['shape'].astype(float)
            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.imgname))
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname))

        try:
            self.keypoints_gt_3d = self.data['S'][:, :, :3] #(23543, 24, 3)
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0
        
        try:
            self.keypoints_gt_2d = self.data['part'][:, :, :2] #(23543, 24, 2)
        except KeyError:
            self.keypoints_gt_2d = np.zeros((len(self.imgname), 24, 2))


    def rgb_processing(self, rgb_img, center, scale):
        """Process rgb image and do augmentation."""
        
        rgb_img = crop(rgb_img, center, scale, 
                    [self.crop_size, self.crop_size], rot=0)
        
        return rgb_img.astype('uint8')


    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        crop_size = self.crop_size
        center = self.center[index].copy()
        img_center = self.img_center[index].copy()
        img_focal = self.img_focal[index].copy()  

        if self.dataset == 'emdb_1':
            self.img_dir = '/home/gejunchen/Work/2024-11/Dataset/EMDB'

        pre_dis = 16
        # Load image
        imgname = str(self.imgname[index], encoding='utf-8')
        imgname = join(self.img_dir, imgname)
        
        if index-pre_dis < 0 or self.img_seq[index-pre_dis] != self.img_seq[index]:
            pre_dis = 0

        pre_pre_dis = 32
        if index-pre_pre_dis < 0 or self.img_seq[index-pre_pre_dis] != self.img_seq[index]:
            pre_pre_dis = pre_dis
    
        pre_imgname = str(self.imgname[index-pre_dis], encoding='utf-8')
        pre_imgname = join(self.img_dir, pre_imgname)
        prepre_imgname = str(self.imgname[index-pre_pre_dis], encoding='utf-8')
        prepre_imgname = join(self.img_dir, prepre_imgname)

        try:
            img = cv2.imread(imgname).copy().astype(np.float32)
            pre_img = cv2.imread(pre_imgname).copy().astype(np.float32)
            Related_M = self.w2c_camera[index]@self.camera[index-pre_dis]
            Related_R = Related_M[:3, :3]
            Related_t = Related_M[:3, 3]
            K = self.intrinsics[index]
            #print("K", K)
            K_inv = np.linalg.inv(K)
            H_pre = K@Related_R@K_inv
            warp_pre_img = cv2.warpPerspective(pre_img, H_pre, (img.shape[1], img.shape[0]))
            print("j2d_cur", self.keypoints_gt_2d[index].shape, self.keypoints_gt_2d[index])
            pre_j3d_cur = np.zeros((24, 4))
            pre_j3d_cur[:, :3] = self.keypoints_gt_3d[index-pre_dis]
            pre_j3d_cur[:, 3] = 1
            pre_j3d_cur = np.einsum('ij,kj->ki', self.camera[index-pre_dis], pre_j3d_cur)
            pre_j3d_cur = np.einsum('ij,kj->ki', self.w2c_camera[index], pre_j3d_cur)
            pre_j3d_cur = pre_j3d_cur[:,:3]
            print("pre_j3d_cur", pre_j3d_cur.shape, pre_j3d_cur)
            pre_j2d_cur = perspective_projection(pre_j3d_cur, img_focal, img_center)
            print("pre_j2d_cur_full", pre_j2d_cur.shape, pre_j2d_cur)
            pre_j2d_img_full = visual_j2d(pre_img.copy(), pre_j2d_cur, color=(0, 255, 0), radius=2)
            b = scale * 200
            pre_j2d_cur_cropped = pre_j2d_cur - (center - b/2)[None, :]
            pre_j2d_cur_cropped = pre_j2d_cur_cropped / b * self.crop_size
            print("pre_j2d_cur_cropped", pre_j2d_cur_cropped.shape, pre_j2d_cur_cropped)
            pre_j2d_img = visual_j2d(img.copy(), pre_j2d_cur_cropped, color=(0, 255, 0), radius=2)

            prepre_img = cv2.imread(prepre_imgname).copy().astype(np.float32)
            Related_M = self.w2c_camera[index]@self.camera[index-pre_pre_dis]
            Related_R = Related_M[:3, :3]
            Related_t = Related_M[:3, 3]
            H_pre_pre = K@Related_R@K_inv
            warp_prepre_img = cv2.warpPerspective(prepre_img, H_pre_pre, (img.shape[1], img.shape[0]))
            invalid = self.invalid[index]
            pre_invalid = self.invalid[index-pre_dis]
            prepre_invalid = self.invalid[index-pre_pre_dis]
            if invalid:
                img_crop = np.zeros([256, 256, 3]).astype('uint8')
            else:
                try:
                    img_crop = self.rgb_processing(img, center, scale)
                except:
                    img_crop = np.zeros([256, 256, 3]).astype('uint8')
            if pre_invalid:
                warp_pre_img = np.zeros([256, 256, 3]).astype('uint8')
            else:
                warp_pre_img = self.rgb_processing(warp_pre_img, center, scale)
            if prepre_invalid:
                warp_prepre_img = np.zeros([256, 256, 3]).astype('uint8')
            else:
                warp_prepre_img = self.rgb_processing(warp_prepre_img, center, scale)
            
            concat_img = np.concatenate((warp_prepre_img, warp_pre_img, img_crop), axis=1)

        except TypeError:
            logger.info(f"cv2 loading error image={imgname}")

        # bug
        if self.dataset=='bedlam' and self.img_center[index][0] == 360: # supposely a bedlam tall image 
            if img.shape[1] != 720:  # but the given image is not ...
                img = np.transpose(img, [1,0,2])[:,::-1,:].copy()

        # Process image
        invalid = self.invalid[index]




        # Store unnormalize image
        item['img'] = img_crop
        item['warp_img'] = concat_img
        item['imgname'] = imgname
        item['warp_prepre_img'] = warp_prepre_img
        item['warp_pre_img'] = warp_pre_img
        item['j3d_cur'] = self.keypoints_gt_3d[index].copy()
        item['pre_j3d_cur'] = pre_j3d_cur
        item['pre_j2d_img'] = pre_j2d_img
        item['pre_j2d_img_full'] = pre_j2d_img_full
        return item


    def __len__(self):
        return len(self.imgname)
    

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

