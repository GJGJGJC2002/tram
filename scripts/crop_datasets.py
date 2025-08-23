import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import cv2
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from data_config import ROOT
from lib.datasets.base_dataset import BaseDataset


SEED_VALUE = 0
os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)
np.random.seed(SEED_VALUE)

# Datasets
# ds_list = ['3dpw_vid', 'h36m_vid', 'bedlam_vid']
ds_list = ['3dpw_vid']
#ds_list = ['emdb_1']

save_dir = {'h36m_vid': ROOT + '/h36m/crops',
            '3dpw_vid': ROOT + '/3dpw/crops',
            'bedlam_vid': ROOT + '/bedlam_30fps/crops',
            '3dpw_vid_test': ROOT + '/3dpw/crops_test',
            'emdb_1': ROOT + '/emdb/crops_1'}

warp_save_dir = {'3dpw_vid': ROOT + '/3dpw/warped_test', 'emdb_1': ROOT + '/emdb/warped_1'}
for ds in ds_list:
    print(f'Processing (crop) {ds} ...')

    # DATASET
    db = BaseDataset(ds, is_train=True, crop_size=256)
    
    for i in range(1):
        item = db[0]
        pre_j2d_img = item['pre_j2d_img_full']
        cv2.imwrite('pre_j2d_img.jpg', pre_j2d_img)
    
    
    #loader = DataLoader(db, batch_size=64, num_workers=15, shuffle=False)
    # imgdir = save_dir[ds]
    # warp_imgdir = warp_save_dir[ds]
    # os.makedirs(imgdir, exist_ok=True)
    # os.makedirs(warp_imgdir, exist_ok=True)
    # c = 0
    # warpc = 0
    # for i, batch in enumerate(tqdm(loader)):
    #     images = batch['img'].numpy()
    #     warp_imges = batch['warp_img'].numpy()
    #     for img in images:
    #         cv2.imwrite(f'{imgdir}/{c:08d}.jpg', img)
    #         c += 1
    #     for img in warp_imges:
    #         cv2.imwrite(f'{warp_imgdir}/{warpc:08d}.jpg', img)
    #         warpc += 1
