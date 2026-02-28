import pickle, torch, numpy as np, gzip

pkl_path = 'results/promptbase_video_depthrefine_emdb1/intermediate/44_indoor_rom/iter0_gvhmr_preprocessing_data.pkl'
with gzip.open(pkl_path, 'rb') as f:
    data = pickle.load(f)

md = data.get('metadata', {})
kp = md.get('vitpose_kp2d', None)
if kp is not None:
    if not isinstance(kp, torch.Tensor):
        kp = torch.tensor(kp)
    nose = kp[0, 0]
    print(f'kp2d frame 0 nose: x={nose[0].item():.1f}, y={nose[1].item():.1f}')

import cv2
img = cv2.imread(data['image_paths'][0])
print(f'Image: {img.shape[1]}w x {img.shape[0]}h')

# Check segmentation stage which should have bboxes
seg_path = 'results/promptbase_video_depthrefine_emdb1/intermediate/44_indoor_rom/iter0_human_segmentation_data.pkl'
with gzip.open(seg_path, 'rb') as f:
    seg_data = pickle.load(f)

print('\nSeg data keys:', [k for k in seg_data.keys() if k != 'metadata'])
if 'bboxes' in seg_data:
    bb = seg_data['bboxes']
    if isinstance(bb, torch.Tensor):
        bb = bb.numpy()
    elif isinstance(bb, list):
        bb = np.array(bb)
    print(f'Seg bboxes shape: {bb.shape}')
    print(f'Seg bboxes[0]: {bb[0]}')

# Check camera_estimation stage  
cam_path = 'results/promptbase_video_depthrefine_emdb1/intermediate/44_indoor_rom/iter0_camera_estimation_data.pkl'
with gzip.open(cam_path, 'rb') as f:
    cam_data = pickle.load(f)
print('\nCam data keys:', [k for k in cam_data.keys() if k != 'metadata'])
if 'bboxes' in cam_data:
    bb = cam_data['bboxes']
    if isinstance(bb, torch.Tensor):
        bb = bb.numpy()
    elif isinstance(bb, list):
        bb = np.array(bb)
    print(f'Cam bboxes shape: {bb.shape}')
    print(f'Cam bboxes[0]: {bb[0]}')
    print(f'Cam bboxes x range: [{bb[:,0].min():.1f}, {bb[:,2].max():.1f}]')
    print(f'Cam bboxes y range: [{bb[:,1].min():.1f}, {bb[:,3].max():.1f}]')
