import gzip
import pickle
import numpy as np

cache_path = "results/emdb_adjacent_render_tram/intermediate/20_outdoor_walk/iter0_camera_estimation_data.pkl"

try:
    with gzip.open(cache_path, 'rb') as f:
        data = pickle.load(f)
except:
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)

print(f"Keys in cache: {data.keys()}")
print(f"Number of frames (image_paths): {len(data.get('image_paths', []))}")

if 'bboxes' in data and data['bboxes'] is not None:
    bboxes = data['bboxes']
    print(f"\nBboxes shape: {bboxes.shape}")
    print(f"Frame 1601 bbox: {bboxes[1601]}")
    
    # 检查无效 bbox
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    w, h = x2-x1, y2-y1
    invalid = (w <= 0) | (h <= 0)
    print(f"Invalid boxes: {np.where(invalid)[0][:20]}")  # 只显示前 20 个
    print(f"Total invalid: {invalid.sum()}")
    
    # 检查 NaN
    has_nan = np.isnan(bboxes).any(axis=1)
    print(f"Frames with NaN: {np.where(has_nan)[0][:20]}")
    print(f"Total with NaN: {has_nan.sum()}")
