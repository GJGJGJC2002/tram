#!/usr/bin/env python3
"""Check if the pipeline intermediate data has bboxes stored."""
import pickle, torch, numpy as np, gzip, os

pkl_path = 'results/promptbase_video_depthrefine_emdb1/intermediate/44_indoor_rom/iter0_gvhmr_preprocessing_data.pkl'
print(f"File: {pkl_path}")
print(f"File size: {os.path.getsize(pkl_path) / 1024 / 1024:.1f} MB")
print(f"Modified: {os.path.getmtime(pkl_path)}")

with gzip.open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print(f"\nTop-level keys: {list(data.keys())}")
print(f"Metadata keys: {list(data['metadata'].keys())}")

# Check bboxes
if 'bboxes' in data:
    bb = data['bboxes']
    print(f"\nbboxes: {type(bb)}")
    if bb is not None:
        if isinstance(bb, np.ndarray):
            print(f"bboxes shape: {bb.shape}, dtype: {bb.dtype}")
            print(f"bboxes[0]: {bb[0]}")
        elif isinstance(bb, torch.Tensor):
            print(f"bboxes shape: {bb.shape}")
            print(f"bboxes[0]: {bb[0]}")
    else:
        print("bboxes is None")
else:
    print("\nNo 'bboxes' key in data")

# Check all possible substep cache dirs
import glob
substep_dirs = glob.glob('**/substep_cache/*vitpose*', recursive=True)
print(f"\nAll vitpose caches found:")
for p in sorted(substep_dirs):
    pt = torch.load(p, map_location='cpu', weights_only=False)
    print(f"  {p}: shape={pt.shape}, nose0=({pt[0,0,0]:.1f}, {pt[0,0,1]:.1f})")

# Also check if there's a substep cache under results/
substep_dirs2 = glob.glob('results/**/substep_cache/*vitpose*', recursive=True)
print(f"\nVitpose caches under results/:")
for p in sorted(substep_dirs2):
    pt = torch.load(p, map_location='cpu', weights_only=False)
    print(f"  {p}: shape={pt.shape}, nose0=({pt[0,0,0]:.1f}, {pt[0,0,1]:.1f})")

# Check which substep_cache_dir the pipeline would use
# From hpe.py line 344: substep_dir = os.path.join(output_dir, 'intermediate', seq_name)
# The prompthmr backend stores substep cache at self.substep_cache_dir / substep_cache / {hash}_vitpose.pt
# So the pipeline should look for the cache at:
# results/promptbase_video_depthrefine_emdb1/intermediate/44_indoor_rom/substep_cache/
substep_path = 'results/promptbase_video_depthrefine_emdb1/intermediate/44_indoor_rom/substep_cache/'
if os.path.exists(substep_path):
    print(f"\n{substep_path} exists!")
    for f in os.listdir(substep_path):
        print(f"  {f}")
else:
    print(f"\n{substep_path} does NOT exist")
    
# Check the GVHMR results directory
gvhmr_substep = 'thirdparty/GVHMR/results/promptbase_video_depthrefine_emdb1/intermediate/44_indoor_rom/substep_cache/'
if os.path.exists(gvhmr_substep):
    print(f"\n{gvhmr_substep} exists!")
    for f in os.listdir(gvhmr_substep):
        fpath = os.path.join(gvhmr_substep, f)
        print(f"  {f} (size: {os.path.getsize(fpath)/1024:.1f} KB)")
