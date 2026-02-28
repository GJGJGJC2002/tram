"""Debug: check emdb_basic data format"""
import numpy as np, torch, sys
sys.path.insert(0, '.')

s = np.load('results/emdb_basic/29_outdoor_stairs_up/smpl.npz')
c = np.load('results/emdb_basic/29_outdoor_stairs_up/camera.npz')

print("=== emdb_basic smpl.npz ===")
# rotmat is in CAMERA space (from VIMO HMR inference)
# trans is in CAMERA space
# global_trans is world-space (from WorldTransform: R_c2w @ incam + T_c2w with pelvis offset)
# BUT: there's no global_orient_w in emdb_basic -> rotmat is still in camera space!

# So generating world vertices from rotmat+trans gives CAMERA-space vertices
# Then c2w transform should give world vertices
# But wait - does global_trans already account for the pelvis offset?

from lib.models.smpl import SMPL
smpl = SMPL().cuda().eval()

rotmat = torch.from_numpy(s['rotmat']).float().cuda()
betas = torch.from_numpy(s['betas']).float().cuda()
trans = torch.from_numpy(s['trans']).float().cuda()  # incam

# Incam vertices  
with torch.no_grad():
    out = smpl(global_orient=rotmat[:1, [0]], body_pose=rotmat[:1, 1:],
               betas=betas[:1], transl=trans[:1], pose2rot=False, default_smpl=True)
    v_incam = out.vertices[0].cpu().numpy()

print(f"incam verts Y: min={v_incam[:,1].min():.3f}, max={v_incam[:,1].max():.3f}")
print(f"incam verts center: {v_incam.mean(0)}")

# Now try c2w  
R_c2w = torch.from_numpy(c['world_R'][:1]).float().cuda()
T_c2w = torch.from_numpy(c['world_T'][:1]).float().cuda()
v_world = (R_c2w[0] @ out.vertices[0].T).T + T_c2w[0]
v_world = v_world.cpu().numpy()
print(f"\nworld verts (via c2w) Y: min={v_world[:,1].min():.3f}, max={v_world[:,1].max():.3f}")
print(f"world verts center: {v_world.mean(0)}")

# Compare: GVHMR's vertices for same sequence
sys.path.insert(0, 'thirdparty/GVHMR')
from hmr4d.utils.smplx_utils import make_smplx
s2 = dict(np.load('results/gvhmr_base_warmstart/29_outdoor_stairs_up/smpl.npz'))
smplx_model = make_smplx("supermotion").cuda().eval()
smplx2smpl = torch.load("thirdparty/GVHMR/hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()

with torch.no_grad():
    out2 = smplx_model(
        global_orient=torch.from_numpy(s2['global_orient_w'][:1]).float().cuda(),
        body_pose=torch.from_numpy(s2['body_pose_aa'][:1]).float().cuda(),
        betas=torch.from_numpy(s2['betas'][:1]).float().cuda(),
        transl=torch.from_numpy(s2['global_trans'][:1]).float().cuda(),
    )
    v_gvhmr = (smplx2smpl @ out2.vertices[0]).cpu().numpy()

print(f"\nGVHMR verts Y: min={v_gvhmr[:,1].min():.3f}, max={v_gvhmr[:,1].max():.3f}")
print(f"GVHMR verts center: {v_gvhmr.mean(0)}")

# The question: in viser with Y-up, negative Y means upside down?
# Let's check the root joint Y  
print(f"\nBasic global_trans[0]: {s['global_trans'][0]}")
print(f"GVHMR global_trans[0]: {s2['global_trans'][0]}")
