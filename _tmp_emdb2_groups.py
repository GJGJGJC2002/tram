import numpy as np, pickle as pkl, os
from glob import glob

dataset_path = 'datasets/EMDB'
roots = []
for p in range(10):
    folder = f'{dataset_path}/P{p}'
    if not os.path.exists(folder):
        continue
    roots.extend(sorted(glob(f'{folder}/*')))

results = []
for root in roots:
    annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
    if not os.path.exists(annfile):
        continue
    ann = pkl.load(open(annfile, 'rb'))
    if not ann.get('emdb2', False):
        continue
    seq = root.split('/')[-1]
    
    # 人体轨迹
    human_trans = ann['smpl']['trans']
    h_diffs = np.diff(human_trans, axis=0)
    human_traj = np.sum(np.linalg.norm(h_diffs, axis=1))
    
    # 人体位移（起点到终点）
    human_disp = np.linalg.norm(human_trans[-1] - human_trans[0])
    
    # 人体最大距离（离起点最远距离）
    human_max_dist = np.max(np.linalg.norm(human_trans - human_trans[0], axis=1))
    
    # 相机轨迹
    ext = ann['camera']['extrinsics']
    R_cw = ext[:, :3, :3]
    t_cw = ext[:, :3, 3]
    cam_pos = np.einsum('bij,bi->bj', R_cw.transpose(0, 2, 1), -t_cw)
    cam_traj = np.sum(np.linalg.norm(np.diff(cam_pos, axis=0), axis=1))
    cam_disp = np.linalg.norm(cam_pos[-1] - cam_pos[0])
    
    results.append((seq, ann['n_frames'], human_traj, human_disp, human_max_dist, cam_traj, cam_disp))

# 按人体轨迹排序
results.sort(key=lambda x: x[2])
print("=== 按人体轨迹长度排序 ===")
print(f"{'Seq':<35} {'Frames':>6} {'HTraj':>8} {'HDisp':>8} {'HMaxD':>8} {'CTraj':>8} {'CDisp':>8}")
print('=' * 90)
for seq, nf, ht, hd, hmd, ct, cd in results:
    print(f'{seq:<35} {nf:>6} {ht:>8.1f} {hd:>8.1f} {hmd:>8.1f} {ct:>8.1f} {cd:>8.1f}')

# 试不同阈值看能不能凑出 5/10/10
print("\n\n=== 寻找能产生 5/10/10 分组的阈值 ===")
all_ht = sorted([r[2] for r in results])
print(f"所有人体轨迹长度(升序): {[f'{x:.1f}' for x in all_ht]}")
print(f"第5和第6个之间: {all_ht[4]:.1f} ~ {all_ht[5]:.1f}")
print(f"第15和第16个之间: {all_ht[14]:.1f} ~ {all_ht[15]:.1f}")

# 如果按 displacement 排序
all_hd = sorted([r[3] for r in results])
print(f"\n所有人体位移(升序): {[f'{x:.1f}' for x in all_hd]}")
print(f"第5和第6个之间: {all_hd[4]:.1f} ~ {all_hd[5]:.1f}")
print(f"第15和第16个之间: {all_hd[14]:.1f} ~ {all_hd[15]:.1f}")

# 如果按 max distance 排序
all_hmd = sorted([r[4] for r in results])
print(f"\n所有人体最大距离(升序): {[f'{x:.1f}' for x in all_hmd]}")
print(f"第5和第6个之间: {all_hmd[4]:.1f} ~ {all_hmd[5]:.1f}")
print(f"第15和第16个之间: {all_hmd[14]:.1f} ~ {all_hmd[15]:.1f}")
