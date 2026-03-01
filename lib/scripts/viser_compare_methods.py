"""
交互式多方法运动序列对比可视化（基于 viser）

在浏览器中实时对比不同 HMR 方法的全局 SMPL 运动序列。
支持实时旋转、缩放、时间轴滑动、方法显隐切换等交互操作。

用法:
    # 对比 GVHMR 和 PromptHMR（GVHMR 格式，使用 SMPL-X supermotion）
    python lib/scripts/viser_compare_methods.py \
        --sequence 29_outdoor_stairs_up \
        --methods \
            GVHMR:results/gvhmr_base_warmstart \
            PromptHMR:results/promptbase_video_warmstart_emdb2 \
        --port 8080

    # 加入 emdb_basic（TRAM/VIMO 格式，使用 c2w 转世界坐标），并指定 subsample
    python lib/scripts/viser_compare_methods.py \
        --sequence 29_outdoor_stairs_up \
        --methods \
            GVHMR:results/gvhmr_base_warmstart \
            PromptHMR:results/promptbase_video_warmstart_emdb2 \
            TRAM:results/emdb_basic \
        --subsample 2 \
        --port 8080

    # 显示 GT
    python lib/scripts/viser_compare_methods.py \
        --sequence 29_outdoor_stairs_up \
        --methods GVHMR:results/gvhmr_base_warmstart \
        --show-gt \
        --dataset-root datasets/EMDB \
        --port 8080
"""

import os
import sys
import time
import copy
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 添加项目路径
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GVHMR_ROOT = os.path.join(PROJECT_ROOT, 'thirdparty', 'GVHMR')
if GVHMR_ROOT not in sys.path:
    sys.path.insert(0, GVHMR_ROOT)


# ========== 颜色预设 ==========
METHOD_COLORS = [
    (100, 200, 130),  # 绿
    (80, 130, 230),   # 蓝
    (230, 180, 80),   # 黄
    (200, 100, 100),  # 红
    (180, 100, 220),  # 紫
    (100, 200, 200),  # 青
]
GT_COLOR = (160, 160, 160)  # 灰色


# ========== 1. SMPL 模型加载（单例） ==========

_smplx_model = None
_smplx2smpl = None
_smpl_faces = None
_J_regressor = None


def _ensure_smplx_model():
    """加载 SMPL-X supermotion 模型（与 GVHMR 渲染流程一致）"""
    global _smplx_model, _smplx2smpl, _smpl_faces, _J_regressor

    if _smplx_model is not None:
        return

    from hmr4d.utils.smplx_utils import make_smplx

    print("[Model] Loading SMPL-X (supermotion) + smplx2smpl...")
    _smplx_model = make_smplx("supermotion").cuda().eval()
    _smplx2smpl = torch.load(
        os.path.join(GVHMR_ROOT, "hmr4d/utils/body_model/smplx2smpl_sparse.pt")
    ).cuda()
    _smpl_faces = make_smplx("smpl").faces
    _J_regressor = torch.load(
        os.path.join(GVHMR_ROOT, "hmr4d/utils/body_model/smpl_neutral_J_regressor.pt")
    ).cuda()
    print("[Model] Loaded.")


def _get_smpl_faces():
    _ensure_smplx_model()
    return _smpl_faces


def _get_J_regressor():
    _ensure_smplx_model()
    return _J_regressor


# ========== 2. Vertices 生成 ==========

def generate_world_vertices_gvhmr(smpl_data: dict) -> torch.Tensor:
    """
    从 GVHMR 格式的 smpl.npz 生成世界坐标系 SMPL vertices。
    使用 SMPL-X supermotion + smplx2smpl（与渲染流程完全一致）。

    Returns: vertices (F, V, 3) tensor on CPU
    """
    _ensure_smplx_model()

    with torch.no_grad():
        out = _smplx_model(
            global_orient=torch.from_numpy(smpl_data['global_orient_w']).float().cuda(),
            body_pose=torch.from_numpy(smpl_data['body_pose_aa']).float().cuda(),
            betas=torch.from_numpy(smpl_data['betas']).float().cuda(),
            transl=torch.from_numpy(smpl_data['global_trans']).float().cuda(),
        )
        verts = torch.stack([_smplx2smpl @ v for v in out.vertices])

    return verts.cpu()


def generate_world_vertices_via_c2w(smpl_data: dict, camera_data: dict) -> torch.Tensor:
    """
    通过 incam SMPL-X params + c2w 变换生成世界坐标系 vertices。
    这种方式比直接用 global_orient_w 更可靠，因为避免了不同 pipeline
    WorldTransform 坐标系约定差异导致的人体翻转问题。

    适用于：PromptHMR pipeline（WorldTransform 坐标系可能与 GVHMR 原生不同）、
    emdb_basic（TRAM/VIMO 格式）等。

    Returns: vertices (F, V, 3) tensor on CPU
    """
    _ensure_smplx_model()

    # 判断使用 SMPL-X 还是 SMPL
    has_gvhmr_fields = 'global_orient_c' in smpl_data and 'body_pose_aa' in smpl_data
    has_rotmat = 'rotmat' in smpl_data

    if has_gvhmr_fields:
        # GVHMR 格式的 incam 参数 -> SMPL-X supermotion
        with torch.no_grad():
            out = _smplx_model(
                global_orient=torch.from_numpy(smpl_data['global_orient_c']).float().cuda(),
                body_pose=torch.from_numpy(smpl_data['body_pose_aa']).float().cuda(),
                betas=torch.from_numpy(smpl_data['betas']).float().cuda(),
                transl=torch.from_numpy(smpl_data['trans']).float().cuda(),
            )
            verts_incam = torch.stack([_smplx2smpl @ v for v in out.vertices])
    elif has_rotmat:
        # TRAM/VIMO 格式 -> 标准 SMPL
        from hmr4d.utils.smplx_utils import make_smplx
        smpl_model = make_smplx("smpl").cuda().eval()
        rotmat = torch.from_numpy(smpl_data['rotmat']).float().cuda()
        with torch.no_grad():
            out = smpl_model(
                global_orient=rotmat[:, [0]],
                body_pose=rotmat[:, 1:],
                betas=torch.from_numpy(smpl_data['betas']).float().cuda(),
                transl=torch.from_numpy(smpl_data['trans']).float().cuda(),
                pose2rot=False,
            )
            verts_incam = out.vertices
        del smpl_model
        torch.cuda.empty_cache()
    else:
        raise ValueError("smpl.npz has neither global_orient_c nor rotmat")

    # c2w 变换
    R_c2w = torch.from_numpy(camera_data['world_R']).float().cuda()
    T_c2w = torch.from_numpy(camera_data['world_T']).float().cuda()

    verts_world = torch.einsum('fij,fvj->fvi', R_c2w, verts_incam) + T_c2w[:, None, :]

    return verts_world.cpu()


# ========== 3. 对齐处理 ==========

def ensure_y_up(verts: torch.Tensor) -> torch.Tensor:
    """
    检测并修正 Y 轴方向：如果头部 Y < 脚部 Y，做 180° X 轴翻转。
    使用前 5 帧的 joints 做判断，鲁棒性更好。

    SMPL joints: 0=pelvis, 15=head, 7/8=ankle(L/R)

    Args:
        verts: (F, V, 3) tensor

    Returns:
        verts: (F, V, 3) tensor, 确保 Y-up
    """
    J_regressor = _get_J_regressor()

    n_check = min(5, verts.shape[0])
    verts_check = verts[:n_check].cuda()
    joints_check = torch.einsum('jv,fvi->fji', J_regressor, verts_check)  # (n_check, J, 3)

    head_y = joints_check[:, 15, 1].mean().item()
    ankle_y = (joints_check[:, 7, 1].mean().item() + joints_check[:, 8, 1].mean().item()) / 2

    if head_y < ankle_y:
        print(f"  [Warning] Detected Y-down (head_y={head_y:.3f} < ankle_y={ankle_y:.3f}), flipping 180° around X")
        # 绕 X 轴翻转 180°: Y -> -Y, Z -> -Z
        verts = verts.clone()
        verts[:, :, 1] = -verts[:, :, 1]
        verts[:, :, 2] = -verts[:, :, 2]
    else:
        print(f"  [OK] Y-up confirmed (head_y={head_y:.3f} > ankle_y={ankle_y:.3f})")

    return verts


def move_to_start_point_face_z(verts: torch.Tensor) -> torch.Tensor:
    """
    将 vertices 归一化：XZ 原点、站在地面、面朝 Z 轴。
    与 GVHMR demo.py 中的实现完全一致。

    注意：调用前须确保 verts 已是 Y-up（可先调 ensure_y_up）。

    Args:
        verts: (F, V, 3) tensor

    Returns:
        verts: (F, V, 3) aligned tensor
    """
    from einops import einsum as einops_einsum
    from hmr4d.utils.geo_transform import compute_T_ayfz2ay, apply_T_on_points

    J_regressor = _get_J_regressor()
    verts = verts.clone().cuda()

    # 1. 位置归一化
    offset = einops_einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3,)
    offset[1] = verts[:, :, 1].min()
    verts = verts - offset

    # 2. 面朝 Z 轴旋转
    T_ay2ayfz = compute_T_ayfz2ay(
        einops_einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"),
        inverse=True
    )
    verts = apply_T_on_points(verts, T_ay2ayfz)

    return verts.cpu()


def compute_xz_bbox(verts: torch.Tensor) -> Tuple[float, float, float, float]:
    """
    计算 vertices 在 XZ 平面上的 bounding box。

    Returns:
        (x_min, x_max, z_min, z_max)
    """
    v = verts.reshape(-1, 3)
    return (v[:, 0].min().item(), v[:, 0].max().item(),
            v[:, 2].min().item(), v[:, 2].max().item())


def spread_methods_no_overlap(methods_data: dict, gap: float = 1.0):
    """
    沿 X 轴平移各方法的 vertices 和 trajectory，使它们的 XZ bounding box 不重叠。
    第一个方法不动，后续方法依次向 +X 方向平移。

    Args:
        methods_data: {name: {'vertices': (F,V,3) tensor, 'trajectory': (F,3) np, ...}}
        gap: 相邻方法之间的 X 间距（米）
    """
    names = list(methods_data.keys())
    if len(names) <= 1:
        return

    # 计算每个方法的 XZ bbox
    bboxes = {}
    for name in names:
        bboxes[name] = compute_xz_bbox(methods_data[name]['vertices'])

    # 第一个方法不动，后续方法依次排列
    x_cursor = bboxes[names[0]][1]  # 第一个方法的 x_max

    for i in range(1, len(names)):
        name = names[i]
        x_min_i, x_max_i, _, _ = bboxes[name]
        width_i = x_max_i - x_min_i

        # 把第 i 个方法的 x_min 放到 x_cursor + gap
        shift_x = (x_cursor + gap) - x_min_i

        print(f"  Shifting {name} by X={shift_x:.2f}m (gap={gap:.1f}m)")
        methods_data[name]['vertices'][:, :, 0] += shift_x
        methods_data[name]['trajectory'][:, 0] += shift_x

        x_cursor = x_cursor + gap + width_i


# ========== 4. 数据加载 ==========

def detect_format(smpl_data: dict) -> str:
    """检测 smpl.npz 的数据格式"""
    if 'global_orient_w' in smpl_data and 'body_pose_aa' in smpl_data:
        return 'gvhmr'
    elif 'rotmat' in smpl_data:
        return 'basic'
    else:
        raise ValueError(f"Unknown smpl.npz format, keys: {list(smpl_data.keys())}")


def load_method_data(result_dir: str, sequence: str) -> Tuple[torch.Tensor, np.ndarray]:
    """
    加载一个方法的结果，返回世界坐标系 vertices + root trajectory。

    路由策略：
    - 优先使用 c2w 路径（incam params + camera.npz 的 world_R/world_T），
      因为这种方式避免了不同 pipeline WorldTransform 坐标系约定差异。
    - 仅当没有 camera.npz 或缺少 world_R 时，才回退到直接使用
      global_orient_w（仅适用于 GVHMR 原生格式，其坐标系天然正确）。

    Returns:
        vertices: (F, V, 3) tensor
        trajectory: (F, 3) numpy array
    """
    smpl_path = os.path.join(result_dir, sequence, 'smpl.npz')
    if not os.path.exists(smpl_path):
        raise FileNotFoundError(f"smpl.npz not found: {smpl_path}")

    smpl_data = dict(np.load(smpl_path))
    fmt = detect_format(smpl_data)

    # 检查是否有 camera.npz 及 c2w 信息
    cam_path = os.path.join(result_dir, sequence, 'camera.npz')
    has_c2w = False
    camera_data = None
    if os.path.exists(cam_path):
        camera_data = dict(np.load(cam_path))
        has_c2w = 'world_R' in camera_data and 'world_T' in camera_data

    if has_c2w:
        # 优先 c2w 路径 —— 适用于所有方法（PromptHMR, TRAM, VIMO 等）
        print(f"  Using c2w path (incam params + camera.npz)")
        verts = generate_world_vertices_via_c2w(smpl_data, camera_data)
        # trajectory: 用 c2w 变换 incam trans
        trans_key = 'trans' if 'trans' in smpl_data else 'global_trans'
        incam_trans = torch.from_numpy(smpl_data[trans_key]).float().cuda()
        R_c2w = torch.from_numpy(camera_data['world_R']).float().cuda()
        T_c2w = torch.from_numpy(camera_data['world_T']).float().cuda()
        traj = (torch.einsum('fij,fj->fi', R_c2w, incam_trans) + T_c2w).cpu().numpy()
    elif fmt == 'gvhmr':
        # 回退：GVHMR 原生格式，直接用 global_orient_w（坐标系天然正确）
        print(f"  Using GVHMR native path (global_orient_w)")
        verts = generate_world_vertices_gvhmr(smpl_data)
        traj = smpl_data['global_trans']  # (F, 3)
    else:
        raise FileNotFoundError(
            f"camera.npz with world_R/world_T required for basic format: {cam_path}"
        )

    return verts, traj


def load_gt_data(dataset_root: str, sequence: str, split: int = 2) -> Tuple[torch.Tensor, np.ndarray]:
    """
    从 EMDB annotations 加载 GT SMPL 参数并生成世界坐标系 vertices。

    Returns:
        vertices: (F, V, 3) tensor
        trajectory: (F, 3) numpy array
    """
    import pickle

    # 找到 annotations 文件
    # sequence 格式: "29_outdoor_stairs_up" -> "P3/29_outdoor_stairs_up"
    seq_num = sequence.split('_')[0]

    # 搜索 annotations: 遍历 dataset_root 下的 P* 目录
    ann_path = None
    for person_dir in sorted(os.listdir(dataset_root)):
        person_path = os.path.join(dataset_root, person_dir)
        if not os.path.isdir(person_path) or not person_dir.startswith('P'):
            continue
        candidate = os.path.join(person_path, sequence, f"{person_dir}_{sequence}_data.pkl")
        if os.path.exists(candidate):
            ann_path = candidate
            break

    if ann_path is None:
        raise FileNotFoundError(
            f"GT annotations not found for sequence '{sequence}' "
            f"under {dataset_root}/P*/{sequence}/"
        )

    print(f"[GT] Loading annotations from {ann_path}")
    with open(ann_path, 'rb') as f:
        ann = pickle.load(f)

    # 提取 GT SMPL 参数（世界坐标系）
    poses_root = ann['smpl']['poses_root']    # (N, 3) axis-angle
    poses_body = ann['smpl']['poses_body']    # (N, 23, 3) axis-angle
    betas_single = ann['smpl']['betas']       # (10,)
    trans_world = ann['smpl']['trans']         # (N, 3)

    N = len(poses_root)
    betas = np.repeat(betas_single.reshape(1, -1), N, axis=0)  # (N, 10)
    poses_body_flat = poses_body.reshape(N, -1)  # (N, 69)

    # GT 使用 SMPL-X supermotion 也没问题（body_pose 只取前 21 joints）
    # 但为保持一致，直接用 SMPL model
    from hmr4d.utils.smplx_utils import make_smplx
    smpl_model = make_smplx("smpl").cuda().eval()

    with torch.no_grad():
        out = smpl_model(
            global_orient=torch.from_numpy(poses_root).float().cuda(),
            body_pose=torch.from_numpy(poses_body_flat).float().cuda(),
            betas=torch.from_numpy(betas).float().cuda(),
            transl=torch.from_numpy(trans_world).float().cuda(),
            pose2rot=True,
        )
        verts = out.vertices.cpu()

    del smpl_model
    torch.cuda.empty_cache()

    return verts, trans_world


# ========== 5. 地面 ==========

def build_ground_geometry(all_verts_list: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据所有方法的 vertices 计算地面几何参数，生成棋盘格。

    Returns:
        ground_verts: (N, 3) float32
        ground_faces: (M, 3) int32
    """
    from lib.vis.tools import checkerboard_geometry

    # 合并所有 vertices 计算范围
    all_verts = np.concatenate([v.numpy().reshape(-1, 3) for v in all_verts_list], axis=0)
    xz = all_verts[:, [0, 2]]  # x 和 z
    xz_min = xz.min(0)
    xz_max = xz.max(0)

    cx = (xz_min[0] + xz_max[0]) / 2
    cz = (xz_min[1] + xz_max[1]) / 2
    scale = max(xz_max[0] - xz_min[0], xz_max[1] - xz_min[1]) * 1.5
    scale = max(scale, 6.0)  # 最小尺寸

    y_min = all_verts[:, 1].min()

    gv, gf, gc, _ = checkerboard_geometry(
        length=scale, c1=cx, c2=cz, up="y",
        color0=[0.85, 0.9, 0.9], color1=[0.65, 0.7, 0.7],
        tile_width=0.5,
    )
    # 将地面放在 y_min 处
    gv[:, 1] = y_min

    return gv.astype(np.float32), gf.astype(np.int32)


# ========== 6. Viser 场景 ==========

def build_viser_scene(
    server,
    methods_data: dict,
    faces: np.ndarray,
    subsample: int = 1,
):
    """
    构建 viser 场景：frame nodes + 方法 mesh + 轨迹线 + 地面

    Args:
        methods_data: {name: {'vertices': (F,V,3) tensor, 'color': (r,g,b), 'trajectory': (F,3) np}}
        faces: SMPL faces array
        subsample: 帧采样间隔

    Returns:
        frame_nodes, method_mesh_groups, sampled_indices, scene_handles
    """
    import viser

    first_method = list(methods_data.values())[0]
    num_frames = first_method['vertices'].shape[0]
    sampled_indices = list(range(0, num_frames, subsample))

    # 地面：实体底色 + wireframe 网格线叠加
    all_verts_list = [d['vertices'][::subsample] for d in methods_data.values()]
    gv, gf = build_ground_geometry(all_verts_list)
    ground_handle = server.scene.add_mesh_simple(
        "/ground/solid",
        vertices=gv,
        faces=gf,
        flat_shading=True,
        wireframe=False,
        color=(210, 218, 218),
        side="double",
    )
    ground_wire_handle = server.scene.add_mesh_simple(
        "/ground/wire",
        vertices=gv,
        faces=gf,
        flat_shading=True,
        wireframe=True,
        color=(140, 150, 150),
        side="double",
    )

    # Frame 根节点
    import viser.transforms as vtf
    server.scene.add_frame(
        "/frames",
        wxyz=vtf.SO3.exp(np.array([0.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )

    frame_nodes = []
    method_mesh_groups = {name: [] for name in methods_data}

    faces_int = faces.astype(np.int32)

    print(f"Building scene: {len(sampled_indices)} frames, {len(methods_data)} methods...")
    for idx, fi in enumerate(tqdm(sampled_indices, desc="Building viser scene")):
        # 每帧一个 frame node
        frame_node = server.scene.add_frame(f"/frames/t{idx}", show_axes=False)
        frame_nodes.append(frame_node)

        # 每个方法一个 mesh
        for method_name, mdata in methods_data.items():
            verts_fi = mdata['vertices'][fi].numpy().astype(np.float32)
            color = mdata['color']

            mesh = server.scene.add_mesh_simple(
                f"/frames/t{idx}/{method_name}/mesh",
                vertices=verts_fi,
                faces=faces_int,
                flat_shading=False,
                wireframe=False,
                color=color,
            )
            method_mesh_groups[method_name].append(mesh)

    # 轨迹线
    trajectory_handles = {}
    for method_name, mdata in methods_data.items():
        traj = mdata['trajectory'][::subsample]
        if len(traj) > 1:
            points = np.array(traj, dtype=np.float32)
            traj_handle = server.scene.add_spline_catmull_rom(
                f"/trajectory/{method_name}",
                positions=points,
                color=mdata['color'],
                line_width=3.0,
            )
            trajectory_handles[method_name] = traj_handle

    # 初始化：只显示第 0 帧
    for i, fn in enumerate(frame_nodes):
        fn.visible = (i == 0)

    scene_handles = {
        'ground': ground_handle,
        'ground_wire': ground_wire_handle,
        'trajectories': trajectory_handles,
    }

    return frame_nodes, method_mesh_groups, sampled_indices, scene_handles


# ========== 7. GUI ==========

def setup_gui(server, methods_data, frame_nodes, method_mesh_groups, num_sampled, scene_handles=None):
    """设置 viser GUI 控件"""

    # --- 播放控件 ---
    gui_timestep = server.gui.add_slider(
        "Timestep", min=0, max=num_sampled - 1, step=1, initial_value=0,
        disabled=True,
    )
    gui_playing = server.gui.add_checkbox("Playing", True)
    gui_fps = server.gui.add_slider(
        "FPS", min=1, max=60, step=0.1, initial_value=25
    )
    gui_fps_options = server.gui.add_button_group(
        "FPS options", ("10", "20", "30", "60")
    )
    gui_next = server.gui.add_button("Next Frame", disabled=True)
    gui_prev = server.gui.add_button("Prev Frame", disabled=True)

    # --- 方法显隐 ---
    server.gui.add_markdown("---")
    server.gui.add_markdown("**Methods**")
    method_toggles = {}
    for method_name in methods_data:
        color = methods_data[method_name]['color']
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        toggle = server.gui.add_checkbox(
            f"Show {method_name} ({color_hex})", True
        )
        method_toggles[method_name] = toggle

    # --- 场景控件 ---
    server.gui.add_markdown("---")
    server.gui.add_markdown("**Scene**")
    gui_show_ground = server.gui.add_checkbox("Show Ground", True)
    gui_show_traj = server.gui.add_checkbox("Show Trajectory", True)
    gui_show_axes = server.gui.add_checkbox("Show World Axes", True)
    gui_show_all = server.gui.add_checkbox("Show All Frames", False)

    # --- 回调 ---
    prev_timestep = [0]
    show_all_state = [False]

    @gui_timestep.on_update
    def _(_):
        if show_all_state[0]:
            return
        cur = gui_timestep.value
        prev = prev_timestep[0]
        if cur != prev:
            with server.atomic():
                frame_nodes[cur].visible = True
                frame_nodes[prev].visible = False
            prev_timestep[0] = cur
            server.flush()

    @gui_next.on_click
    def _(_):
        gui_timestep.value = (gui_timestep.value + 1) % num_sampled

    @gui_prev.on_click
    def _(_):
        gui_timestep.value = (gui_timestep.value - 1) % num_sampled

    @gui_playing.on_update
    def _(_):
        gui_timestep.disabled = gui_playing.value
        gui_next.disabled = gui_playing.value
        gui_prev.disabled = gui_playing.value

    @gui_fps_options.on_click
    def _(_):
        gui_fps.value = int(gui_fps_options.value)

    # 方法显隐回调（闭包捕获）
    for method_name, toggle in method_toggles.items():
        meshes = method_mesh_groups[method_name]

        def _make_toggle_cb(meshes, toggle):
            def cb(_):
                for m in meshes:
                    m.visible = toggle.value
            return cb

        toggle.on_update(_make_toggle_cb(meshes, toggle))

    @gui_show_ground.on_update
    def _(_):
        if scene_handles:
            if 'ground' in scene_handles:
                scene_handles['ground'].visible = gui_show_ground.value
            if 'ground_wire' in scene_handles:
                scene_handles['ground_wire'].visible = gui_show_ground.value

    @gui_show_all.on_update
    def _(_):
        show_all = gui_show_all.value
        show_all_state[0] = show_all
        with server.atomic():
            if show_all:
                for fn in frame_nodes:
                    fn.visible = True
                gui_playing.value = False
            else:
                cur = gui_timestep.value
                for i, fn in enumerate(frame_nodes):
                    fn.visible = (i == cur)
                prev_timestep[0] = cur
        server.flush()

    @gui_show_traj.on_update
    def _(_):
        if scene_handles and 'trajectories' in scene_handles:
            for handle in scene_handles['trajectories'].values():
                handle.visible = gui_show_traj.value

    @gui_show_axes.on_update
    def _(_):
        server.scene.world_axes.visible = gui_show_axes.value

    return gui_playing, gui_timestep, gui_fps, gui_show_all


# ========== 8. 入口 ==========

@dataclass
class ViserCompareConfig:
    """交互式多方法对比可视化配置"""

    sequence: str = "29_outdoor_stairs_up"
    """EMDB 序列名"""

    methods: List[str] = field(default_factory=lambda: [
        "GVHMR:results/gvhmr_base_warmstart",
        "PromptHMR:results/promptbase_video_warmstart_emdb2",
    ])
    """方法列表，格式: Name:result_dir"""

    show_gt: bool = False
    """是否显示 Ground Truth"""

    dataset_root: str = "datasets/EMDB"
    """EMDB 数据集根目录（显示 GT 时需要）"""

    gt_split: int = 2
    """EMDB split (1 or 2)"""

    subsample: int = 1
    """帧采样间隔（大序列建议设为 2-5 以加速加载）"""

    start_frame: int = 0
    """起始帧索引（截取片段起点，0-based）"""

    end_frame: int = -1
    """结束帧索引（截取片段终点，-1 表示到最后一帧）"""

    align: bool = True
    """是否对所有方法做 ensure_y_up + move_to_start_point_face_z 对齐"""

    spread: bool = True
    """是否沿 X 轴平移使各方法 bounding box 不重叠"""

    spread_gap: float = 1.0
    """方法间 X 轴间距（米），仅 spread=True 时生效"""

    port: int = 8080
    """viser 服务端口"""


def main(cfg: ViserCompareConfig):
    import viser
    import viser.transforms as vtf

    # 1. 加载各方法数据
    print(f"\n{'='*60}")
    print(f"Sequence: {cfg.sequence}")
    print(f"Methods: {cfg.methods}")
    print(f"Subsample: {cfg.subsample}, Align: {cfg.align}, Spread: {cfg.spread}")
    print(f"Frame range: [{cfg.start_frame}, {cfg.end_frame}]")
    print(f"{'='*60}\n")

    methods_data = {}
    for i, method_spec in enumerate(cfg.methods):
        parts = method_spec.split(':')
        if len(parts) != 2:
            raise ValueError(
                f"Invalid method spec '{method_spec}', expected 'Name:result_dir'"
            )
        name, result_dir = parts
        print(f"[{i+1}/{len(cfg.methods)}] Loading {name} from {result_dir}/{cfg.sequence}/...")

        verts, traj = load_method_data(result_dir, cfg.sequence)

        color = METHOD_COLORS[i % len(METHOD_COLORS)]
        methods_data[name] = {
            'vertices': verts,
            'trajectory': traj,
            'color': color,
        }
        print(f"  Loaded: {verts.shape[0]} frames, {verts.shape[1]} vertices")

    # GT（可选）
    if cfg.show_gt:
        print(f"\nLoading GT from {cfg.dataset_root}...")
        try:
            gt_verts, gt_traj = load_gt_data(
                cfg.dataset_root, cfg.sequence, cfg.gt_split
            )
            methods_data['GT'] = {
                'vertices': gt_verts,
                'trajectory': gt_traj,
                'color': GT_COLOR,
            }
            print(f"  GT loaded: {gt_verts.shape[0]} frames")
        except Exception as e:
            print(f"  Warning: Failed to load GT: {e}")

    # 帧数对齐检查
    frame_counts = {name: d['vertices'].shape[0] for name, d in methods_data.items()}
    print(f"\nFrame counts: {frame_counts}")
    min_frames = min(frame_counts.values())
    if len(set(frame_counts.values())) > 1:
        print(f"Warning: Frame counts differ, truncating to {min_frames}")
        for name in methods_data:
            methods_data[name]['vertices'] = methods_data[name]['vertices'][:min_frames]
            methods_data[name]['trajectory'] = methods_data[name]['trajectory'][:min_frames]

    # 帧范围截取（在对齐之前，使对齐基于截取片段的起点）
    sf = max(0, cfg.start_frame)
    ef = cfg.end_frame if cfg.end_frame > 0 else min_frames
    ef = min(ef, min_frames)
    if sf >= ef:
        raise ValueError(f"Invalid frame range: start_frame={sf} >= end_frame={ef}")
    if sf > 0 or ef < min_frames:
        print(f"Clipping frames [{sf}, {ef}) -> {ef - sf} frames")
        for name in methods_data:
            methods_data[name]['vertices'] = methods_data[name]['vertices'][sf:ef]
            methods_data[name]['trajectory'] = methods_data[name]['trajectory'][sf:ef]
        min_frames = ef - sf

    # 对齐（基于截取后的片段，起点归一化到原点）
    if cfg.align:
        print(f"\nAligning all methods...")
        for name in methods_data:
            print(f"  [{name}]")
            verts = methods_data[name]['vertices']
            verts = ensure_y_up(verts)
            print(f"  Aligning (move_to_start_point_face_z)...")
            verts = move_to_start_point_face_z(verts)
            J_reg = _get_J_regressor()
            joints = torch.einsum('jv,fvi->fji', J_reg.cpu(), verts)
            traj = joints[:, 0, :].numpy()
            methods_data[name]['vertices'] = verts
            methods_data[name]['trajectory'] = traj

    # 沿 X 轴平移使各方法不重叠
    if cfg.spread and len(methods_data) > 1:
        print(f"\nSpreading methods along X axis (gap={cfg.spread_gap}m)...")
        spread_methods_no_overlap(methods_data, gap=cfg.spread_gap)

    # 2. 启动 viser server
    faces = _get_smpl_faces()
    server = viser.ViserServer(port=cfg.port)
    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+y")

    # 3. 构建场景
    frame_nodes, method_mesh_groups, sampled, scene_handles = build_viser_scene(
        server, methods_data, faces, subsample=cfg.subsample
    )

    # 4. 设置 GUI
    gui_playing, gui_timestep, gui_fps, gui_show_all = setup_gui(
        server, methods_data, frame_nodes, method_mesh_groups, len(sampled),
        scene_handles=scene_handles
    )

    print(f"\n{'='*60}")
    print(f"Viser server running at: http://localhost:{cfg.port}")
    print(f"Methods: {list(methods_data.keys())}")
    print(f"Sequence: {cfg.sequence}")
    print(f"Total frames: {min_frames}, Sampled: {len(sampled)} (step={cfg.subsample})")
    print(f"{'='*60}")
    print(f"Open http://localhost:{cfg.port} in your browser")
    print(f"Press Ctrl+C to stop\n")

    # 5. 主循环
    try:
        while True:
            if gui_playing.value and not gui_show_all.value:
                gui_timestep.value = (gui_timestep.value + 1) % len(sampled)
            time.sleep(1.0 / gui_fps.value)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    import tyro
    cfg = tyro.cli(ViserCompareConfig)
    main(cfg)
