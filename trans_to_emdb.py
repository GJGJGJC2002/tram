import joblib
import numpy as np
from pathlib import Path
import os
from glob import glob
import torch
from pytorch3d.transforms import matrix_to_axis_angle


def process_wham_output(seq_folder, output_dir, flames_num):
    # 加载 tram 输出数据
    hps_folder = f'{seq_folder}/hps'
    hps_files = sorted(glob(f'{hps_folder}/*.npy'))
    pred_cam = np.load(f'{seq_folder}/camera.npy', allow_pickle=True).item()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    np.save(output_dir / 'camera_MD.npy', pred_cam)
    #print("pred_cam", pred_cam.keys())
    
    img_focal = pred_cam['img_focal'].item()
    pred_cam_R = torch.tensor(pred_cam['pred_cam_R'])
    pred_cam_T = torch.tensor(pred_cam['pred_cam_T'])
    #print("pred_cam_R", pred_cam_R.shape)
    #print("pred_cam_T", pred_cam_T.shape)
    max_track = len(hps_files)
    #print("find max_track", max_track)
    # 创建字典存储每一帧的数据
    frame_data = {}
    for i in range(max_track):
        hps_file = hps_files[i]
        pred_smpl = np.load(hps_file, allow_pickle=True).item()
        #print(pred_smpl.keys())
        #print(pred_smpl['pred_pose'].shape) #(N, 144)
        pred_rotmat = pred_smpl['pred_rotmat']
        pred_shape = pred_smpl['pred_shape']
        pred_trans = pred_smpl['pred_trans']
        frame = pred_smpl['frame']
        frame_length = len(frame)

        print(pred_rotmat.shape, pred_shape.shape, pred_trans.shape, frame.shape)

        axis_angle_pose = matrix_to_axis_angle(pred_rotmat)
        flat_pose = axis_angle_pose.reshape(frame_length, 72)
        #print("flat_pose", flat_pose.shape)
        #取出frame的pred_cam_R, pred_cam_T
        cam_R = pred_cam_R[frame]
        cam_T = pred_cam_T[frame]
        #global_trans = camera_to_global(cam_R, cam_T, pred_trans)
        #global_trans = global_trans.squeeze(1)
        #print("get global point", global_trans.shape)
        
        mean_shape = pred_shape.mean(dim=0, keepdim=True)
        pred_shape = mean_shape.repeat(len(pred_shape), 1)
        
        #转化为np数组
        pred_shape = pred_shape.cpu().numpy()
        pred_pose = flat_pose.cpu().numpy()
        pred_trans = pred_trans.squeeze(1).cpu().numpy()
        frame = frame.cpu().numpy()
        for j in range(frame_length):
            frame_id = frame[j]
            pose = pred_pose[j]            # (72,)
            tran = pred_trans[j]
            beta = pred_shape[j]              # (10,)
            # 如果字典里已存在此帧，则替换为最新的结果
            frame_data[frame_id] = (pose, tran, beta)
            #print("write in", frame_id)
        
    # 按帧 ID 排序
    sorted_frame_ids = sorted(frame_data.keys())
    #print("sorted_frame_ids", sorted_frame_ids)
    # 创建填充后的帧数据
    pose_hat = []
    trans_hat = []
    betas_hat = []

    for frame_id in range(flames_num):
        if frame_id in frame_data:
            pose, tran, beta = frame_data[frame_id]
            #print(f"frame_id: {frame_id}, pose: {pose.shape}, tran: {tran.shape}, betas: {beta.shape}")
        else:
            # 填充空帧
            pose = np.zeros(72)  # (72,)
            tran = np.zeros(3)   # (3,)
            beta = np.zeros(10)  # (10,)
        
        pose_hat.append(pose)
        trans_hat.append(tran)
        betas_hat.append(beta)

    pose_hat = np.array(pose_hat)  # (N, 72)
    trans_hat = np.array(trans_hat)  # (N, 3)
    shape_hat = np.array(betas_hat)  # (N, 10)

    # 创建输出目录

    
    # 保存结果
    np.save(output_dir / 'pose_hat.npy', pose_hat)
    np.save(output_dir / 'shape_hat.npy', shape_hat)
    np.save(output_dir / 'local_trans_hat.npy', trans_hat)
    
    print(f"结果已保存至 {output_dir}")

if __name__ == "__main__":
    # 获取所有需要转换的文件夹，/home/gejunchen/Work/2024-10/Baseline/WHAM/output/demo/“下的以P开头的文件夹名称
    dirs = Path("/home/gejunchen/Work/2024-11/Baseline/tram/tram_results").glob("P*")
    for dir in dirs:
        #print(f"处理文件夹: {dir}")
        if dir.name[-1] != "I":
            continue
        if dir == "/home/gejunchen/Work/2024-11/Baseline/tram/results/P1_13_outdoor_long_walk_I":
            continue
        # 获取P开头的文件夹名称
        Px = dir.name[0:2]
        name = dir.name[3:-2]
        
        # 拼接输出目录路径
        if not os.path.exists(str(Path("/home/gejunchen/Work/2024-11/Dataset/EMDB") / Px / name)):
            print(f"{Px} {name} 没有目录")
            continue
        
        output_dir = str(Path("/home/gejunchen/Work/2024-11/Dataset/EMDB") / Px / name / "tram_emdb")
        frames_dir = os.path.join("/home/gejunchen/Work/2024-11/Dataset/EMDB", Px, name, "images")
        
        if not os.path.exists(frames_dir):
            print(f"{frames_dir} 没有图像文件")
            continue
        dirs = os.listdir(frames_dir)
        flames_num = len(dirs)  # 图像帧数
        print(f"输入文件: {dir}")
        if not os.path.exists(os.path.join(dir, "camera.npy")):
            print(f"{dir} 没有 camera.npy 文件")
            continue
        print(f"输出目录: {output_dir}")
        # 处理数据
        process_wham_output(dir, output_dir, flames_num)
        #break
