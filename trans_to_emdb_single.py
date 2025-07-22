import joblib
import numpy as np
from pathlib import Path
import os
from glob import glob
import torch
from pytorch3d.transforms import matrix_to_axis_angle

from trans_to_emdb import process_wham_output

dir = "/home/gejunchen/Work/2024-11/Baseline/tram/results/P0_00_mvs_a_S"
output_dir = "/home/gejunchen/Work/2024-11/Dataset/EMDB/P0/00_mvs_a/tram_small"
frames_dir = "/home/gejunchen/Work/2024-11/Dataset/EMDB/P0/00_mvs_a/images"

if not os.path.exists(frames_dir):
    print(f"{frames_dir} 没有图像文件")
    
dirs = os.listdir(frames_dir)
flames_num = len(dirs)  # 图像帧数
print(f"输入文件: {dir}")
if not os.path.exists(os.path.join(dir, "camera.npy")):
    print(f"{dir} 没有 camera.npy 文件")

print(f"输出目录: {output_dir}")
# 处理数据
process_wham_output(dir, output_dir, flames_num)