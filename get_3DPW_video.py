import cv2
import os

def get_video(image_folder, video_folder, video_name):
    
    fps = 30  # 帧率（与原视频一致）

    # 获取所有图片并排序
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()  # 确保按数字顺序排列
    # 读取第一张图片确定视频尺寸
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    # 创建视频写入对象（注意编码器选择）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 适用于.mp4
    print("saving at", video_folder)
    video = cv2.VideoWriter(video_folder, fourcc, fps, (width, height))

    # 逐帧写入
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)
    #保存视频
    video.release()

import os
import glob
import subprocess
#CUDA_VISIBLE_DEVICES=1
# 定义数据集目录
dataset_dir = "/home/gejunchen/Work/2024-7/Dataset/3DPW/imageFiles"
saving_dir = "/home/gejunchen/Work/2024-7/Dataset/3DPW/videos"
dir_names = os.listdir(dataset_dir)
print(dir_names) 

for dir_name in dir_names:
    #获得文件夹名称
    video_name = dir_name + ".mp4"
    print("Processing", dir_name, video_name)
    get_video(os.path.join(dataset_dir, dir_name), os.path.join(saving_dir, video_name), video_name)
    
    
