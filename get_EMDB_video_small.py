import cv2
import os

def get_video(folder_path, folder_name):
    # 配置参数
    image_folder = os.path.join(folder_path, 'images')  # 图片文件夹路径
    video_name = folder_name + '.mp4'  # 输出视频文件名
    fps = 30  # 帧率（与原视频一致）

    # 获取所有图片并排序
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()  # 确保按数字顺序排列

    # 创建视频写入对象（注意编码器选择）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 适用于.mp4
    print("saving at", os.path.join(folder_path, video_name))
    video = cv2.VideoWriter(os.path.join(folder_path, video_name), fourcc, fps, (224, 224))

    # 逐帧写入
    for image in images:
        img_path = os.path.join(image_folder, image)
        img = cv2.imread(img_path)
        print(img.shape)
        W, H, _ = img.shape
        if W > H:
            tH = 224
            tW = int(W * 224 / H)
        else:
            tW = 224
            tH = int(H * 224 / W)
        img = cv2.resize(img, (tH, tW))
        cx, cy = tW//2, tH//2
        half = min(cx, cy)
        img = img[cx-half:cx+half, cy-half:cy+half, :]
        video.write(img)
    #保存视频
    video.release()

import os
import glob
import subprocess
#CUDA_VISIBLE_DEVICES=1
# 定义数据集目录
dataset_dir = "/home/gejunchen/Work/2024-11/Dataset/EMDB"
# 递归获取所有 MP4 文件
video_files = sorted(glob.glob(os.path.join(dataset_dir, "*/*/*.mp4")))
print(f"Found {len(video_files)} video files.")
print(video_files)
# 遍历视频文件并逐个运行
for video in video_files:
    if not video.endswith("_video.mp4"):
        continue
    video_name = os.path.basename(video)[:-10] + '_S'
    video_dir = os.path.dirname(video)
    print("Processing", video_name, video_dir)
    get_video(video_dir, video_name)

