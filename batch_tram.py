import os
import glob
import subprocess
#CUDA_VISIBLE_DEVICES=1
# 定义数据集目录
dataset_dir = "/home/gejunchen/Work/2024-11/Dataset/EMDB"
result_dir = "/home/gejunchen/Work/2024-11/Baseline/tram/results"
# 递归获取所有 MP4 文件
video_files = sorted(glob.glob(os.path.join(dataset_dir, "*/*/*I.mp4")))
print(f"Found {len(video_files)} video files.")
print(video_files)
# 遍历视频文件并逐个运行
for video in video_files:
    if(video[-5:] != "I.mp4"):
        continue
    if video != "/home/gejunchen/Work/2024-11/Dataset/EMDB/P1/13_outdoor_long_walk/P1_13_outdoor_long_walk_I.mp4":
        continue
    print(f"Processing {video}")
    
    video_name = os.path.basename(video)[:-4]
    print(video_name)
    if os.path.exists(os.path.join(result_dir, video_name, "hps")):
        print(f"Skipping {video}, output already exists.")
        continue
    # 执行 demo.py 命令，直到当前进程结束后再运行下一个
    process = subprocess.run(["python", "scripts/estimate_camera.py", "--video", video], check=True)
    if process.returncode != 0:
        print(f"Error processing {video} SLAM, skipping to next.")
    else:
        print(f"Finished processing {video}")
        process = subprocess.run(["python", "scripts/estimate_humans.py", "--video", video], check=True)
        if process.returncode != 0:
            print(f"Error processing {video} HMR, skipping to next.")
        else:
            print(f"Finished processing {video}")

print("All videos processed.")
