root = "/home/gejunchen/Work/2024-11/Baseline/tram/tram_results"

import os

#获得root下所有images文件夹
def get_images_folders(root):
    images_folders = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "images" in dirnames:
            images_folders.append(os.path.join(dirpath, "images"))
    return images_folders

images_folders = get_images_folders(root)
print(f"Found {len(images_folders)} images folders.")
print(images_folders)

#删除所有images文件夹和里面的内容
def delete_images(images_folders):
    for folder in images_folders:
        if os.path.exists(folder):
            print(f"Deleting folder: {folder}")
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
            os.rmdir(folder)
            print(f"Deleted folder: {folder}")
        else:
            print(f"Folder does not exist: {folder}")
            
delete_images(images_folders)