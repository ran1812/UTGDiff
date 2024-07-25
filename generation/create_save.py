import os
import shutil
from datetime import datetime

def archive_python_files(folder_path, archive_folder):
    # 获取当前文件夹下所有的Python文件
    python_files = [f for f in os.listdir(folder_path) if f.endswith('.py')]
    
    # 将Python文件复制到存档文件夹中
    for file in python_files:
        shutil.copy(os.path.join(folder_path, file), archive_folder)

    # 遍历子文件夹并递归调用
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f)) and f != 'file_check']

    for subfolder in subfolders:
        current_archive_folder = os.path.join(archive_folder, subfolder)
        os.makedirs(current_archive_folder)
        archive_python_files(os.path.join(folder_path, subfolder), current_archive_folder)

    print("存档完成！")

if __name__ == "__main__":
    # 获取当前文件夹路径
    current_folder = os.getcwd()
    
    # 创建一个名为file_check的文件夹
    archive_folder = os.path.join(current_folder, 'file_check')
    os.makedirs(archive_folder, exist_ok=True)

    # 获取当前时间
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # 创建一个以当前时间命名的文件夹
    archive_folder = os.path.join(archive_folder, current_time)
    os.makedirs(archive_folder)

    # 调用函数开始存档
    archive_python_files(current_folder, archive_folder)