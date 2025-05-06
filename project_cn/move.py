import os
import shutil

# 源文件夹
source_dir = '/media/ntu/volume1/home/s124md306_06/project_cn/data'
# 目标文件夹
target_dir = '/media/ntu/volume1/home/s124md306_06/project_cn/all_pdfs'

# 如果目标文件夹不存在，创建它
os.makedirs(target_dir, exist_ok=True)

# 遍历 source_dir 及其所有子目录
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.lower().endswith('.pdf'):
            source_file = os.path.join(root, file)
            # 为防止文件名重复，可以在名字前加上子文件夹路径（可选）
            target_file = os.path.join(target_dir, os.path.relpath(source_file, source_dir).replace(os.sep, '_'))
            try:
                shutil.copy2(source_file, target_file)
                print(f"复制: {source_file} -> {target_file}")
            except Exception as e:
                print(f"复制 {source_file} 时出错: {e}")
