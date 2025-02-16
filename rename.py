import os

def rename_images_in_folder(root_folder, naming_rule):
    # 遍历根文件夹中的每个子文件夹
    for subdir, _, files in os.walk(root_folder):
        image_count = 1  # 计数器，用于为图片重命名

        for file in files:
            # 检查文件扩展名是否为常见图片格式
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                old_path = os.path.join(subdir, file)
                
                # 使用 naming_rule 和 image_count 生成新的文件名
                new_filename = f"{naming_rule}{image_count}.jpg"  # 这里以.jpg为例
                new_path = os.path.join(subdir, new_filename)

                # 重命名图片
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} to {new_path}")

                image_count += 1

# 使用方法
root_folder_path = r"D:\package_detect\wine_labels_dataset\test"  # 文件夹路径，注意前面加 r 以处理反斜杠
naming_rule = "label3_image"  # 替换为你想使用的命名规则
rename_images_in_folder(root_folder_path, naming_rule)
