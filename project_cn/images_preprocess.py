import os
from PIL import Image, ImageOps

# 配置参数
input_root = "/media/ntu/volume1/home/s124md306_06/project_cn/data"  # 输入根目录
output_root = "/media/ntu/volume1/home/s124md306_06/project_cn/processed_data"  # 输出根目录
target_size = 224  # 目标尺寸
padding_color = (128, 128, 128)  # 填充颜色 (R, G, B)
valid_extensions = {'.jpg', '.jpeg'}  # 支持的文件扩展名

def process_and_save(image_path, output_path):
    """处理单个图像并保存"""
    try:
        img = Image.open(image_path).convert('RGB')
        processed = ImageOps.pad(img, (target_size, target_size), 
                               color=padding_color, 
                               method=Image.Resampling.LANCZOS)
        processed.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def batch_process():
    # 遍历所有子目录
    for root, dirs, files in os.walk(input_root):
        # 计算相对路径
        relative_path = os.path.relpath(root, input_root)
        output_dir = os.path.join(output_root, relative_path)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理当前目录下的文件
        processed_count = 0
        for filename in files:
            # 检查文件扩展名
            ext = os.path.splitext(filename)[1].lower()
            if ext not in valid_extensions:
                continue
                
            # 构建完整路径
            input_path = os.path.join(root, filename)
            output_path = os.path.join(output_dir, filename)
            
            # 处理并保存
            if process_and_save(input_path, output_path):
                processed_count += 1
        
        # 打印进度
        if processed_count > 0:
            print(f"Processed {processed_count} images in {relative_path}")

if __name__ == "__main__":
    print("Start batch processing...")
    batch_process()
    print("All images processed!")