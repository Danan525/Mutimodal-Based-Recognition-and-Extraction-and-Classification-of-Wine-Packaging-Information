import os
from PIL import Image
import pyheif

def convert_heic_to_jpg_and_replace(file_path):
    try:
        # 读取HEIC文件
        heif_file = pyheif.read(file_path)
        
        # 转换为Pillow Image对象
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
        
        # 生成新的JPG文件路径（相同路径，不同扩展名）
        jpg_path = os.path.splitext(file_path)[0] + ".jpg"
        
        # 保存为JPG格式
        image.save(jpg_path, "JPEG")
        
        # 删除原始HEIC文件
        os.remove(file_path)
        print(f"转换成功并替换: {file_path}")
        
    except Exception as e:
        print(f"处理 {file_path} 时出错: {str(e)}")

def process_directory(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".heic"):
                heic_path = os.path.join(root, file)
                convert_heic_to_jpg_and_replace(heic_path)

if __name__ == "__main__":
    base_dir = "/media/ntu/volume1/home/s124md306_06/project_cn/data"
    process_directory(base_dir)
    print("所有HEIC文件转换完成！")