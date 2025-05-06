import os
import fitz  # PyMuPDF
from PIL import Image
import shutil

# 配置参数
input_root = "/media/ntu/volume1/home/s124md306_06/project_cn/data"
output_root = "/media/ntu/volume1/home/s124md306_06/project_cn/processed_gcms"
crop_ratio = 0.43  # 裁切上方40%
dpi = 600  # 导出图像分辨率

def convert_pdf_to_image(pdf_path, output_folder):
    """将PDF转换为PNG图像"""
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            # 设置渲染参数
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            img_path = os.path.join(output_folder, 
                                  f"{os.path.splitext(os.path.basename(pdf_path))[0]}_p{page_num}.png")
            pix.save(img_path)
        return True
    except Exception as e:
        print(f"Error converting {pdf_path}: {str(e)}")
        return False

def crop_image(image_path, output_path):
    """裁切图像上方指定比例"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            # 计算裁切区域：去掉上方30%，保留下方70%
            crop_height = int(height * (1 - crop_ratio))
            cropped = img.crop((0, int(height * crop_ratio), 
                             width, 
                             height))
            cropped.save(output_path)
        return True
    except Exception as e:
        print(f"Error cropping {image_path}: {str(e)}")
        return False

def batch_process():
    # 创建临时目录
    temp_dir = os.path.join(output_root, "_temp")
    os.makedirs(temp_dir, exist_ok=True)

    # 第一阶段：PDF转PNG
    print("Starting PDF conversion...")
    for root, dirs, files in os.walk(input_root):
        for filename in files:
            if filename.lower().endswith('.pdf'):
                # 创建对应输出目录
                relative_path = os.path.relpath(root, input_root)
                output_dir = os.path.join(temp_dir, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                
                # 转换PDF
                input_path = os.path.join(root, filename)
                convert_pdf_to_image(input_path, output_dir)

    # 第二阶段：图像裁剪
    print("Starting image cropping...")
    for root, dirs, files in os.walk(temp_dir):
        for filename in files:
            if filename.lower().endswith('.png'):
                # 创建最终输出目录
                relative_path = os.path.relpath(root, temp_dir)
                final_output_dir = os.path.join(output_root, relative_path)
                os.makedirs(final_output_dir, exist_ok=True)
                
                # 处理图像
                input_path = os.path.join(root, filename)
                output_path = os.path.join(final_output_dir, filename)
                crop_image(input_path, output_path)

    # 清理临时文件
    shutil.rmtree(temp_dir)
    print("Temporary files cleaned.")

if __name__ == "__main__":
    print("Starting batch processing...")
    os.makedirs(output_root, exist_ok=True)
    batch_process()
    print("All GC-MS images processed!")