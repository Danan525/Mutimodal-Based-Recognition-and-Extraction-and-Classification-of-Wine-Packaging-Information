import os
from pdf2image import convert_from_path
from tqdm import tqdm

# 设置输入和输出路径
input_dir = '/media/ntu/volume1/home/s124md306_06/project_cn/data'
output_dir = '/media/ntu/volume1/home/s124md306_06/project_cn/all_images'

# 创建输出目录（如不存在）
os.makedirs(output_dir, exist_ok=True)

# 遍历目录，收集所有PDF文件路径
pdf_paths = []
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith('.pdf'):
            pdf_paths.append(os.path.join(root, file))

print(f'共找到 {len(pdf_paths)} 个 PDF 文件，开始转换...')

# 转换每个 PDF 文件为图片
for pdf_path in tqdm(pdf_paths, desc='转换PDF中'):
    try:
        images = convert_from_path(pdf_path, dpi=300)
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]

        for i, image in enumerate(images):
            img_name = f"{base_name}_page_{i+1}.jpg"
            img_path = os.path.join(output_dir, img_name)
            image.save(img_path, 'JPEG')
    except Exception as e:
        print(f"转换失败：{pdf_path}，错误信息：{e}")
