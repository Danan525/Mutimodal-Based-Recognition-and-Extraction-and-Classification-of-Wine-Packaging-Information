import os
import cv2
import easyocr
import pandas as pd

# 初始化 OCR 识别器（中英文）
reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)

# 目标文件夹路径
root_dir = "/media/ntu/volume1/home/s124md306_06/project_cn/processed_gcms"

# 支持的图像格式
image_exts = ['.png', '.jpg', '.jpeg', '.bmp']

# 用于存储最终结果的 DataFrame
all_columns = []

# 遍历所有子目录和图片
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if any(file.lower().endswith(ext) for ext in image_exts):
            image_path = os.path.join(subdir, file)
            print(f"处理图片: {image_path}")
            # 读取图像
            img = cv2.imread(image_path)

            # OCR 识别文本
            results = reader.readtext(img, detail=0)

            # 提取峰面积和成分信息
            area_list = []
            component_list = []
            for i in range(len(results) - 1):
                text1 = results[i]
                text2 = results[i + 1]
                # 简单的逻辑识别：数值 + 中文成分名
                if text1.replace('.', '', 1).isdigit() and not any(c.isdigit() for c in text2):
                    area_list.append(text1)
                    component_list.append(text2)

            # 将该图片的一列峰面积和一列成分组成 DataFrame
            df = pd.DataFrame({
                f"{file}_峰面积": area_list,
                f"{file}_成分": component_list
            })

            all_columns.append(df)

# 将所有结果拼接到一起（按列拼接）
final_df = pd.concat(all_columns, axis=1)

# 保存到 Excel 文件
output_path = "gcms_result.xlsx"
final_df.to_excel(output_path, index=False)
print(f"已保存到 {output_path}")
