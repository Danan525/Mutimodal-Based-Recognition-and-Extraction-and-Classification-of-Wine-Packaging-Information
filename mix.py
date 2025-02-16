import os
import clip
import torch
from PIL import Image
import easyocr
import pandas as pd

# 初始化CLIP模型和OCR工具
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
reader = easyocr.Reader(['ch_sim', 'en'], gpu=device == "cuda")

# 图片文件夹路径
image_folder = r"D:\package_detect\wine_labels_dataset\train"

# 匹配的文本
target_texts = [
    "序号", "品牌", "名称品名", "香型", "产品类型",
    "酒精度", "生产日期", "原料/配料", "保质期", "执行标准", "产地", "生产厂家信息"
]

# 初始化结果存储
results = []

# 遍历文件夹中的图片
for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    if os.path.isfile(image_path):
        try:
            # 加载图片并进行OCR
            image = Image.open(image_path).convert("RGB")
            ocr_result = reader.readtext(image_path, detail=0)

            # 将OCR提取的文本拼接为单个字符串
            detected_text = " ".join(ocr_result)

            # 使用CLIP计算图像和文本的相似性
            image_input = preprocess(image).unsqueeze(0).to(device)
            text_inputs = clip.tokenize(target_texts).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)

                # 计算相似性
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).squeeze(0).cpu().numpy()

            # 获取最高相似的文本
            matched_text = target_texts[similarity.argmax()]
            max_similarity = similarity.max()

            # 存储结果
            results.append({
                "Filename": filename,
                "Detected Text": detected_text,
                "Matched Text": matched_text,
                "Similarity Score": max_similarity
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# 保存结果到Excel
df = pd.DataFrame(results)
output_path = r"D:\package_detect\matching_results.xlsx"
df.to_excel(output_path, index=False)
print(f"Results saved to {output_path}")
