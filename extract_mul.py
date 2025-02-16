import os
import clip
import torch
from PIL import Image
import easyocr
import pandas as pd
import re

# 初始化CLIP模型和OCR工具
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
reader = easyocr.Reader(['ch_sim', 'en'], gpu=device == "cuda")

# 图片文件夹路径
image_folder = r"D:\package_detect\wine_labels_dataset\train"

# 匹配的关键词
target_texts = [
    "序号", "品牌", "名称品名", "香型", "产品类型",
    "酒精度", "生产日期", "原料/配料", "保质期", "执行标准", "产地", "生产厂家信息"
]

# 定义提取内容的规则
extract_rules = {
    "酒精度": r"酒精度[:：]?\s*(\d+(\.\d+)?%?)",
    "生产日期": r"生产日期[:：]?\s*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?)",
    "保质期": r"保质期[:：]?\s*(\d+\s*(年|个月))",
    "品牌":
}

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
            detected_text = " ".join(ocr_result)

            # 预处理图像用于CLIP模型
            image_input = preprocess(image).unsqueeze(0).to(device)
            text_inputs = clip.tokenize(target_texts).to(device)

            # 使用CLIP计算相似性
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).squeeze(0).cpu().numpy()

            # 汇总匹配信息
            matched_info = []
            for idx, text in enumerate(target_texts):
                if similarity[idx] > 0.3:
                    extracted_content = None
                    if text in extract_rules:
                        match = re.search(extract_rules[text], detected_text)
                        if match:
                            extracted_content = match.group(1)
                    # 汇总匹配信息：匹配项 + 提取内容
                    if extracted_content:
                        matched_info.append(f"{text}为{extracted_content}")
                    else:
                        matched_info.append(f"{text}匹配")

            # 保存每张图片的结果
            results.append({
                "Filename": filename,
                "OCR Text": detected_text,
                "Matched Information": "; ".join(matched_info)  # 汇总所有匹配信息
            })

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# 格式化结果并保存到Excel
df = pd.DataFrame(results)
output_path = r"D:\package_detect\matching_results_summarized.xlsx"
df.to_excel(output_path, index=False)
print(f"Results saved to {output_path}")
