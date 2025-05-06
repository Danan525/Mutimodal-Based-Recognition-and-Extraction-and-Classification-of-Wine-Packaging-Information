import os
import pdfplumber
import pandas as pd

# PDF 文件夹路径
pdf_folder = "/media/ntu/volume1/home/s124md306_06/project_cn/5"
# 输出路径
output_excel = "extracted_from_pdfs.xlsx"

# 最终合并的数据
all_data = {}

# 遍历目录中的所有 PDF 文件
for filename in sorted(os.listdir(pdf_folder)):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, filename)
        print(f"处理文件: {filename}")
        peak_areas = []
        components = []

        # 打开 PDF 并提取文本
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue

                # 拆分为每一行
                lines = text.split("\n")
                for line in lines:
                    # 处理类似于："3280819  0.1085   693821  L-乳酸乙酯"
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            peak = float(parts[0].replace(",", ""))  # 峰面积
                            name = parts[-1]  # 成分名通常是最后一个
                            peak_areas.append(peak)
                            components.append(name)
                        except ValueError:
                            continue  # 跳过无法转换的行

        # 将当前 PDF 的数据加入字典中，每个 PDF 占两列
        col_area = f"{filename}_面积"
        col_comp = f"{filename}_成分"
        all_data[col_area] = peak_areas
        all_data[col_comp] = components

# 转换为 DataFrame（自动对齐列）
df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_data.items()]))

# 保存为 Excel
df.to_excel(output_excel, index=False)
print(f"所有 PDF 的结果已保存至：{output_excel}")
