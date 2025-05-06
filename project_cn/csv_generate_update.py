import os
import random
import pandas as pd
from tqdm import tqdm

def generate_aligned_pairs(packaging_root, gcms_root, output_csv="aligned_pairs.csv", neg_per_pos=1):
    """
    生成跨目录对齐的数据对
    :param packaging_root: 包装图根目录
    :param gcms_root: GC-MS图根目录
    :param output_csv: 输出CSV路径
    :param neg_per_pos: 每个正样本对应的负样本数
    """
    # 数据结构：{轮次: {数字文件夹: 文件列表}}
    packaging_data = {}
    gcms_data = {}

    # 扫描包装图目录
    print("扫描包装图目录...")
    for round_dir in os.listdir(packaging_root):
        round_path = os.path.join(packaging_root, round_dir)
        if not os.path.isdir(round_path):
            continue
        
        packaging_data[round_dir] = {}
        for num_folder in os.listdir(round_path):
            folder_path = os.path.join(round_path, num_folder)
            if not os.path.isdir(folder_path):
                continue
            
            jpgs = [os.path.join(folder_path, f) 
                   for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.jpg', '.jpeg'))]
            if jpgs:
                packaging_data[round_dir][num_folder] = jpgs

    # 扫描GC-MS目录
    print("扫描GC-MS目录...")
    for round_dir in os.listdir(gcms_root):
        round_path = os.path.join(gcms_root, round_dir)
        if not os.path.isdir(round_path):
            continue
        
        gcms_data[round_dir] = {}
        for num_folder in os.listdir(round_path):
            folder_path = os.path.join(round_path, num_folder)
            if not os.path.isdir(folder_path):
                continue
            
            pngs = [os.path.join(folder_path, f) 
                   for f in os.listdir(folder_path)
                   if f.lower().endswith('.png')]
            if pngs:
                gcms_data[round_dir][num_folder] = pngs

    # 生成正样本对
    print("生成正样本对...")
    positive_pairs = []
    for round_dir in tqdm(packaging_data.keys()):
        if round_dir not in gcms_data:
            continue
            
        common_folders = set(packaging_data[round_dir].keys()) & set(gcms_data[round_dir].keys())
        for num_folder in common_folders:
            for jpg in packaging_data[round_dir][num_folder]:
                for png in gcms_data[round_dir][num_folder]:
                    positive_pairs.append((jpg, png, 1))

    # 生成负样本对
    print("生成负样本对...")
    negative_pairs = []
    all_gcms_pngs = [png for round_data in gcms_data.values() 
                    for folder_pngs in round_data.values() 
                    for png in folder_pngs]

    for jpg, _, _ in tqdm(positive_pairs):
        # 提取当前样本的轮次和数字文件夹
        path_parts = jpg.split(os.sep)
        src_round = path_parts[-3]
        src_num = path_parts[-2]

        # 收集候选负样本（不同数字文件夹的GC-MS）
        candidates = []
        for png in all_gcms_pngs:
            png_parts = png.split(os.sep)
            png_round = png_parts[-3]
            png_num = png_parts[-2]
            
            # 匹配规则：同一轮次但不同数字文件夹 或 不同轮次
            if (png_round == src_round and png_num != src_num) or (png_round != src_round):
                candidates.append(png)

        # 随机采样
        if candidates:
            selected = random.sample(candidates, min(neg_per_pos, len(candidates)))
            for png in selected:
                negative_pairs.append((jpg, png, 0))

    # 合并数据
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    # 保存到CSV
    df = pd.DataFrame(all_pairs, columns=["packaging_path", "gcms_path", "label"])
    df.to_csv(output_csv, index=False)

    # 打印统计信息
    print("\n生成完成！统计信息：")
    print(f"- 有效轮次数量：{len(set(packaging_data.keys()) & set(gcms_data.keys()))}")
    print(f"- 正样本对：{len(positive_pairs)}")
    print(f"- 负样本对：{len(negative_pairs)}")
    print(f"- 总计样本：{len(all_pairs)}")
    print(f"保存路径：{output_csv}")

# 使用示例
if __name__ == "__main__":
    generate_aligned_pairs(
        packaging_root="/media/ntu/volume1/home/s124md306_06/project_cn/data",
        gcms_root="/media/ntu/volume1/home/s124md306_06/project_cn/processed_gcms",
        neg_per_pos=1
    )