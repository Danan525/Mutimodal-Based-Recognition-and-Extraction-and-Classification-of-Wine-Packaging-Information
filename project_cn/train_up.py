import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===================== 配置参数 =====================
CSV_PATH = "/media/ntu/volume1/home/s124md306_06/project_cn/aligned_pairs.csv"
BATCH_SIZE = 4
EPOCHS = 5000
LR = 1e-4
IMG_SAVE_DIR = "./training_up"
os.makedirs(IMG_SAVE_DIR, exist_ok=True)
MODEL_PATH = "./training_artifacts/best_model.pth"  # 之前保存的最佳模型路径

# ===================== 数据集（调整为统一尺寸） =====================
class DualChannelDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        
        # Resize 操作，将图像调整为统一尺寸
        self.resize_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 将图像统一调整为 224x224 大小
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 包装图和 GC-MS 图像的转换操作
        self.pack_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gcms_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pack_img = Image.open(self.df.iloc[idx]['packaging_path'])
        gcms_img = Image.open(self.df.iloc[idx]['gcms_path'])

        # 进行 Resize 操作，统一调整图像大小
        pack_img = self.resize_transform(pack_img)
        gcms_img = self.resize_transform(gcms_img)

        label = torch.tensor(self.df.iloc[idx]['label'], dtype=torch.float32)
        return (
            pack_img,
            gcms_img,
            label
        )

def dynamic_collate_fn(batch):
    pack, gcms, labels = zip(*batch)
    return list(pack), list(gcms), torch.stack(labels)

# ===================== 修正后的模型 =====================
class DynamicDualBranchMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用 ResNet50 替换 ResNet18
        self.pack_net = nn.Sequential(
            *list(models.resnet50(weights='IMAGENET1K_V1').children())[:-2],
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 修正后的GC-MS分支
        self.gcms_net = nn.Sequential(
            *list(models.resnet50(weights='IMAGENET1K_V1').children())[:-2],
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(2048*2, 256),  # 由于使用了 ResNet50, 特征维度变为2048
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, pack_imgs, gcms_imgs):
        pack_feats = [self.pack_net(img.unsqueeze(0)).flatten(1) for img in pack_imgs]
        gcms_feats = [self.gcms_net(img.unsqueeze(0)).flatten(1) for img in gcms_imgs]
        fused = torch.cat(
            [torch.cat(pack_feats, dim=0), 
             torch.cat(gcms_feats, dim=0)], 
            dim=1
        )
        return self.classifier(fused).squeeze()

# ===================== 训练循环 =====================
def main():
    # 自动选择空闲的 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # 初始化
    dataset = DualChannelDataset(CSV_PATH)
    train_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=dynamic_collate_fn,
        num_workers=4
    )
    
    model = DynamicDualBranchMatcher().to(device)
    
    # 先加载之前训练的模型权重（只加载与当前模型结构匹配的部分）
    model_dict = model.state_dict()
    pretrained_dict = torch.load(MODEL_PATH)
    
    # 过滤掉与当前模型结构不匹配的部分
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    
    # 更新模型参数
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("加载了已保存的最佳模型权重（只加载匹配的部分）。")

    # 设置损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 训练
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for pack, gcms, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            # 移动数据到设备
            pack = [img.to(device) for img in pack]
            gcms = [img.to(device) for img in gcms]
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(pack, gcms)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 保存最佳模型
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(IMG_SAVE_DIR, "best_model.pth"))
            print(f"Epoch {epoch+1}: 保存新最佳模型 (loss={avg_loss:.4f})")

if __name__ == "__main__":
    main()
