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
from torch.optim.lr_scheduler import StepLR

# ===================== 配置参数 =====================
CSV_PATH = "/media/ntu/volume1/home/s124md306_06/project_cn/aligned_pairs.csv"
BATCH_SIZE = 4
EPOCHS = 5000  # 增加训练轮次
LR = 1e-4
IMG_SAVE_DIR = "./training_update"
os.makedirs(IMG_SAVE_DIR, exist_ok=True)

# ===================== 数据集（保持不变） =====================
class DualChannelDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        # 数据增强
        self.pack_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.RandomHorizontalFlip(),  # 增加水平翻转
            transforms.RandomRotation(20),      # 随机旋转
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 调整亮度、对比度等
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gcms_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pack_img = Image.open(self.df.iloc[idx]['packaging_path'])
        gcms_img = Image.open(self.df.iloc[idx]['gcms_path'])
        return (
            self.pack_transform(pack_img),
            self.gcms_transform(gcms_img),
            torch.tensor(self.df.iloc[idx]['label'], dtype=torch.float)
        )

def dynamic_collate_fn(batch):
    pack, gcms, labels = zip(*batch)
    return list(pack), list(gcms), torch.stack(labels)

# ===================== 修正后的模型 =====================
class DynamicDualBranchMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        # 修正后的包装图分支
        self.pack_net = nn.Sequential(
            *list(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).children())[:-2],  # 使用ResNet50
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 修正后的GC-MS分支
        self.gcms_net = nn.Sequential(
            *list(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).children())[:-2],  # 使用ResNet50
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(2048*2, 512),  # ResNet50输出通道数为2048
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)  # L2正则化
    
    # 使用学习率调度器
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # 每10轮衰减学习率

    # 早期停止设置
    patience = 50  # 如果50轮没有改进，则停止训练
    best_loss = float('inf')
    epochs_without_improvement = 0

    # 训练
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
        
        # 更新学习率
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(IMG_SAVE_DIR, "best_model.pth"))
            print(f"Epoch {epoch+1}: 保存新最佳模型 (loss={avg_loss:.4f})")
        else:
            epochs_without_improvement += 1

        # 早期停止判断
        if epochs_without_improvement >= patience:
            print("早期停止：没有改进，停止训练")
            break

if __name__ == "__main__":
    main()
