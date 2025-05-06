import os
import torch
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn


# ========== Dataset ==========
class DualChannelDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.resize_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pack_img = Image.open(self.df.iloc[idx]['packaging_path'])
        gcms_img = Image.open(self.df.iloc[idx]['gcms_path'])
        pack_img = self.resize_transform(pack_img)
        gcms_img = self.resize_transform(gcms_img)
        label = torch.tensor(self.df.iloc[idx]['label'], dtype=torch.float32)
        return pack_img, gcms_img, label


def dynamic_collate_fn(batch):
    pack, gcms, labels = zip(*batch)
    return list(pack), list(gcms), torch.stack(labels)


# ========== 模型 ==========
class DynamicDualBranchMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.pack_net = nn.Sequential(
            *list(models.resnet18(weights='IMAGENET1K_V1').children())[:-2],
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.gcms_net = nn.Sequential(
            *list(models.resnet18(weights='IMAGENET1K_V1').children())[:-2],
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
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
            [torch.cat(pack_feats, dim=0), torch.cat(gcms_feats, dim=0)], dim=1
        )
        return self.classifier(fused).squeeze()


# ========== 评估函数 ==========
def evaluate(model, val_loader, device, dataframe, thresholds):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for pack, gcms, labels in val_loader:
            pack = [p.to(device) for p in pack]
            gcms = [g.to(device) for g in gcms]
            labels = labels.to(device)

            outputs = model(pack, gcms)
            preds.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    # 存储每个 threshold 的指标
    metric_rows = []

    for threshold in thresholds:
        binary_preds = [1 if p >= threshold else 0 for p in preds]

        acc = accuracy_score(targets, binary_preds)
        prec = precision_score(targets, binary_preds)
        rec = recall_score(targets, binary_preds)
        f1 = f1_score(targets, binary_preds)
        auc = roc_auc_score(targets, preds)
        cm = confusion_matrix(targets, binary_preds)

        print(f"\n[评估 @ threshold={threshold:.2f}] Acc={acc:.4f}, F1={f1:.4f}")

        # 保存预测值 CSV
        result_df = dataframe.copy()
        result_df["prediction_prob"] = preds
        result_df["prediction_label"] = binary_preds
        result_df.to_csv(f"predictions_threshold_{threshold}.csv", index=False)

        # 混淆矩阵
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
        plt.title(f"Confusion Matrix (threshold={threshold})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(f"confusion_matrix_threshold_{threshold}.png")
        plt.close()

        # ROC 曲线
        fpr, tpr, _ = roc_curve(targets, preds)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC Curve (threshold={threshold})")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"roc_curve_threshold_{threshold}.png")
        plt.close()

        # PR 曲线
        precision, recall, _ = precision_recall_curve(targets, preds)
        plt.figure()
        plt.plot(recall, precision, label=f"Threshold {threshold}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve (threshold={threshold})")
        plt.grid(True)
        plt.savefig(f"pr_curve_threshold_{threshold}.png")
        plt.close()

        # 指标保存
        metric_rows.append({
            "threshold": threshold,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "auc": auc
        })

    # 汇总 CSV
    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv("evaluation_summary.csv", index=False)
    print(f"[INFO] 所有评估结果已保存至 evaluation_summary.csv")

    # F1 vs Threshold
    plt.figure()
    plt.plot(metric_df["threshold"], metric_df["f1_score"], marker='o', label="F1 Score")
    plt.plot(metric_df["threshold"], metric_df["accuracy"], marker='x', label="Accuracy")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("F1 & Accuracy vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.savefig("f1_accuracy_vs_threshold.png")
    print(f"[INFO] 指标曲线图已保存至 f1_accuracy_vs_threshold.png")


# ========== Main ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="路径：CSV 文件")
    parser.add_argument("--model", type=str, required=True, help="路径：模型文件 .pth")
    parser.add_argument("--thresholds", type=str, default="0.3,0.5,0.7",
                        help="多个阈值，用英文逗号隔开，例如：0.3,0.5,0.7")
    args = parser.parse_args()

    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")

    df = pd.read_csv(args.csv)
    dataset = DualChannelDataset(args.csv)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=dynamic_collate_fn)

    model = DynamicDualBranchMatcher().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    evaluate(model, val_loader, device, df, thresholds)
