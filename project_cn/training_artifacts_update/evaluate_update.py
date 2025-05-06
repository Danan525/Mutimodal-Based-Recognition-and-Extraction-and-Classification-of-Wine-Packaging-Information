import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve,
    auc, roc_auc_score, average_precision_score,
    accuracy_score, f1_score, classification_report
)
from sklearn.preprocessing import label_binarize
import seaborn as sns

# ========== 配置参数 ==========
CLASS_NAMES = ['浏阳河', '杜康', '金六福', '老窖二曲', '剑南春', 
              '北京二锅头', '大明', '丰谷', '全兴', '青酒']
N_CLASSES = len(CLASS_NAMES)

# ========== 数据准备 ==========
# 替换为您的真实数据
y_true = [...]  # 真实标签（数值型，0-9对应CLASS_NAMES顺序）
y_pred = [...]  # 预测标签
y_scores = [...]  # 预测概率矩阵（shape=[n_samples, n_classes]）

# ========== 生成预测结果CSV ==========
results_df = pd.DataFrame({
    'true_label': [CLASS_NAMES[i] for i in y_true],
    'predicted_label': [CLASS_NAMES[i] for i in y_pred],
    **{f'prob_{cls}': y_scores[:, i] for i, cls in enumerate(CLASS_NAMES)}
})
results_df.to_csv('improved_predictions.csv', index=False)

# ========== 混淆矩阵 ==========
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.savefig('confusion_matrix.png', bbox_inches='tight')

# ========== 多类别PR曲线 ==========
def plot_pr_curves(y_true, y_scores):
    y_true_bin = label_binarize(y_true, classes=range(N_CLASSES))
    
    plt.figure(figsize=(10, 8))
    for i in range(N_CLASSES):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_scores[:, i])
        plt.plot(recall, precision, lw=2,
                 label=f'{CLASS_NAMES[i]} (AP={ap:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.title('Multiclass PR Curves')
    plt.savefig('pr_curves.png', bbox_inches='tight')

# ========== 多类别ROC曲线 ==========
def plot_roc_curves(y_true, y_scores):
    y_true_bin = label_binarize(y_true, classes=range(N_CLASSES))
    
    plt.figure(figsize=(10, 8))
    for i in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2,
                 label=f'{CLASS_NAMES[i]} (AUC={roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.title('Multiclass ROC Curves')
    plt.savefig('roc_curves.png', bbox_inches='tight')

# ========== 生成评估报告 ==========
def generate_report(y_true, y_pred, y_scores):
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    # 添加AUC和AP指标
    y_true_bin = label_binarize(y_true, classes=range(N_CLASSES))
    auc_scores = []
    ap_scores = []
    for i in range(N_CLASSES):
        auc_scores.append(roc_auc_score(y_true_bin[:, i], y_scores[:, i]))
        ap_scores.append(average_precision_score(y_true_bin[:, i], y_scores[:, i]))
    
    df_report['AUC'] = auc_scores + [np.nan]*3
    df_report['AP'] = ap_scores + [np.nan]*3
    df_report.to_csv('detailed_metrics.csv')

# ========== 执行所有分析 ==========
plot_confusion_matrix(y_true, y_pred)
plot_pr_curves(y_true, y_scores)
plot_roc_curves(y_true, y_scores)
generate_report(y_true, y_pred, y_scores)

print("""
=== The assessment report has been generated successfully. ===
File generated：
1. improved_predictions.csv - Including detailed prediction probabilities
2. confusion_matrix.png - Confusion matrix
3. pr_curves.png - Multi-class PR curve
4. roc_curves.png - Multi-class ROC curve
5. detailed_metrics.csv - Detailed classification indicators
""")