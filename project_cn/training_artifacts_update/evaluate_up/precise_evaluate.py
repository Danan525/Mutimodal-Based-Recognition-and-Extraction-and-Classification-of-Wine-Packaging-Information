import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score,  # 添加 roc_auc_score
    roc_curve, auc, confusion_matrix, precision_recall_curve,
    average_precision_score
)
import seaborn as sns

# 设置随机种子保证可复现性
np.random.seed(42)

# ---------- 1. 生成模拟数据 ----------
X, y = make_classification(
    n_samples=2000,
    n_features=10,
    n_classes=2,
    weights=[0.85, 0.15],  # 模拟类别不平衡
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------- 2. 训练模型 ----------
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# ---------- 3. 生成预测结果 ----------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # 正类的预测概率

# 保存预测结果到CSV
results_df = pd.DataFrame({
    'true_label': y_test,
    'predicted_label': y_pred,
    'probability_pos_class': y_proba
})
results_df.to_csv('model_predictions.csv', index=False)

# ---------- 4. 计算评价指标 ----------
def calculate_metrics(y_true, y_pred, y_proba):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_proba),
        'Average Precision': average_precision_score(y_true, y_proba)
    }

metrics = calculate_metrics(y_test, y_pred, y_proba)
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv('evaluation_summary.csv', index=False)

# ---------- 5. 可视化 ----------
plt.figure(figsize=(15, 12))

# 混淆矩阵
plt.subplot(2, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['Predicted 0', 'Predicted 1'],
           yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')

# ROC曲线
plt.subplot(2, 2, 2)
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC Curve (AUC = {metrics["AUC-ROC"]:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC Curve')

# PR曲线
plt.subplot(2, 2, 3)
precision, recall, _ = precision_recall_curve(y_test, y_proba)
average_precision = average_precision_score(y_test, y_proba)
plt.plot(recall, precision, color='blue', lw=2,
         label=f'PR Curve (AP = {average_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.title('Precision-Recall Curve')

# 动态阈值分析（F1/Accuracy）
thresholds = np.linspace(0, 1, 50)
f1_scores = []
accuracies = []

for thresh in thresholds:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_thresh))
    accuracies.append(accuracy_score(y_test, y_pred_thresh))

plt.subplot(2, 2, 4)
plt.plot(thresholds, f1_scores, 'b-', label='F1 Score')
plt.plot(thresholds, accuracies, 'g--', label='Accuracy')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.title('Threshold Analysis')

plt.tight_layout()
plt.savefig('model_evaluation_plots.png', dpi=300)
plt.show()

# ---------- 输出结果说明 ----------
print(f'''
=== 评估报告生成完成 ===
生成文件：
1. model_predictions.csv - 包含预测标签和概率
2. evaluation_summary.csv - 评价指标汇总
3. model_evaluation_plots.png - 可视化图表

关键指标：
- AUC-ROC: {metrics["AUC-ROC"]:.3f}（目标 > 0.8）
- F1 Score: {metrics["F1"]:.3f}（目标 > 0.7）
- Recall: {metrics["Recall"]:.3f}（根据场景调整需求）
''')