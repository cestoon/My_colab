import torch
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# 加载预测结果和标签
all_preds = []
all_labels = []

for i in range(5):  # 5折交叉验证
    preds = torch.load(f'./results/predictions_fold_{i}.pt')
    labels = torch.load(f'./results/val_labels_fold_{i}.pt')
    all_preds.append(preds)
    all_labels.append(labels)

# 合并所有折的结果
all_preds = torch.cat(all_preds, dim=0)
all_labels = torch.cat(all_labels, dim=0)

# 计算分类报告和混淆矩阵
print("Classification Report")
for i, label in enumerate(['T1', 'T2', 'T3', 'T4']):
    print(f"Classification Report for {label}")
    print(classification_report(all_labels[:, i], all_preds[:, i], target_names=[f'not_{label}', label]))

print("Confusion Matrix")
for i, label in enumerate(['T1', 'T2', 'T3', 'T4']):
    print(f"Confusion Matrix for {label}")
    print(confusion_matrix(all_labels[:, i], all_preds[:, i]))
