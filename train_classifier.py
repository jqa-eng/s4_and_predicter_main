#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_classifier.py - 4类细胞毒性分类器训练脚本v4（修正版，符合guidance.md）

v4关键修正（2025-10-15）:
1. 删除错误的交叉验证和OOF逻辑
2. 使用外部验证集进行早停
3. 训练单个模型而非集成模型
4. 保持类别不平衡处理和评估指标
5. 使用随机划分数据集（通过get_datasets.py --classifier实现）

目标:
- Val Macro-F1: >0.60（参考基线）
- 测试集性能与验证集接近，无明显过拟合
- 数据划分方式明确且与论文对齐（随机划分）
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# XGBoost
import xgboost as xgb

# Sklearn
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, matthews_corrcoef,
    cohen_kappa_score, roc_auc_score, average_precision_score,
    confusion_matrix
)

# ============ 辅助函数 ============

def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)


def class_balanced_weights(y_train, num_classes, beta=0.9995):
    """计算Class-Balanced样本权重（考虑有效样本数）

    参数:
        y_train: 训练标签数组
        num_classes: 类别总数
        beta: 平滑系数，默认0.9995（论文推荐值）

    返回:
        每个样本的权重数组

    公式: eff = (1 - beta^n) / (1 - beta), 其中n为样本数
          weight_class = 1 / eff
    """
    counts = np.bincount(y_train, minlength=num_classes).astype(float)
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    w_class = 1.0 / (eff + 1e-12)
    w_class *= (num_classes / w_class.sum())  # 归一化到均值≈1
    return w_class[y_train]


def plot_learning_curve(model, output_path):
    """绘制单个模型的学习曲线"""
    evals_result = model.evals_result()
    train_loss = evals_result["validation_0"]["mlogloss"]
    val_loss = evals_result["validation_1"]["mlogloss"]

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Train", linewidth=2)
    plt.plot(val_loss, label="Validation", linewidth=2)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("mlogloss", fontsize=12)
    plt.title("Learning Curve (External Validation)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved learning curve: {output_path}")


def plot_confusion_matrix(y_true, y_pred, classes, output_path, title="Confusion Matrix"):
    """绘制混淆矩阵（支持中文标签）"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
    plt.rcParams['axes.unicode_minus'] = False

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix: {output_path}")


def compute_classification_metrics(y_true, y_pred, y_proba, n_classes):
    """计算8个评估指标"""
    metrics = {}

    # 基础指标
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro")
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted")
    metrics["balanced_acc"] = balanced_accuracy_score(y_true, y_pred)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)

    # AUC指标（需要二值化标签）
    y_bin = label_binarize(y_true, classes=range(n_classes))

    try:
        metrics["roc_auc_ovr_macro"] = roc_auc_score(y_bin, y_proba,
                                                      average="macro",
                                                      multi_class="ovr")
        metrics["pr_auc_macro"] = average_precision_score(y_bin, y_proba,
                                                           average="macro")
    except:
        metrics["roc_auc_ovr_macro"] = None
        metrics["pr_auc_macro"] = None

    return metrics


# ============ 主训练流程 ============

def main():
    parser = argparse.ArgumentParser(description="4-class Toxicity Classifier (External Validation)")
    parser.add_argument('--train', required=True, help='Training CSV path')
    parser.add_argument('--val', required=True, help='Validation CSV path')
    parser.add_argument('--test', required=True, help='Test CSV path')
    parser.add_argument('--task', required=True, help='Task name (e.g., toxicity)')
    parser.add_argument('--feat-start', type=int, required=True, help='Feature start column index')
    parser.add_argument('--feat-count', type=int, required=True, help='Number of features')
    parser.add_argument('--output-dir', required=True, help='Output directory for models')
    args = parser.parse_args()

    # 创建输出目录
    ensure_dir(args.output_dir)
    plots_dir = os.path.join(args.output_dir, "plots")
    ensure_dir(plots_dir)

    print("=" * 70)
    print("4-Class Toxicity Classifier Training (External Validation)")
    print("=" * 70)
    print(f"Task: {args.task}")
    print(f"Train: {args.train}")
    print(f"Val: {args.val}")
    print(f"Test: {args.test}")
    print(f"Features: [{args.feat_start}:{args.feat_start + args.feat_count}]")
    print(f"Output: {args.output_dir}")

    # Step 1: 加载数据
    print("\n" + "=" * 70)
    print("Step 1: Loading data")
    print("=" * 70)

    df_train = pd.read_csv(args.train, encoding='utf-8-sig')
    df_val = pd.read_csv(args.val, encoding='utf-8-sig')
    df_test = pd.read_csv(args.test, encoding='utf-8-sig')

    print(f"Train shape: {df_train.shape}")
    print(f"Val shape: {df_val.shape}")
    print(f"Test shape: {df_test.shape}")

    # Step 2: 标签编码
    print("\n" + "=" * 70)
    print("Step 2: Label encoding")
    print("=" * 70)

    label_col = f"{args.task}_label"

    # 合并所有数据进行编码（确保一致性）
    label_encoder = LabelEncoder()
    label_encoder.fit(df_train[label_col])

    # 保存标签映射
    label_mapping = {int(i): label for i, label in enumerate(label_encoder.classes_)}
    with open(os.path.join(args.output_dir, "label_mapping.json"), 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)

    print(f"Label mapping: {label_mapping}")

    # 编码标签
    y_train = label_encoder.transform(df_train[label_col])
    y_val = label_encoder.transform(df_val[label_col])
    y_test = label_encoder.transform(df_test[label_col])

    n_classes = len(label_encoder.classes_)
    print(f"Number of classes: {n_classes}")
    print(f"Train label distribution: {np.bincount(y_train)}")
    print(f"Val label distribution: {np.bincount(y_val)}")
    print(f"Test label distribution: {np.bincount(y_test)}")

    # Step 3: 切片特征
    print("\n" + "=" * 70)
    print("Step 3: Slicing features")
    print("=" * 70)

    # 获取特征列名（从训练集DataFrame中直接读取）
    feature_columns = df_train.columns[args.feat_start:args.feat_start + args.feat_count]
    feature_names = feature_columns.tolist()

    print(f"Feature columns range: [{args.feat_start}:{args.feat_start + args.feat_count}]")
    print(f"Total features: {len(feature_names)}")
    print(f"First 5 feature names: {feature_names[:5]}")
    print(f"Last 5 feature names: {feature_names[-5:]}")

    X_train = df_train.iloc[:, args.feat_start:args.feat_start + args.feat_count].values
    X_val = df_val.iloc[:, args.feat_start:args.feat_start + args.feat_count].values
    X_test = df_test.iloc[:, args.feat_start:args.feat_start + args.feat_count].values

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")

    # Step 4: 调整后的超参配置（高维稀疏友好）
    print("\n" + "=" * 70)
    print("Step 4: Hyperparameters (adjusted for high-dim sparse features)")
    print("=" * 70)

    params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "max_depth": 4,              # 5 → 4（降低复杂度）
        "min_child_weight": 8,       # 3 → 8（增加正则）
        "subsample": 0.7,            # 0.8 → 0.7
        "colsample_bytree": 0.3,     # 0.8 → 0.3（关键：高维稀疏友好）
        "reg_lambda": 25.0,          # 5 → 25（增强L2正则）
        "reg_alpha": 8.0,            # 0.5 → 8（增强L1正则）
        "learning_rate": 0.03,       # 0.05 → 0.03（降低学习率）
        "n_estimators": 10000,       # 保持10000（足够early_stopping）
        "random_state": 42,
        "verbosity": 0
    }

    print(f"Parameters: {params}")

    # Step 5: 训练单个模型（使用外部验证集早停）
    print("\n" + "=" * 70)
    print("Step 5: Training single model with external validation early stopping")
    print("=" * 70)

    # 计算样本权重（Class-Balanced）
    sample_weight = class_balanced_weights(y_train, n_classes)
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    print(f"Sample weights distribution: min={sample_weight.min():.3f}, max={sample_weight.max():.3f}, mean={sample_weight.mean():.3f}")

    # 训练模型（使用外部验证集早停）
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        early_stopping_rounds=50,
        verbose=50
    )

    # 保存模型
    model_path = os.path.join(args.output_dir, "toxicity_classifier.pkl")
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved: {model_path}")

    # 获取best_iteration
    best_iter = model.best_iteration
    print(f"Best iteration: {best_iter}")

    # Step 6: 验证集评估
    print("\n" + "=" * 70)
    print("Step 6: Validation set evaluation")
    print("=" * 70)

    y_pred_val = model.predict(X_val)
    y_proba_val = model.predict_proba(X_val)
    metrics_val = compute_classification_metrics(y_val, y_pred_val, y_proba_val, n_classes)

    print("Validation set metrics:")
    for key, value in metrics_val.items():
        if value is not None:
            print(f"  {key}: {value:.4f}")

    # Step 7: 测试集评估
    print("\n" + "=" * 70)
    print("Step 7: Test set evaluation")
    print("=" * 70)

    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)
    metrics_test = compute_classification_metrics(y_test, y_pred_test, y_proba_test, n_classes)

    print("Test set metrics:")
    for key, value in metrics_test.items():
        if value is not None:
            print(f"  {key}: {value:.4f}")

    # Step 8: 保存元数据
    print("\n" + "=" * 70)
    print("Step 8: Saving metadata")
    print("=" * 70)

    meta = {
        "task": args.task,
        "n_classes": n_classes,
        "label_mapping": label_mapping,
        "features": {
            "start": args.feat_start,
            "count": args.feat_count,
            "description": "4274-dim sparse fingerprints (no PCA): 11 continuous + 2048 ECFP4 + 167 MACCS + 2048 RDK",
            "feature_names": feature_names
        },
        "training_strategy": {
            "method": "external_validation",
            "early_stopping_rounds": 50,
            "use_best_iteration": True,
            "description": "Single model training with external validation early stopping (no CV/OOF)"
        },
        "val_metrics": metrics_val,
        "test_metrics": metrics_test,
        "hyperparameters": params,
        "sample_weighting": {
            "method": "class_balanced",
            "beta": 0.9995,
            "description": "Class-Balanced Loss (Cui et al., 2019)"
        },
        "best_iteration": best_iter
    }

    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {meta_path}")

    # Step 9: 绘制学习曲线
    print("\n" + "=" * 70)
    print("Step 9: Plotting learning curve")
    print("=" * 70)

    loss_curve_path = os.path.join(plots_dir, "loss_curve.png")
    plot_learning_curve(model, loss_curve_path)

    # Step 10: 绘制混淆矩阵
    print("\n" + "=" * 70)
    print("Step 10: Plotting confusion matrices")
    print("=" * 70)

    classes = [label_mapping[i] for i in range(n_classes)]

    cm_val_path = os.path.join(plots_dir, "cm_val.png")
    plot_confusion_matrix(y_val, y_pred_val, classes, cm_val_path,
                         title="Validation Set Confusion Matrix")

    cm_test_path = os.path.join(plots_dir, "cm_test.png")
    plot_confusion_matrix(y_test, y_pred_test, classes, cm_test_path,
                         title="Test Set Confusion Matrix")

    # 完成
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)
    print(f"Model saved: {model_path}")
    print(f"Metadata: {meta_path}")
    print(f"Plots: {plots_dir}/")
    print(f"\nKey Results:")
    print(f"  Val Macro-F1: {metrics_val['macro_f1']:.4f}")
    print(f"  Test Macro-F1: {metrics_test['macro_f1']:.4f}")
    print(f"  Val-Test F1 diff: {abs(metrics_val['macro_f1'] - metrics_test['macro_f1']):.4f}")

    # 验证标准检查
    val_f1 = metrics_val['macro_f1']
    test_f1 = metrics_test['macro_f1']
    f1_diff = abs(val_f1 - test_f1)

    print(f"\nValidation against guidance.md standards:")
    print(f"  Using external validation (no CV/OOF): PASS")
    print(f"  Single model training: PASS")
    print(f"  Val Macro-F1 > 0.6: {'PASS' if val_f1 > 0.6 else 'FAIL'} ({val_f1:.4f})")
    print(f"  Val-Test F1 gap < 0.2: {'PASS' if f1_diff < 0.2 else 'FAIL'} ({f1_diff:.4f})")


if __name__ == "__main__":
    main()
