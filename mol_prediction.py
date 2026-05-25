#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mol_prediction.py - 三属性分子预测脚本

功能：
1. 读取molecules_topred.csv文件中的分子SMILES
2. 调用enhanced_datasets.py的get_full_features()函数生成4274维特征
3. 加载三个预测器模型（毒性、aureus、ecoli）
4. 进行特征对齐和预测
5. 转换logMIC到MIC值，分类结果到标签
6. 输出molecules_processed.csv（表头：smiles,toxicity,aureus_MIC,ecoli_MIC）
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# 导入enhanced_datasets模块的get_full_features函数
from enhanced_datasets import get_full_features

# ============ 预测器配置 ============
PREDICTORS_CONFIG = {
    'toxicity': {
        'model_dir': 'models_predicter/toxicity_classifier_4274d',
        'model_file': 'toxicity_classifier.pkl',
        'feat_start': 2,  # 第2列开始
        'feat_count': 4274,
        'selected_indices': None,  # 使用全部4274维特征
        'task_type': 'classification',
        'n_classes': 4,
        'label_mapping': {"0": "中毒", "1": "低毒", "2": "微毒", "3": "高毒"}
    },
    'aureus': {
        'model_dir': 'models_predicter/aureus_regresser_merged',
        'model_file': 'model.json',  # XGBoost模型
        'feat_start': 3,  # 第5列开始
        'feat_count': 4274,
        'selected_indices': [281, 747, 818, 2399, 2418, 2507, 2570, 2750, 2819, 2918, 2937, 2960, 3149, 3183, 3249, 3524, 3550, 3628, 3629, 3771, 3776, 3911, 3915, 4271],  # 24个特征
        'task_type': 'regression',
        'output_transform': 'log_to_mic'  # logMIC -> MIC
    },
    'ecoli': {
        'model_dir': 'models_predicter/ecoli_regresser',
        'model_file': 'model.json',  # XGBoost模型
        'feat_start': 3,  # 第3列开始
        'feat_count': 4274,
        'selected_indices': [2875,2418,3382,3776,4028,1680,3792,758,
                             2879,3996,3124,1728,108,4243,3595,2347,
                             2283,3982,2216,2498,2743,2687,2671,2637],  # 24个特征
        'task_type': 'regression',
        'output_transform': 'log_to_mic'  # logMIC -> MIC
    }
}

def load_predictor_model(predictor_name, config):
    """加载预测器模型和meta.json配置"""
    model_path = os.path.join(config['model_dir'], config['model_file'])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"  Loading {predictor_name} model from {model_path}")

    if config['task_type'] == 'classification' and config['model_file'].endswith('.pkl'):
        # 毒性分类器使用joblib加载
        model = joblib.load(model_path)
    elif config['model_file'].endswith('.json'):
        # XGBoost回归器使用XGBoost加载
        model = xgb.Booster()
        model.load_model(model_path)
    else:
        raise ValueError(f"Unsupported model file format: {config['model_file']}")

    # 加载meta.json（如果存在）- 用于回归器的z-score反变换和线性校准
    meta_path = os.path.join(config['model_dir'], 'meta.json')
    meta = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            print(f"    Loaded meta.json with normalization and calibration parameters")
        except Exception as e:
            print(f"    Warning: Failed to load meta.json: {e}")

    return model, meta

def extract_features_for_predictor(X_full, feature_ids, config):
    """为特定预测器提取和对齐特征"""
    # X_full: (N, 4274) 来自get_full_features的完整特征矩阵
    # feature_ids: 特征名称列表

    if config['selected_indices'] is None:
        # 毒性分类器使用全部4274维特征
        return X_full
    else:
        # aureus和ecoli使用选定的24个特征
        selected_features = X_full[:, config['selected_indices']]
        print(f"    Selected {selected_features.shape[1]} features from {X_full.shape[1]} total features")
        return selected_features

def predict_with_model(model, X, config, meta=None):
    """使用模型进行预测（支持z-score反变换和线性校准）"""
    if config['task_type'] == 'classification':
        # 分类任务：返回概率分布，取最大概率对应的类别
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            predictions = np.argmax(proba, axis=1)
        else:
            predictions = model.predict(X)
        return predictions

    elif config['task_type'] == 'regression':
        # 回归任务：XGBoost模型
        dmatrix = xgb.DMatrix(X)
        predictions_z = model.predict(dmatrix)  # 这是z-score标准化后的预测值

        # 应用反z变换和线性校准（如果meta.json存在）
        if meta is not None:
            # 步骤1：反z变换 (z-score -> 原始logMIC尺度)
            target_norm = meta.get('target_normalization', {})
            if target_norm.get('enabled', False):
                y_mean = target_norm.get('y_mean', 0.0)
                y_std = target_norm.get('y_std', 1.0)
                predictions_logMIC = predictions_z * y_std + y_mean
                print(f"      Applied inverse z-transform: y_mean={y_mean:.4f}, y_std={y_std:.4f}")
            else:
                predictions_logMIC = predictions_z

            # 步骤2：线性校准 (使用验证集的校准参数)
            calibration = meta.get('calibration', {})
            if calibration.get('enabled', False):
                a_val = calibration.get('a_val', 1.0)
                b_val = calibration.get('b_val', 0.0)
                predictions_calibrated = a_val * predictions_logMIC + b_val
                print(f"      Applied linear calibration: a_val={a_val:.4f}, b_val={b_val:.4f}")
                return predictions_calibrated
            else:
                return predictions_logMIC
        else:
            # 如果没有meta.json，直接返回原始预测（兼容旧模型）
            print(f"      Warning: meta.json not found, using raw predictions without transformation")
            return predictions_z

    else:
        raise ValueError(f"Unsupported task type: {config['task_type']}")

def transform_predictions(predictions, config):
    """转换预测结果"""
    if config['task_type'] == 'classification':
        # 将数值标签转换为文本标签
        label_mapping = config['label_mapping']
        transformed = [label_mapping[str(int(pred))] for pred in predictions]
        return transformed

    elif config.get('output_transform') == 'log_to_mic':
        # logMIC -> MIC: 10^x
        transformed = np.power(10, predictions)
        return transformed

    else:
        return predictions

def main():
    print("=" * 70)
    print("三属性分子预测脚本")
    print("=" * 70)

    # 步骤1: 读取待预测分子
    input_file = "validation/mol_topred.csv"
    output_file = "validation/mol_processed.csv"

    print(f"步骤1: 读取待预测分子从 {input_file}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df_input = pd.read_csv(input_file, encoding='utf-8-sig')
    print(f"  读取了 {len(df_input)} 个分子")

    if 'smiles' not in df_input.columns:
        raise ValueError("Input file must contain 'smiles' column")

    smiles_list = df_input['smiles'].tolist()
    print(f"  SMILES示例: {smiles_list[0] if smiles_list else 'N/A'}")

    # 步骤2: 生成4274维特征
    print(f"\n步骤2: 使用enhanced_datasets.get_full_features()生成特征")
    X_full, feature_ids, meta = get_full_features(smiles_list)
    print(f"  特征矩阵形状: {X_full.shape}")
    print(f"  特征总数: {len(feature_ids)}")
    print(f"  管线信息: {meta.get('total_features')} 维特征")

    # 步骤3: 加载三个预测器模型
    print(f"\n步骤3: 加载三个预测器模型")
    models = {}
    metas = {}
    for predictor_name, config in PREDICTORS_CONFIG.items():
        model, meta = load_predictor_model(predictor_name, config)
        models[predictor_name] = model
        metas[predictor_name] = meta

    # 步骤4: 进行预测
    print(f"\n步骤4: 进行预测")
    results = {'smiles': smiles_list}

    for predictor_name, config in PREDICTORS_CONFIG.items():
        print(f"  预测 {predictor_name}...")

        # 特征提取和对齐
        X_predictor = extract_features_for_predictor(X_full, feature_ids, config)

        # 进行预测（传入meta以支持反变换和校准）
        predictions = predict_with_model(
            models[predictor_name],
            X_predictor,
            config,
            meta=metas[predictor_name]
        )
        print(f"    原始预测范围: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")

        # 转换预测结果（注意：对于回归任务，此时predictions已经是校准后的logMIC）
        transformed_predictions = transform_predictions(predictions, config)

        if config['task_type'] == 'classification':
            print(f"    分类结果示例: {transformed_predictions[:3]}")
            results['toxicity'] = transformed_predictions
        elif predictor_name == 'aureus':
            print(f"    Aureus MIC范围: [{np.min(transformed_predictions):.3f}, {np.max(transformed_predictions):.3f}]")
            results['aureus_MIC'] = transformed_predictions
        elif predictor_name == 'ecoli':
            print(f"    E.coli MIC范围: [{np.min(transformed_predictions):.3f}, {np.max(transformed_predictions):.3f}]")
            results['ecoli_MIC'] = transformed_predictions

    # 步骤5: 构建输出DataFrame
    print(f"\n步骤5: 构建输出结果")
    result_df = pd.DataFrame(results)

    # 确保列顺序正确
    result_df = result_df[['smiles', 'toxicity', 'aureus_MIC', 'ecoli_MIC']]

    print(f"  输出数据形状: {result_df.shape}")
    print(f"  输出列名: {list(result_df.columns)}")

    # 步骤6: 保存结果
    print(f"\n步骤6: 保存结果到 {output_file}")
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  已保存 {len(result_df)} 条预测结果")

    # 显示前几行结果作为验证
    print(f"\n预测结果预览:")
    print(result_df.head())

    print("\n" + "=" * 70)
    print("预测完成！")
    print("=" * 70)

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)