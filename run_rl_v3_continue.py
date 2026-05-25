#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reinforcement learning driver v3_continue_psafe - 毒性约束改为 p_safe 口径（对齐评估）

本脚本基于 run_rl_v3_continue.py，将毒性约束口径从 1-R_tox 改为 p_safe = P(低毒)+P(微毒)，
与评估口径对齐，避免"训练 viol≈0、评估全挂"的问题。

核心改进（v3_continue_psafe）：
===============================
1. **双重抗菌硬门**（课程式过渡，T_warm=5000步）
   - 早期：使用连续抗菌标量 B（cheby/softmin/linear），保持梯度
   - 后期：切换到硬门 (Aureus_MIC<12 AND E.coli_MIC<12)
   - 公式：anti_gate = (1-α)*B + α*anti_hard，其中 α=min(step/5000, 1)

2. **毒性概率约束**（改为 p_safe 口径，移除标签硬门）
   - 从 module_extras 获取四类概率（规范列序：中毒, 低毒, 微毒, 高毒）
   - 计算 p_safe = P(低毒) + P(微毒)
   - 违反量：viol = relu(tau - p_safe)，其中 tau = 1 - tox_ceil
   - Warmup 缩放：前 1000 步逐步打开惩罚，避免策略突变

3. **拉格朗日约束**
   - 惩罚项：penalty = warm * λ * viol
   - 基础奖励：base_reward = anti_gate - penalty
   - 不再使用 tox_gate 和 task_gate（移除标签依赖）

4. **训练与评估口径对齐**
   - 训练：minimize viol = relu(tau - p_safe)
   - 评估：筛选 p_safe ≥ tau
   - 同一量，训练 viol≈0 即评估能过线

使用方式：
=========
从 v3 checkpoint 继续训练：
python run_rl_v3_continue.py \\
    --base-model reinforced_s4_v3/checkpoint_best.pt \\
    --output-dir reinforced_s4_v3_continue_psafe \\
    --prefix "Cc1n[nH]nc1" \\
    --max-steps 10000 \\
    --batch-size 512 \\
    --temp-sample 1.1 \\
    --fusion-mode hard \\
    --tox-ceil 0.30  # 等价 tau=0.70

关键参数：
- T_warm_anti: 5000 步（双抗菌硬门过渡期）
- T_warmup_viol: 1000 步（毒性惩罚 warmup）
- --tox-ceil 0.30 → tau=0.70（p_safe≥0.70）

监控指标：
- p_safe_mean：安全概率均值（应右移至 ≥0.70）
- viol_mean、viol_q90：违反量均值与 q90（应下降）
- warm：warmup 进度（0→1）
- tau：目标阈值
- rate_above_tau：违反比例（应下降）

继承自 v3 的机制（保持不变）：
==========================
- 双重抗菌必须同时有效（已在 v3 实现）
- 禁止单通道更新（已在 v3 实现）
- KL 正则化、长度惩罚、三唑门等
"""

import argparse
import json
import math
import os
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from reward_modules import AureusRewardModule, EcoliRewardModule, ToxicityRewardModule
from s4dd import S4forDenovoDesign

try:
    from rdkit import Chem  # type: ignore
except ImportError:
    Chem = None


# === keep-model-on-grammar ===
BETA_KL    = 0.02   # 先验锚定强度（小而稳，避免"杀红眼"）
LEN_MAX    = 120    # 合理长度上限（将在main中动态设置为模型的sequence_length）
LEN_PENAL  = 0.10   # 线性长度惩罚：超长部分按占比计罚

# ==== Toxic constraint quick-fix (no new CLI) ====
_LAGRANGE_INIT = 0.30          # λ 初值（非 0，避免"很久起不来"）
_LAGRANGE_LR   = 0.05          # λ 的步长（对偶上升）
_LAGRANGE_MAX  = 5.0           # λ 上限
_LAGRANGE_EVERY= 50            # 每隔多少 step 更新一次 λ
_TOX_Q         = 0.90          # 用 q90（上尾）来抬 λ，盯最"危险"的那批

EMA_DECAY = 0.95
LOG_EVERY = 20

# 统一的三唑SMARTS模式（与评测脚本保持一致）
TRIAZOLE_PATTERNS = (
    "c1n[nH]nc1",  # 1,2,4-triazole (aromatic)
    "c1nc[nH]n1",  # 1,2,4-triazole (tautomer)
    "c1c[nH]nn1",  # 1,2,3-triazole (aromatic)
    "c1[nH]nnc1",  # 1,2,3-triazole (tautomer)
    "n1ncnc1",     # 1,2,4-triazole (deprotonated)
    "n1nccn1",     # 1,2,3-triazole (deprotonated)
)

TETRAZOLE_PATTERNS = (
    "c1nnn[nH]1",  # tetrazole aromatic
    "c1nn[nH]n1",  # tetrazole tautomer
    "c1[nH]nnn1",  # tetrazole tautomer 2
    "n1nnnn1",     # deprotonated tetrazole
)

if Chem is not None:
    _TRIAZOLE_SMARTS = tuple(
        smarts
        for smarts in (Chem.MolFromSmarts(pattern) for pattern in TRIAZOLE_PATTERNS)
        if smarts is not None
    )
    _TETRAZOLE_SMARTS = tuple(
        smarts
        for smarts in (Chem.MolFromSmarts(pattern) for pattern in TETRAZOLE_PATTERNS)
        if smarts is not None
    )
else:
    _TRIAZOLE_SMARTS = ()
    _TETRAZOLE_SMARTS = ()


def validate_triazole(smiles: str) -> bool:
    """训练期的宽松三唑检测：放宽解析，但保证属性缓存就绪，再做匹配。
    - 先用 sanitize=False 解析，防止一上来就抛异常；
    - 立刻 UpdatePropertyCache(strict=False)，补上隐式价/氢等；
    - 做最小化的环/芳香感知（catchErrors=True，尽量不抛）；
    - Substructure 匹配放进 try/except 的安全包装，失败视为 False。
    """
    if Chem is None:
        return True  # 训练期不阻断；评估/筛选阶段再严格
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return False

        # 关键补丁：补齐隐式价/氢等属性缓存，避免 getNumImplicitHs 报错
        mol.UpdatePropertyCache(strict=False)

        # 最小化的环/芳香感知（不要用全量 SANITIZE_ALL；catchErrors 避免抛）
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=(
                    Chem.SanitizeFlags.SANITIZE_FINDRINGS |
                    Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                    Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                ),
                catchErrors=True
            )
        except Exception:
            pass

        # 安全匹配包装，避免个别分子在匹配内部仍抛异常
        def _safe_match(patterns):
            for patt in patterns:
                try:
                    if mol.HasSubstructMatch(patt):
                        return True
                except Exception:
                    continue
            return False

        # 先排除四唑，再判定三唑
        if _safe_match(_TETRAZOLE_SMARTS):
            return False
        return _safe_match(_TRIAZOLE_SMARTS)

    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RL fine-tuning with soft reward fusion and relaxed early stopping."
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,  # 改为必填，避免误用默认值
        help="Directory that contains the pretrained S4 weights (init_arguments.json + model.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reinforced_s4_v3",
        help="Directory where checkpoints, history, and best weights are stored.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="Cc1n[nH]nc1",
        help="SMILES prefix enforced during rollout (prefix replay).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of molecules generated per policy update.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=40000,
        help="Maximum number of policy updates (hard stop).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,  # 默认1，相当于禁用早停（只在最后判断）
        help="Logical epoch count for early stopping statistics (1 = disabled).",
    )
    parser.add_argument(
        "--temp-sample",
        type=float,
        default=1.0,
        help="Sampling temperature for exploration rollouts (higher = more exploration).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=12,  # 兜底，确保不会中途早停
        help="Number of bad epochs to allow before early stopping (9999 = disabled).",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.02,
        help="Minimum epoch improvement required to reset patience.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Checkpoint frequency in steps.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping threshold; disabled when set to 0 or negative.",
    )
    parser.add_argument(
        "--preset",
        choices=["both", "positive-only", "negative-only"],
        default="both",
        help="Reward channel preset helper.",
    )
    parser.add_argument(
        "--disable-aureus",
        action="store_true",
        help="Disable the S. aureus reward stream.",
    )
    parser.add_argument(
        "--disable-ecoli",
        action="store_true",
        help="Disable the E. coli reward stream.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g. cuda, cuda:1, cpu). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--fusion-mode",
        type=str,
        choices=["soft", "hard"],
        default="hard",  # ★改为硬融合：两路都有效才计入更新
        help="Reward fusion mode: 'soft' = at least one module valid; 'hard' = all modules valid.",
    )
    
# --- 抗菌性两路权衡 ---
    parser.add_argument("--anti-mode", choices=["cheby","softmin","linear"], default="cheby",
                        help="aureus/ecoli 的权衡：cheby=推短板，softmin=平滑最小化，linear=原线性加权")
    parser.add_argument("--anti-weights", type=str, default="pos=0.5,neg=0.5",
                        help="阳/阴两路权重，如 pos=0.5,neg=0.5")
    parser.add_argument("--anti-beta", type=float, default=6.0,
                        help="softmin 锐度（越大越接近最小值）")

    # --- 毒性拉格朗日约束 ---
    parser.add_argument("--tox-ceil", type=float, default=0.5,
                        help="允许的细胞毒性上限（用 p_tox 口径；若只有 R_tox，则近似 p_tox=1-R_tox）")
    parser.add_argument("--lagrange-lr", type=float, default=0.02,
                        help="拉格朗日乘子 λ 的学习率（每 epoch 用平均违反量更新）")
    parser.add_argument("--lagrange-max", type=float, default=5.0,
                        help="λ 的上界，防止爆掉")

    # --- 长度惩罚参数 ---
    parser.add_argument("--len-max", type=int, default=None,
                        help="长度惩罚阈值（默认使用模型的sequence_length）")

    return parser.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_channels(args: argparse.Namespace) -> Dict[str, bool]:
    preset = {
        "both": {"aureus": True, "ecoli": True, "tox": True},
        "positive-only": {"aureus": True, "ecoli": False, "tox": True},
        "negative-only": {"aureus": False, "ecoli": True, "tox": True},
    }[args.preset]

    if args.disable_aureus:
        preset["aureus"] = False
    if args.disable_ecoli:
        preset["ecoli"] = False
    preset["tox"] = True  # tox channel is mandatory

    if not any(preset.values()):
        raise ValueError("At least one reward channel must remain enabled.")
    if not preset["tox"]:
        raise ValueError("Toxicity reward cannot be disabled.")
    return preset


def parse_weights(raw: str, enabled: Dict[str, bool]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    if raw:
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        for token in tokens:
            if "=" not in token:
                raise ValueError(f"Invalid weight fragment '{token}', expected key=value.")
            key, value = token.split("=", 1)
            key = key.strip().lower()
            weights[key] = float(value)
    # guarantee defaults for enabled channels
    for key, is_on in enabled.items():
        if not is_on:
            continue
        weights.setdefault(key, 1.0)

    filtered = {k: v for k, v in weights.items() if enabled.get(k, False)}
    if not filtered:
        raise ValueError("No reward weights supplied for the enabled channels.")
    total = sum(filtered.values())
    if total <= 0:
        raise ValueError("Reward weights must sum to a positive value.")
    return {k: v / total for k, v in filtered.items()}




def _parse_kv(s: str, keys):
    out = {k: None for k in keys}
    for kv in s.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            k = k.strip(); v = v.strip()
            if k in out:
                try:
                    out[k] = float(v)
                except Exception:
                    pass
    for k in keys:
        if out[k] is None:
            out[k] = 0.5
    return out


class RewardEMA:
    def __init__(self, decay: float):
        self.decay = decay
        self.value = None

    def current(self) -> float:
        if self.value is None:
            return 0.0
        return float(self.value)

    def update(self, sample: float) -> float:
        if self.value is None:
            self.value = float(sample)
        else:
            self.value = self.decay * float(self.value) + (1.0 - self.decay) * float(sample)
        return float(self.value)


def stack_traces(
    actions: List[torch.Tensor],
    logprobs: List[torch.Tensor],
    masks: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not actions:
        raise ValueError("Rollout returned no actions.")
    actions_tensor = torch.stack(actions, dim=0)
    logprobs_tensor = torch.stack(logprobs, dim=0)
    masks_tensor = torch.stack(masks, dim=0)
    return actions_tensor, logprobs_tensor, masks_tensor

def compute_combined_reward_v3(
    smiles: List[str],
    modules: Dict[str, object],
    anti_mode: str,
    anti_weights: str,
    anti_beta: float,
    tox_ceil: float,
    lam_tox: torch.Tensor,
    actions: torch.Tensor = None,
    logprobs: torch.Tensor = None,
    masks: torch.Tensor = None,
    s4_prior_model = None,
    s4_current_model = None,
    temperature: float = 1.0,
    fusion_mode: str = "soft",
    step: int = 0  # 新增：训练步数，用于课程式门控
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Dict[str, float]], torch.Tensor, torch.Tensor]:
    """
    v3_continue: 两路抗菌性做"短板优先"标量化 + 课程式硬门（双抗菌AND + 毒性标签）

    改进：
    1. 双重抗菌硬门：要求 Aureus<12 且 E.coli<12 同时达标（带课程式过渡）
    2. 毒性标签硬门：要求标签=低毒(1)或微毒(2)（带课程式过渡）
    3. 早期保持连续梯度，T_warm步后切换到硬门，避免拒学

    返回: combined, valid_mask, stats, B, viol
    """
    batch_size = len(smiles)
    if batch_size == 0:
        raise ValueError("Empty SMILES batch.")

    module_rewards: Dict[str, torch.Tensor] = {}
    module_valid: Dict[str, torch.Tensor] = {}
    module_extras: Dict[str, Dict[str, object]] = {}  # 新增：收集可选字段
    stats: Dict[str, Dict[str, float]] = {}

    # 1. 逐模块评分
    for name, module in modules.items():
        result = module.compute_reward(smiles)
        rewards = result.get("rewards", [])
        if len(rewards) != batch_size:
            raise ValueError(f"Reward module '{name}' returned {len(rewards)} values for batch of size {batch_size}.")
        reward_tensor = torch.tensor([r if r is not None else 0.0 for r in rewards], dtype=torch.float32)
        valid_tensor  = torch.tensor([r is not None for r in rewards], dtype=torch.bool)
        module_rewards[name] = reward_tensor
        module_valid[name] = valid_tensor
        # 新增：收集可选字段（若模块没有提供则为None）
        module_extras[name] = {
            "labels": result.get("labels", None),  # 毒性分类的 0/1/2/3（若模块提供）
            "probs":  result.get("probs",  None),  # 毒性四类概率（若模块提供）
            "mic":    result.get("mic",    None),  # 抗菌的原始 MIC 数值（若模块提供）
        }
        stats[name] = {
            "valid_rate": valid_tensor.float().mean().item(),
            "mean_reward": reward_tensor[valid_tensor].mean().item() if valid_tensor.any() else float("nan"),
        }

    # 2. 取 key
    pos_key = next((k for k in module_rewards.keys() if "aureus" in k.lower()), None)
    neg_key = next((k for k in module_rewards.keys() if "ecoli"  in k.lower()), None)
    tox_key = next((k for k in module_rewards.keys() if "tox"    in k.lower()), None)

    device = lam_tox.device if isinstance(lam_tox, torch.Tensor) else torch.device("cpu")
    B = torch.zeros(batch_size, dtype=torch.float32, device=device)
    viol = torch.zeros(batch_size, dtype=torch.float32, device=device)

    # 3. 抗菌性标量化（默认 cheby 推短板）
    v_pos = module_valid.get(pos_key, None) if pos_key else None
    v_neg = module_valid.get(neg_key, None) if neg_key else None
    r_pos = module_rewards.get(pos_key, None) if pos_key else None
    r_neg = module_rewards.get(neg_key, None) if neg_key else None

    # 拷到 device
    def to_dev(x):
        return x.to(device) if x is not None else None
    v_pos = to_dev(v_pos); v_neg = to_dev(v_neg)
    r_pos = to_dev(r_pos); r_neg = to_dev(r_neg)

    aw = _parse_kv(anti_weights, keys=("pos","neg"))
    w_pos, w_neg = float(aw["pos"]), float(aw["neg"])
    s = max(w_pos + w_neg, 1e-8); w_pos/=s; w_neg/=s
    eps = 1e-8

    valid_B = torch.zeros(batch_size, dtype=torch.bool, device=device)
    if (r_pos is not None) and (r_neg is not None):
        if anti_mode == "cheby":
            B = 1.0 - torch.maximum(w_pos*(1.0 - r_pos), w_neg*(1.0 - r_neg))
        elif anti_mode == "softmin":
            beta = float(anti_beta)
            B = -torch.log(w_pos*torch.exp(-beta*r_pos) + w_neg*torch.exp(-beta*r_neg) + eps) / beta
        else:
            B = w_pos*r_pos + w_neg*r_neg
        # ★两路都有效才是有效样本；堵死"只打一边也能更新"的捷径
        valid_B = (v_pos & v_neg) if (fusion_mode == "hard") else (v_pos | v_neg)
    else:
        # ★任一通道缺失：不产生奖励、不参与更新（保持 B=0, valid_B=False）
        B = B  # keep zeros
        valid_B = valid_B  # keep False

    # === 3.x 抗菌硬门（带课程式过渡） ===
    # B 是现有的连续抗菌标量（cheby/softmin/linear）
    # 构造 anti_hard ∈{0,1}，要求双菌均达标
    # ★ 标定说明：MIC<12 的最低奖励约 0.48（0.38底分+0.10加成），取 0.44 兜底（略放松）
    r_thr = 0.44  # 近似阈值：若无 MIC 原值，认为 reward≥0.44 ≈ "MIC<12"
    anti_hard = torch.zeros(batch_size, dtype=torch.float32, device=device)

    # 优先用 MIC 原值（允许 None：转为 nan 并用掩码过滤）
    mic_pos = module_extras.get(pos_key, {}).get("mic", None) if pos_key else None
    mic_neg = module_extras.get(neg_key, {}).get("mic", None) if neg_key else None
    if isinstance(mic_pos, (list, np.ndarray)) and isinstance(mic_neg, (list, np.ndarray)) and len(mic_pos)==batch_size and len(mic_neg)==batch_size:
        def _to_mic_tensor(lst):
            out = torch.empty(batch_size, dtype=torch.float32, device=device)
            for i, v in enumerate(lst):
                out[i] = float(v) if (v is not None and np.isfinite(float(v))) else float('nan')
            return out
        mic_pos_t = _to_mic_tensor(mic_pos)
        mic_neg_t = _to_mic_tensor(mic_neg)
        valid_pair = (~torch.isnan(mic_pos_t)) & (~torch.isnan(mic_neg_t))
        anti_hard = torch.zeros(batch_size, dtype=torch.float32, device=device)
        if valid_pair.any():
            anti_hard[valid_pair] = ((mic_pos_t[valid_pair] < 12.0) & (mic_neg_t[valid_pair] < 12.0)).float()
    elif (r_pos is not None) and (r_neg is not None):
        # 退回用 reward 阈值近似
        anti_hard = ((r_pos >= r_thr) & (r_neg >= r_thr)).float()

    # 课程式过渡：前期沿用 B（连续、可导），T_warm 后变为硬门
    T_warm = 5000.0
    alpha = torch.clamp(torch.tensor(step, device=device, dtype=torch.float32) / T_warm, 0.0, 1.0)
    anti_gate = (1.0 - alpha) * B + alpha * anti_hard

    # === 4. 毒性概率约束（改为 p_safe 口径，对齐评估） ===
    if tox_key is not None:
        v_tox = module_valid[tox_key].to(device)

        # 从 module_extras 获取四类概率，计算 p_safe = P(低毒) + P(微毒)
        tox_probs = module_extras.get(tox_key, {}).get("probs", None)

        # ★ 强校验：禁止退化到R_tox兜底（确保一定使用概率口径）
        if tox_probs is None:
            raise RuntimeError(
                f"[FATAL] tox_probs missing at step={step}! "
                "ToxicityRewardModule must return 'probs' field. "
                "Refusing to fallback to R_tox to prevent train/eval mismatch."
            )

        # 安全构造：处理 None 值
        safe_probs = [(p if (isinstance(p, (list, tuple)) and len(p) == 4) else [float('nan')] * 4)
                     for p in tox_probs]
        tp = torch.tensor(safe_probs, dtype=torch.float32, device=device)  # [B,4]
        if tp.ndim != 2 or tp.shape[1] != 4:
            raise RuntimeError(
                f"[FATAL] tox_probs shape invalid at step={step}: expected [B,4], got {tp.shape}. "
                "Check ToxicityRewardModule output format."
            )

        # p_safe = P(低毒)[1] + P(微毒)[2]，规范列序：中毒,低毒,微毒,高毒
        p_safe = (tp[:, 1] + tp[:, 2]).nan_to_num(0.0).clamp(0, 1)

        # 违反量定义：tau - p_safe（与评估对齐）
        # tau = 1 - tox_ceil（从原有 p_tox≤tox_ceil 反推 p_safe≥tau）
        # ★ tau 热身：前 1000 步从 0.58 渐进到目标值，避免初期过严
        T_warmup_viol = 1000.0
        warm = torch.clamp(torch.tensor(step, device=device, dtype=torch.float32) / T_warmup_viol, 0.0, 1.0)
        tau_target = 1.0 - float(tox_ceil)
        tau_eff = 0.58 + warm * (tau_target - 0.58)  # 0.58 → tau_target
        viol = torch.relu(tau_eff - p_safe)

        # Warmup 缩放：前 1000 步逐步打开惩罚，避免策略突变
        penalty = warm * lam_tox * viol

        # ★ 惩罚上限：防止偶发批次惩罚过大吞光抗菌奖励
        penalty = torch.minimum(penalty, 0.8 * anti_gate.detach())

        # 基础奖励 = 抗菌门 - 毒性惩罚（不再使用 tox_gate）
        base_reward = anti_gate - penalty
        valid_mask = valid_B & v_tox
        if not valid_mask.any():
            valid_mask = valid_B | v_tox
    else:
        base_reward = anti_gate
        valid_mask = valid_B

    # 5. KL正则化和长度惩罚（新增的语法约束）
    kl_ptok = torch.zeros(batch_size, dtype=torch.float32, device=device)
    len_ratio = torch.zeros(batch_size, dtype=torch.float32, device=device)

    if actions is not None and masks is not None and s4_prior_model is not None and s4_current_model is not None:
        # 进入 KL 计算分支前，先对齐设备
        dev = lam_tox.device if isinstance(lam_tox, torch.Tensor) else torch.device("cpu")
        if actions is not None and actions.device != dev:
            actions = actions.to(dev)
        if masks is not None and masks.device != dev:
            masks = masks.to(dev)

        # 定义一致的logits处理函数
        def apply_logits_mask(logits_t):
            # 温度缩放
            logits_t = logits_t / max(1e-6, float(temperature))
            # 这里可以添加其他一致的口罩逻辑（如果rollout中使用了的话）
            return logits_t

        try:
            batch_sz = actions.shape[1]  # actions: [T, B]
            T = actions.shape[0]

            if T <= 1:
                # 序列太短，跳过KL计算
                kl_ptok = torch.zeros(batch_sz, dtype=torch.float32, device=dev)
                len_ratio = torch.zeros(batch_sz, dtype=torch.float32, device=dev)
            else:
                # ---------- 当前策略（带梯度，teacher-forcing重算） ----------
                s4_current_model.eval()  # 关掉dropout，与prior一致
                s4_current_model.reset_state(batch_size=batch_sz, device=dev)

                cur_lp_steps = []
                for t in range(T-1):
                    # 用 actions[t] 更新状态，预测下一步分布
                    logits_t = s4_current_model.recurrent_step(actions[t])  # [B, vocab_size]
                    logits_t = apply_logits_mask(logits_t)
                    logp_t = torch.log_softmax(logits_t, dim=-1)
                    # 取 a_{t+1} 的logprob
                    cur_lp_steps.append(logp_t.gather(1, actions[t+1].unsqueeze(1)).squeeze(1))

                cur_lp_steps = torch.stack(cur_lp_steps, dim=0)  # [T-1, B]

                # ---------- prior（不带梯度，teacher-forcing） ----------
                with torch.no_grad():
                    s4_prior_model.eval()
                    s4_prior_model.reset_state(batch_size=batch_sz, device=dev)

                    pri_lp_steps = []
                    for t in range(T-1):
                        logits_t = s4_prior_model.recurrent_step(actions[t])  # [B, vocab_size]
                        logits_t = apply_logits_mask(logits_t)  # 同温度、同口罩
                        logp_t = torch.log_softmax(logits_t, dim=-1)
                        # 取 a_{t+1} 的logprob
                        pri_lp_steps.append(logp_t.gather(1, actions[t+1].unsqueeze(1)).squeeze(1))

                    pri_lp_steps = torch.stack(pri_lp_steps, dim=0)  # [T-1, B]

                # ---------- 计算平均logprob（对齐到 a_{1..T-1}） ----------
                token_mask = masks.float().to(dev)         # masks 已是 [T-1,B]（外层传入的是 alive[1:, :])
                token_mask_sum = token_mask.sum(dim=0).clamp(min=1.0)  # [B]

                cur_logp_seq = (cur_lp_steps * token_mask).sum(dim=0) / token_mask_sum  # [B]
                pri_logp_seq = (pri_lp_steps * token_mask).sum(dim=0) / token_mask_sum  # [B]

                # ---------- KL/token ----------
                kl_ptok = (cur_logp_seq - pri_logp_seq).clamp_min(0.0).clamp_max(5.0)

                # ---------- 长度惩罚 ----------
                lengths = token_mask_sum  # [B] 有效token数
                len_ratio = (lengths - LEN_MAX).clamp_min(0.0) / max(1, float(LEN_MAX))

            # 将KL和长度惩罚移到主设备（与base_reward对齐）
            kl_ptok = kl_ptok.to(device)
            len_ratio = len_ratio.to(device)

        except Exception as e:
            # 如果KL计算失败，保持为0（兜底，使用主设备）
            print(f"Warning: KL calculation failed: {e}")
            import traceback
            traceback.print_exc()
            kl_ptok = torch.zeros(batch_size, dtype=torch.float32, device=device)
            len_ratio = torch.zeros(batch_size, dtype=torch.float32, device=device)

    # 合成：两项正则只在训练期生效
    combined = base_reward - BETA_KL * kl_ptok - LEN_PENAL * len_ratio

    # 附加统计
    stats["antibacterial_scalar"] = {"valid_rate": valid_B.float().mean().item(),
                                     "mean_reward": B[valid_B].mean().item() if valid_B.any() else float("nan")}
    if tox_key is not None:
        stats["tox_constraint"] = {
            "valid_rate": v_tox.float().mean().item(),
            "mean_p_safe": p_safe[v_tox].mean().item() if v_tox.any() else float("nan"),  # 改为监控 p_safe
            "mean_violation": viol[v_tox].mean().item() if v_tox.any() else 0.0,
            "viol_q90": viol[v_tox].quantile(0.90).item() if v_tox.any() else 0.0  # 新增：违反 q90
        }

    # 添加KL和长度统计
    stats["kl_regularization"] = {"mean_kl_ptok": kl_ptok.mean().item()}
    stats["length_penalty"] = {"mean_length": len_ratio.mean().item() * LEN_MAX + LEN_MAX}

    return combined, valid_mask, stats, B, viol, module_extras




def ensure_dirs(output_dir: str) -> Dict[str, str]:
    checkpoints = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints, exist_ok=True)
    return {"output": output_dir, "checkpoints": checkpoints}


def save_history(history: List[Tuple[int, float, float, float]], output_dir: str) -> None:
    if not history:
        return
    csv_path = os.path.join(output_dir, "reward_history.csv")
    png_path = os.path.join(output_dir, "reward_curve.png")
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("step,mean_reward,valid_rate,triazole_rate\n")
        for step, mean_reward, valid_rate, triazole_rate in history:
            handle.write(f"{step},{mean_reward:.6f},{valid_rate:.6f},{triazole_rate:.6f}\n")

    steps = [item[0] for item in history]
    means = [item[1] for item in history]
    plt.figure()
    plt.plot(steps, means, label="mean reward")
    plt.xlabel("step")
    plt.ylabel("mean reward")
    plt.title("RL reward curve (v3)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()


def save_checkpoint(s4_model: S4forDenovoDesign, path: str) -> None:
    s4_model.save(path)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    enabled = resolve_channels(args)

    dirs = ensure_dirs(args.output_dir)

    print("=" * 70)
    print("=== Reinforcement Learning Configuration ===")
    print("=" * 70)
    print(f"Base model       : {args.base_model}")
    print(f"Output directory : {args.output_dir}")
    print(f"Device           : {device}")
    print(f"Channels enabled : {enabled}")
    print(f"Fusion mode      : {args.fusion_mode} {'(soft = at least one module valid)' if args.fusion_mode == 'soft' else '(hard = all modules valid)'}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Max steps        : {args.max_steps}")
    print(f"Epochs           : {args.epochs} {'(early stopping disabled)' if args.epochs == 1 else ''}")
    print(f"Patience         : {args.patience} {'(early stopping disabled)' if args.patience >= 9999 else ''}")
    print(f"Temperature      : {args.temp_sample} {'(increased exploration)' if args.temp_sample > 1.0 else ''}")
    print(f"Learning rate    : {args.lr}")
    print(f"Min delta        : {args.min_delta}")
    print("=" * 70)

    # 检查底模路径是否存在
    if not os.path.exists(args.base_model):
        raise FileNotFoundError(f"Base model directory not found: {args.base_model}")

    model_pt_path = os.path.join(args.base_model, "model.pt")
    init_args_path = os.path.join(args.base_model, "init_arguments.json")

    if not os.path.exists(model_pt_path):
        raise FileNotFoundError(f"model.pt not found in base model: {model_pt_path}")
    if not os.path.exists(init_args_path):
        raise FileNotFoundError(f"init_arguments.json not found in base model: {init_args_path}")

    print(f"Base model validated: {args.base_model}")
    print()

    s4 = S4forDenovoDesign.from_file(args.base_model)
    s4.device = str(device)
    s4.s4_model.to(device)

    # 动态设置长度惩罚阈值为模型的sequence_length
    global LEN_MAX
    LEN_MAX = int(s4.sequence_length) if (args.len_max is None or args.len_max <= 0) else int(args.len_max)
    print(f"Length penalty threshold set to: {LEN_MAX} (model sequence_length)")

    # 冻结 prior（基于当前加载的 base-model 权重）
    import copy
    s4_prior = copy.deepcopy(s4)
    s4_prior.s4_model.to(device)
    s4_prior.s4_model.eval()
    for p in s4_prior.s4_model.parameters():
        p.requires_grad_(False)

    # 关键：为 prior 开启逐步推理模式（step 模式）
    for module in s4_prior.s4_model.modules():
        if hasattr(module, "setup_step"):
            module.setup_step()

    aureus = AureusRewardModule() if enabled.get("aureus") else None
    ecoli = EcoliRewardModule() if enabled.get("ecoli") else None
    toxicity = ToxicityRewardModule() if enabled.get("tox") else None
    modules = {
        name: module
        for name, module in [
            ("aureus", aureus),
            ("ecoli", ecoli),
            ("tox", toxicity),
        ]
        if module is not None
    }

    optimizer = torch.optim.AdamW(s4.s4_model.parameters(), lr=args.lr)
    ema = RewardEMA(EMA_DECAY)

    steps_per_epoch = max(args.max_steps // args.epochs, 1)
    history: List[Tuple[int, float, float, float]] = []
    epoch_rewards: List[float] = []
    best_epoch_score = -math.inf
    best_state = None
    patience_counter = 0
    start_time = time.time()

# 拉格朗日乘子（毒性约束）
    class _State: pass
    state = _State()
    state.lambda_tox = torch.tensor(_LAGRANGE_INIT, device=device)

    # 统计 epoch 内毒性违反均值
    stats_epoch = {"tox_viol_sum": 0.0, "tox_viol_cnt": 0}

    # ★ 自检统计：监控 probs 缺失次数（应始终为 0）
    stats_diag = {"tox_probs_missing": 0, "tox_probs_invalid_shape": 0}

    print("Starting RL training loop...")
    print()

    for step_idx in range(1, args.max_steps + 1):
        # 1. Rollout: 生成分子
        actions, logprobs, masks, smiles = s4.rollout(
            batch_size=args.batch_size,
            max_len=s4.sequence_length,
            required_prefix_smiles=args.prefix,
            temperature=args.temp_sample,
        )

        # 堆叠
        A  = torch.stack(actions,  dim=0)            # [T,B]
        LP = torch.stack(logprobs, dim=0)            # [T,B]
        M  = torch.stack(masks,    dim=0).to(device) # [T,B]  True=非END/PAD

        # "存活"掩码：一旦某步为 False（遇到 END/PAD），其后的步全 False
        alive = M.cumprod(dim=0) > 0                 # [T,B] bool

        # 真实平均长度（便于日志/早停）
        true_lengths = alive.float().sum(dim=0)  # [B]
        len_mean_realtime = float(true_lengths.mean().item())

        # 2. 计算奖励
        combined_reward, valid_mask_reward, stats, B_scalar, viol, module_extras = compute_combined_reward_v3(
            smiles, modules,
            anti_mode=args.anti_mode,
            anti_weights=args.anti_weights,
            anti_beta=args.anti_beta,
            tox_ceil=args.tox_ceil,
            lam_tox=state.lambda_tox,
            actions=A,
            logprobs=LP,
            masks=alive[:-1, :] & M[1:, :],  # [T-1,B] 用于 KL 的 teacher-forcing
            s4_prior_model=s4_prior.s4_model,
            s4_current_model=s4.s4_model,
            temperature=args.temp_sample,
            fusion_mode=args.fusion_mode,
            step=step_idx  # 新增：传入训练步数用于课程式门控
        )
        # 累计 epoch 内毒性违反均值（按有效样本）
        if (viol.numel() > 0) and (valid_mask_reward.any()):
            vm = viol[valid_mask_reward].mean().item()
            stats_epoch["tox_viol_sum"] += float(vm)
            stats_epoch["tox_viol_cnt"] += 1

        # === batch 级"离散口径"监控（用模块输出：标签 & MIC）===
        if (step_idx == 1) or (step_idx % LOG_EVERY == 0):
            try:
                B = len(smiles)

                tox = (module_extras or {}).get("tox", {})
                aur = (module_extras or {}).get("aureus", {})
                eco = (module_extras or {}).get("ecoli",  {})

                labels = tox.get("labels")  # 长度应为 B，元素∈{0,1,2,3}或None
                mic_a  = aur.get("mic")
                mic_e  = eco.get("mic")

                # —— 毒性：优先 labels → 其次 probs → 最后 R_tox 近似 ——
                if isinstance(labels, list) and len(labels) == B:
                    label_safe_mask = [(l in (1, 2)) if l is not None else False for l in labels]
                    v_tox_mask      = [l is not None for l in labels]
                else:
                    probs = tox.get("probs")
                    if isinstance(probs, list) and len(probs) == B:
                        p_safe = [(p[1] + p[2]) if (p and len(p) == 4) else None for p in probs]
                    else:
                        Rt = tox.get("rewards", None)
                        p_safe = [float(r) if r is not None else None for r in (Rt if isinstance(Rt, list) else [None]*B)]
                    label_safe_mask = [(ps is not None) and (ps > 0.5) for ps in p_safe]
                    v_tox_mask      = [ps is not None for ps in p_safe]

                # —— 双菌：优先 mic 原值 → 否则用 reward≥0.44 近似 MIC<12 ——
                rA = aur.get("rewards"); rE = eco.get("rewards"); thr = 0.44
                both_ok_mask, v_mic_mask = [], []
                for i in range(B):
                    ma = mic_a[i] if isinstance(mic_a, list) and i < len(mic_a) else None
                    me = mic_e[i] if isinstance(mic_e, list) and i < len(mic_e) else None
                    if (ma is not None) and (me is not None):
                        both_ok_mask.append(float(ma) < 12.0 and float(me) < 12.0); v_mic_mask.append(True)
                    else:
                        ra = rA[i] if isinstance(rA, list) and i < len(rA) else None
                        re = rE[i] if isinstance(rE, list) and i < len(rE) else None
                        if (ra is not None) and (re is not None):
                            both_ok_mask.append(float(ra) >= thr and float(re) >= thr); v_mic_mask.append(True)
                        else:
                            both_ok_mask.append(False); v_mic_mask.append(False)

                n_tox = max(1, sum(v_tox_mask)); n_mic = max(1, sum(v_mic_mask))
                safe_rate = 100.0 * (sum(1 for s, v in zip(label_safe_mask, v_tox_mask) if v and s) / n_tox)
                both12    = 100.0 * (sum(1 for b, v in zip(both_ok_mask, v_mic_mask) if v and b) / n_mic)

                five_hit = 0
                if isinstance(mic_a, list) and isinstance(mic_e, list):
                    for i in range(B):
                        if (labels and i < len(labels) and (labels[i] in (1, 2)) and
                            (mic_a[i] is not None) and (mic_e[i] is not None) and
                            float(mic_a[i]) < 3.0 and float(mic_e[i]) < 3.0):
                            five_hit += 1

                print(f"[hard-metrics step={step_idx}] "
                      f"batch_label_safe_rate={safe_rate:.2f}% | "
                      f"batch_both_mic<12_rate={both12:.2f}% | "
                      f"batch_five_hit={five_hit} | "
                      f"valid_tox={n_tox} valid_mic={n_mic}")
            except Exception as e:
                print(f"[hard-metrics step={step_idx}] ERROR: {e}")

        # 3. 三唑结构过滤（软门控：RDKit 或 前缀，避免一锅端）
        triazole_mask_cpu = torch.tensor([validate_triazole(smi) for smi in smiles], dtype=torch.bool)
        prefix_mask_cpu = torch.tensor([smi.startswith(args.prefix) for smi in smiles], dtype=torch.bool)
        chem_mask_cpu = triazole_mask_cpu | prefix_mask_cpu

        # 4. 最终有效样本：奖励有效 AND 化学门
        chem_mask = chem_mask_cpu.to(valid_mask_reward.device)
        valid_mask_gpu = valid_mask_reward & chem_mask
        valid_mask_cpu = valid_mask_gpu.detach().to('cpu')

        # 4.1 兜底：若整批仍然空（极端情况），退回“只看奖励有效”，以保证有梯度
        if not valid_mask_cpu.any():
            valid_mask_gpu = valid_mask_reward
            valid_mask_cpu = valid_mask_reward.detach().to('cpu')

        # 5. 策略梯度更新（只对有效样本）
        if valid_mask_cpu.any():
            # GPU 张量用 GPU 口罩
            filtered_rewards = combined_reward[valid_mask_gpu]
            mean_reward = filtered_rewards.mean().item()
            baseline_value = ema.current() if ema.value is not None else mean_reward
            advantage = filtered_rewards - baseline_value  # 与 filtered_rewards 同设备（GPU）

            # 分离两套掩码
            dev = filtered_rewards.device  # 与 advantage 同设备（通常是 CUDA）
            mask_pg = alive.to(dev)[:, valid_mask_gpu].float()   # [T,Bv]   用于 policy 的序列 logprob
            LP_sel  = LP.to(dev)[:, valid_mask_gpu]              # [T,Bv]

            # 用 mask_pg 计算序列级 logprob
            den_pg      = mask_pg.sum(dim=0).clamp(min=1.0)                      # [Bv]
            seq_logprob = (LP_sel * mask_pg).sum(dim=0) / den_pg                 # [Bv]
            loss = -(advantage.detach() * seq_logprob).mean()

            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(s4.s4_model.parameters(), args.max_grad_norm)
            optimizer.step()

            ema.update(mean_reward)
        else:
            # 不写 0，沿用 EMA（或你也可以写成 float('nan') 再在画图时用 nanmean）
            mean_reward = ema.current()
            # 可选：极端保护——批有效率接近 0 时，临时放松 λ，防止下一批继续“压塌”
            if state.lambda_tox.item() > 0:
                state.lambda_tox = state.lambda_tox * 0.8

        valid_rate = valid_mask_cpu.float().mean().item()
        triazole_rate = triazole_mask_cpu.float().mean().item()
        history.append((step_idx, mean_reward, valid_rate, triazole_rate))
        epoch_rewards.append(mean_reward)

        # --- step 级 λ 更新：每 _LAGRANGE_EVERY 步，用 viol 的 q90 抬 λ ---
        if (step_idx % _LAGRANGE_EVERY == 0) and valid_mask_gpu.any():
            v = viol[valid_mask_gpu].detach()
            if v.numel() > 0:
                target = v.quantile(_TOX_Q)
                lam_old = state.lambda_tox
                state.lambda_tox = torch.clamp(lam_old + _LAGRANGE_LR * target,
                                               min=0.0, max=_LAGRANGE_MAX)
                print(f"[lagrange-step] λ_tox {lam_old.item():.4f} → {state.lambda_tox.item():.4f} "
                      f"(viol_q{int(_TOX_Q*100)}={float(target):.4f})")

        # 6. 日志打印
        if step_idx % LOG_EVERY == 0 or step_idx == 1:
            elapsed = time.time() - start_time
            per_module_summary = ", ".join(
                f"{name}: mean={stats[name]['mean_reward']:.3f}, valid={stats[name]['valid_rate']:.2f}"
                for name in stats
                if name not in ["kl_regularization", "length_penalty", "antibacterial_scalar", "tox_constraint"]  # 排除统计项
            )

            # 提取KL和长度统计
            kl_mean = stats.get("kl_regularization", {}).get("mean_kl_ptok", 0.0)
            len_mean = stats.get("length_penalty", {}).get("mean_length", 0.0)

            # ★ 提取毒性约束统计（用于诊断）
            tox_stats = stats.get("tox_constraint", {})
            p_safe_mean = tox_stats.get("mean_p_safe", float('nan'))  # 改为 p_safe 均值
            tox_viol_mean = tox_stats.get("mean_violation", 0.0)
            viol_q90 = tox_stats.get("viol_q90", 0.0)

            print(
                f"[step {step_idx:6d}] reward={mean_reward:.4f} valid={valid_rate:.3f} "
                f"baseline={ema.current():.4f} triazole={triazole_rate:.3f} "
                f"time={elapsed/60:.2f}m | {per_module_summary} | "
                f"B={(B_scalar[valid_mask_gpu].mean().item() if valid_mask_gpu.any() else float('nan')):.3f} "
                f"viol={(viol[valid_mask_gpu].mean().item() if valid_mask_gpu.any() else 0.0):.3f} "
                f"KL/token={kl_mean:.6f} len={len_mean:.1f}"
            )

            # ★ 新增：诊断输出（监控 p_safe 和违反量）
            if not np.isnan(p_safe_mean):
                viol_mean = (viol[valid_mask_gpu].mean().item() if valid_mask_gpu.any() else 0.0)
                # 计算 tau 和 warm 用于显示
                warm_val = min(float(step_idx) / 1000.0, 1.0)
                tau_target = 1.0 - args.tox_ceil
                tau_eff = 0.58 + warm_val * (tau_target - 0.58)  # 当前生效的 tau

                # ★ 真正的达标率：p_safe >= tau_eff 的占比（用概率口径重算，使用当前生效的 tau）
                tox_probs_raw = module_extras.get("tox", {}).get("probs", None)
                if tox_probs_raw and isinstance(tox_probs_raw, (list, np.ndarray)):
                    tp = torch.tensor(
                        [p if (isinstance(p, (list, tuple)) and len(p) == 4) else [0, 0, 0, 0] for p in tox_probs_raw],
                        dtype=torch.float32, device=valid_mask_gpu.device
                    )
                    p_safe_batch = (tp[:, 1] + tp[:, 2]).clamp(0, 1)
                    rate_above_tau = (p_safe_batch >= tau_eff).float().mean().item()  # ← 真正的"≥tau_eff 占比"
                    rate_below_tau = 1.0 - rate_above_tau

                    # ★ 四类概率和检查（应≈1，诊断概率口径是否生效）
                    probs_sums = [sum(p) for p in tox_probs_raw if (isinstance(p, (list, tuple)) and len(p) == 4)]
                    if probs_sums:
                        mean_prob_sum = np.mean(probs_sums)
                        min_prob_sum = np.min(probs_sums)
                        max_prob_sum = np.max(probs_sums)
                        prob_sum_diag = f"prob_sum=[{min_prob_sum:.3f}, {mean_prob_sum:.3f}, {max_prob_sum:.3f}]"
                    else:
                        prob_sum_diag = "prob_sum=N/A"
                else:
                    rate_above_tau = float('nan')
                    rate_below_tau = float('nan')
                    prob_sum_diag = "prob_sum=MISSING"

                print(f"[diag] p_safe_mean={p_safe_mean:.4f} tau_eff={tau_eff:.3f} (→{tau_target:.2f}) warm={warm_val:.3f} "
                      f"viol_mean={viol_mean:.4f} viol_q90={viol_q90:.4f} "
                      f"rate_above_tau={rate_above_tau:.3f} rate_below_tau={rate_below_tau:.3f} "
                      f"len_realtime={len_mean_realtime:.1f} {prob_sum_diag}")

        # 7. 定期保存检查点
        if step_idx % args.save_every == 0:
            ckpt_dir = os.path.join(dirs["checkpoints"], f"step_{step_idx:06d}")
            save_checkpoint(s4, ckpt_dir)

        # 8. Epoch结束判断 + 早停检查
        if step_idx % steps_per_epoch == 0:
            epoch_idx = step_idx // steps_per_epoch
            epoch_mean = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
            epoch_rewards.clear()
            improved = epoch_mean > (best_epoch_score + args.min_delta)

            if improved:
                best_epoch_score = epoch_mean
                best_state = {
                    "model": {k: v.detach().cpu().clone() for k, v in s4.s4_model.state_dict().items()},
                    "step": step_idx,
                    "epoch": epoch_idx,
                    "mean_reward": epoch_mean,
                }
                patience_counter = 0
                print(f"[epoch {epoch_idx:3d}] Improvement detected (mean={epoch_mean:.4f}); snapshotting best state.")
            else:
                patience_counter += 1
                print(
                    f"[epoch {epoch_idx:3d}] No improvement (mean={epoch_mean:.4f}); "
                    f"patience {patience_counter}/{args.patience}."
                )


            # —— 拉格朗日 λ 更新 ——
            if stats_epoch["tox_viol_cnt"] > 0:
                mean_viol = stats_epoch["tox_viol_sum"] / max(stats_epoch["tox_viol_cnt"], 1)
            else:
                mean_viol = 0.0
            lam_old = state.lambda_tox
            state.lambda_tox = torch.clamp(lam_old + args.lagrange_lr * torch.tensor(mean_viol, device=device),
                                           min=0.0, max=args.lagrange_max)
            print(f"[lagrange] λ_tox {lam_old.item():.4f} → {state.lambda_tox.item():.4f} (viol={mean_viol:.4f})")
            stats_epoch = {"tox_viol_sum": 0.0, "tox_viol_cnt": 0}

            # ★ 自检报告：probs缺失统计（应始终为0）
            print(f"[self-check] tox_probs_missing={stats_diag['tox_probs_missing']} "
                  f"tox_probs_invalid_shape={stats_diag['tox_probs_invalid_shape']} "
                  f"(both should be 0)")

            if patience_counter >= args.patience:
                print("Early stopping triggered by patience.")
                break
            if epoch_idx >= args.epochs:
                print("Reached maximum epoch budget.")
                break

    save_history(history, args.output_dir)

    # 9. 恢复最佳权重
    if best_state is not None:
        s4.s4_model.load_state_dict(best_state["model"])
        print(
            f"Best model restored from step {best_state['step']} "
            f"(epoch {best_state['epoch']}), mean reward {best_state['mean_reward']:.4f}."
        )
    else:
        print("Warning: no improvement observed; saving current state as reinforced S4 model.")

    # 10. 保存最终模型
    s4.save(args.output_dir)

    metadata = {
        "version": "v3",
        "fusion_mode": args.fusion_mode,
        "best_step": best_state["step"] if best_state else None,
        "best_epoch": best_state["epoch"] if best_state else None,
        "best_epoch_mean_reward": best_state["mean_reward"] if best_state else None,
        "channels_enabled": enabled,
        "temperature": args.temp_sample,
        "base_model": args.base_model,
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("=" * 70)
    print(f"Reinforced S4 v3 saved to {args.output_dir}")
    print("=" * 70)
    print("Training complete!")


if __name__ == "__main__":
    main()
