#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reinforcement learning driver v3 - 最小权衡+约束版 - 修复关键问题版本

主要改进：
1. 软融合奖励门控（替代硬AND门）- 解决有效样本率过低问题
2. 放宽早停策略 - 避免过早停止训练
3. 明确底模路径 - 防止误用旧模型
4. 增强探索 + 降低毒性权重 - 提高学习信号
5. 统一三唑SMARTS过滤器 - 避免训练/评测不一致

关键修复点：
- compute_combined_reward_v2: 软融合逻辑，只要有一个模块有效就保留样本
- 默认epochs=1（禁用早停），patience=9999（兜底）
- 默认temp_sample=1.2（增强探索）
- 默认权重: tox=0.2, aureus=0.5, ecoli=0.3（降低毒性卡脖子）
- 统一使用TRIAZOLE_PATTERNS（训练+评测）
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
_LAGRANGE_LR   = 0.20          # λ 的步长（对偶上升）
_LAGRANGE_MAX  = 5.0           # λ 上限
_LAGRANGE_EVERY= 10            # 每隔多少 step 更新一次 λ
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
        default=1.1,  # 提高探索温度
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
        default="soft",  # 默认使用软融合
        help="Reward fusion mode: 'soft' = at least one module valid; 'hard' = all modules valid (v1 behavior).",
    )
    
# --- 抗菌性两路权衡 ---
    parser.add_argument("--anti-mode", choices=["cheby","softmin","linear"], default="cheby",
                        help="aureus/ecoli 的权衡：cheby=推短板，softmin=平滑最小化，linear=原线性加权")
    parser.add_argument("--anti-weights", type=str, default="pos=0.5,neg=0.5",
                        help="阳/阴两路权重，如 pos=0.5,neg=0.5")
    parser.add_argument("--anti-beta", type=float, default=6.0,
                        help="softmin 锐度（越大越接近最小值）")

    # --- 毒性拉格朗日约束 ---
    parser.add_argument("--tox-ceil", type=float, default=0.40,
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
    fusion_mode: str = "soft"
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Dict[str, float]], torch.Tensor, torch.Tensor]:
    """
    v3: 两路抗菌性做“短板优先”标量化 + 毒性做拉格朗日约束；
        仍保留“软融合有效性”的思想：只要抗菌或毒性任一有效即可形成梯度。
    返回: combined, valid_mask, stats, B, viol
    """
    batch_size = len(smiles)
    if batch_size == 0:
        raise ValueError("Empty SMILES batch.")

    module_rewards: Dict[str, torch.Tensor] = {}
    module_valid: Dict[str, torch.Tensor] = {}
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
        valid_B = (v_pos & v_neg) if (fusion_mode == "hard") else (v_pos | v_neg)
    elif r_pos is not None:
        B = r_pos; valid_B = v_pos
    elif r_neg is not None:
        B = r_neg; valid_B = v_neg
    # else: 维持默认 0 和 False

    # 4. 毒性为约束：combined = B - lam * relu(p_tox - ceil)
    if tox_key is not None:
        R_tox = module_rewards[tox_key].to(device).float()
        v_tox = module_valid[tox_key].to(device)
        # 数值稳健：先夹在 (0,1) 内，再换到"有毒概率"口径
        R_tox = R_tox.clamp_(1e-6, 1.0 - 1e-6)
        p_tox = 1.0 - R_tox
        # ★ 用平方铰链：超阈越多，罚得越重
        viol = F.relu(p_tox - float(tox_ceil))
        penalty = lam_tox * viol
        base_reward = B - penalty
        valid_mask = valid_B | v_tox
    else:
        base_reward = B
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
        stats["tox_constraint"] = {"valid_rate": v_tox.float().mean().item(),
                                   "mean_reward": R_tox[v_tox].mean().item() if v_tox.any() else float("nan"),
                                   "mean_violation": viol[v_tox].mean().item() if v_tox.any() else 0.0}

    # 添加KL和长度统计
    stats["kl_regularization"] = {"mean_kl_ptok": kl_ptok.mean().item()}
    stats["length_penalty"] = {"mean_length": len_ratio.mean().item() * LEN_MAX + LEN_MAX}

    return combined, valid_mask, stats, B, viol




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
        combined_reward, valid_mask_reward, stats, B_scalar, viol = compute_combined_reward_v3(
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
            fusion_mode=args.fusion_mode
        )
        # 累计 epoch 内毒性违反均值（按有效样本）
        if (viol.numel() > 0) and (valid_mask_reward.any()):
            vm = viol[valid_mask_reward].mean().item()
            stats_epoch["tox_viol_sum"] += float(vm)
            stats_epoch["tox_viol_cnt"] += 1

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
            tox_mean = tox_stats.get("mean_reward", float('nan'))  # R_tox均值
            tox_viol_mean = tox_stats.get("mean_violation", 0.0)

            print(
                f"[step {step_idx:6d}] reward={mean_reward:.4f} valid={valid_rate:.3f} "
                f"baseline={ema.current():.4f} triazole={triazole_rate:.3f} "
                f"time={elapsed/60:.2f}m | {per_module_summary} | "
                f"B={(B_scalar[valid_mask_gpu].mean().item() if valid_mask_gpu.any() else float('nan')):.3f} "
                f"viol={(viol[valid_mask_gpu].mean().item() if valid_mask_gpu.any() else 0.0):.3f} "
                f"KL/token={kl_mean:.6f} len={len_mean:.1f}"  # ★ KL精度提升到6位
            )

            # ★ 新增：诊断输出（监控毒性和真实长度）
            if not np.isnan(tox_mean):
                ptox_mean = 1.0 - tox_mean
                rate_above = (viol[valid_mask_gpu] > 0).float().mean().item() if valid_mask_gpu.any() else float('nan')
                viol_mean = (viol[valid_mask_gpu].mean().item() if valid_mask_gpu.any() else 0.0)
                print(f"[diag] Rtox_mean={tox_mean:.4f} ptox_mean={ptox_mean:.4f} "
                      f"viol_mean={viol_mean:.4f} rate_above_ceil={rate_above:.3f} "
                      f"len_realtime={len_mean_realtime:.1f}")

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
