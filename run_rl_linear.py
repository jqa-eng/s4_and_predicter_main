#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear multi-objective RL baseline aligned to the senior author's paper.

Reward: R = w_aureus * r_aureus + w_ecoli * r_ecoli + w_tox * r_tox
  - r_aureus = sigmoid((log10(MIC_center) - log10(MIC_aureus)) * sharpness)
  - r_ecoli  = sigmoid((log10(MIC_center) - log10(MIC_ecoli )) * sharpness)
  - r_tox    = predicted safety probability (0-1)

All channels are continuous in (0,1); no constraints, penalties, or curricula.
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

from reward_modules import AureusRewardModule, EcoliRewardModule, ToxicityRewardModule
from s4dd import S4forDenovoDesign


# --- Paper-aligned shaping constants ---
MIC_SIG_CENTER = 3.0
MIC_SIG_SHARPNESS = 2.0

# --- Training defaults ---
EMA_DECAY = 0.95
LOG_EVERY = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linear weighted multi-objective RL baseline (paper-aligned)."
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Directory containing the pretrained S4 weights (init_arguments.json + model.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reinforced_s4_linear",
        help="Directory where checkpoints, history, and best weights are stored.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="c1n[nH]nc1",
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
        default=100,
        help="Logical epoch count for early stopping statistics (1 = disabled).",
    )
    parser.add_argument(
        "--temp-sample",
        type=float,
        default=1.1,
        help="Sampling temperature for exploration rollouts.",
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
        default=12,
        help="Number of bad epochs to allow before early stopping (set high to disable).",
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
        "--weights",
        type=str,
        default="aureus=2.0,ecoli=2.0,tox=1.0",
        help="Linear reward weights in key=value format (comma separated).",
    )
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


def mic_to_sigmoid(mic_tensor: torch.Tensor) -> torch.Tensor:
    mic_clamped = mic_tensor.clamp_min(1e-8)
    logmic = torch.log10(mic_clamped)
    return torch.sigmoid((math.log10(MIC_SIG_CENTER) - logmic) * MIC_SIG_SHARPNESS)


def compute_combined_reward_linear(
    smiles: List[str],
    modules: Dict[str, object],
    weights: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Dict[str, float]]]:
    batch_size = len(smiles)
    if batch_size == 0:
        raise ValueError("Empty SMILES batch.")

    module_rewards: Dict[str, torch.Tensor] = {}
    module_valid: Dict[str, torch.Tensor] = {}
    module_extras: Dict[str, Dict[str, object]] = {}
    stats: Dict[str, Dict[str, float]] = {}

    for name, module in modules.items():
        result = module.compute_reward(smiles)
        rewards = result.get("rewards", [])
        if len(rewards) != batch_size:
            raise ValueError(f"Reward module '{name}' returned {len(rewards)} values for batch of size {batch_size}.")
        reward_tensor = torch.tensor([r if r is not None else 0.0 for r in rewards], dtype=torch.float32)
        valid_tensor = torch.tensor([r is not None for r in rewards], dtype=torch.bool)
        module_rewards[name] = reward_tensor
        module_valid[name] = valid_tensor
        module_extras[name] = result
        stats[name] = {
            "valid_rate": valid_tensor.float().mean().item(),
            "mean_reward": reward_tensor[valid_tensor].mean().item() if valid_tensor.any() else float("nan"),
        }

    pos_key = next((k for k in module_rewards.keys() if "aureus" in k.lower()), None)
    neg_key = next((k for k in module_rewards.keys() if "ecoli" in k.lower()), None)
    tox_key = next((k for k in module_rewards.keys() if "tox" in k.lower()), None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def channel_from_mic(key: str, valid_tensor: torch.Tensor, reward_tensor: torch.Tensor):
        extras = module_extras.get(key, {}) if key else {}
        mic_vals = extras.get("mic")
        if mic_vals is not None:
            mic_tensor = torch.tensor(
                [m if m is not None else 0.0 for m in mic_vals],
                dtype=torch.float32,
            )
            mic_valid = torch.tensor([m is not None for m in mic_vals], dtype=torch.bool)
            return mic_to_sigmoid(mic_tensor.to(device)), mic_valid.to(device)
        # Fallback to provided reward (already 0-1)
        return reward_tensor.to(device), valid_tensor.to(device)

    if pos_key:
        r_pos, v_pos = channel_from_mic(pos_key, module_valid[pos_key], module_rewards[pos_key])
    else:
        r_pos = torch.zeros(batch_size, dtype=torch.float32, device=device)
        v_pos = torch.zeros(batch_size, dtype=torch.bool, device=device)

    if neg_key:
        r_neg, v_neg = channel_from_mic(neg_key, module_valid[neg_key], module_rewards[neg_key])
    else:
        r_neg = torch.zeros(batch_size, dtype=torch.float32, device=device)
        v_neg = torch.zeros(batch_size, dtype=torch.bool, device=device)

    if tox_key:
        r_tox = module_rewards[tox_key].to(device).clamp(0.0, 1.0)
        v_tox = module_valid[tox_key].to(device)
    else:
        r_tox = torch.zeros(batch_size, dtype=torch.float32, device=device)
        v_tox = torch.zeros(batch_size, dtype=torch.bool, device=device)

    combined = (
        weights.get("aureus", 0.0) * r_pos +
        weights.get("ecoli", 0.0) * r_neg +
        weights.get("tox", 0.0) * r_tox
    )

    valid_mask = v_pos | v_neg | v_tox

    stats["combined"] = {
        "valid_rate": valid_mask.float().mean().item(),
        "mean_reward": combined[valid_mask].mean().item() if valid_mask.any() else float("nan"),
    }

    return combined, valid_mask, stats


def ensure_dirs(output_dir: str) -> Dict[str, str]:
    checkpoints = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints, exist_ok=True)
    return {"output": output_dir, "checkpoints": checkpoints}


def save_history(history: List[Tuple[int, float, float]], output_dir: str) -> None:
    if not history:
        return
    csv_path = os.path.join(output_dir, "reward_history.csv")
    png_path = os.path.join(output_dir, "reward_curve.png")
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("step,mean_reward,valid_rate\n")
        for step, mean_reward, valid_rate in history:
            handle.write(f"{step},{mean_reward:.6f},{valid_rate:.6f}\n")

    steps = [item[0] for item in history]
    means = [item[1] for item in history]
    plt.figure()
    plt.plot(steps, means, label="mean reward")
    plt.xlabel("step")
    plt.ylabel("mean reward")
    plt.title("RL reward curve (linear baseline)")
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
    weights = parse_weights(args.weights, enabled)
    dirs = ensure_dirs(args.output_dir)

    print("=" * 70)
    print("=== Linear RL Configuration (paper-aligned) ===")
    print("=" * 70)
    print(f"Base model       : {args.base_model}")
    print(f"Output directory : {args.output_dir}")
    print(f"Device           : {device}")
    print(f"Channels enabled : {enabled}")
    print(f"Weights          : {weights}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Max steps        : {args.max_steps}")
    print(f"Epochs           : {args.epochs}")
    print(f"Patience         : {args.patience}")
    print(f"Temperature      : {args.temp_sample}")
    print(f"Learning rate    : {args.lr}")
    print(f"Min delta        : {args.min_delta}")
    print("=" * 70)

    if not os.path.exists(args.base_model):
        raise FileNotFoundError(f"Base model directory not found: {args.base_model}")

    model_pt_path = os.path.join(args.base_model, "model.pt")
    init_args_path = os.path.join(args.base_model, "init_arguments.json")
    if not os.path.exists(model_pt_path):
        raise FileNotFoundError(f"model.pt not found in base model: {model_pt_path}")
    if not os.path.exists(init_args_path):
        raise FileNotFoundError(f"init_arguments.json not found in base model: {init_args_path}")

    s4 = S4forDenovoDesign.from_file(args.base_model)
    s4.device = str(device)
    s4.s4_model.to(device)

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
    history: List[Tuple[int, float, float]] = []
    epoch_rewards: List[float] = []
    best_epoch_score = -math.inf
    best_state = None
    patience_counter = 0
    start_time = time.time()

    print("Starting linear RL training loop...")
    print()

    for step_idx in range(1, args.max_steps + 1):
        actions, logprobs, masks, smiles = s4.rollout(
            batch_size=args.batch_size,
            max_len=s4.sequence_length,
            required_prefix_smiles=args.prefix,
            temperature=args.temp_sample,
        )

        A, LP, M = stack_traces(actions, logprobs, masks)

        combined_reward, valid_mask_cpu, stats = compute_combined_reward_linear(
            smiles, modules, weights
        )

        valid_mask = valid_mask_cpu.to(LP.device)

        if valid_mask.any():
            filtered_rewards = combined_reward[valid_mask_cpu]
            mean_reward = filtered_rewards.mean().item()
            baseline_value = ema.current() if ema.value is not None else mean_reward
            advantage = filtered_rewards.to(LP.device) - baseline_value

            token_mask = M[:, valid_mask].float()
            token_mask_sum = token_mask.sum(dim=0).clamp(min=1.0)
            seq_logprob = (LP[:, valid_mask] * token_mask).sum(dim=0) / token_mask_sum
            loss = -(advantage.detach() * seq_logprob).mean()

            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(s4.s4_model.parameters(), args.max_grad_norm)
            optimizer.step()

            ema.update(mean_reward)
        else:
            mean_reward = ema.current()

        valid_rate = valid_mask_cpu.float().mean().item()
        history.append((step_idx, mean_reward, valid_rate))
        epoch_rewards.append(mean_reward)

        if step_idx % LOG_EVERY == 0 or step_idx == 1:
            elapsed = time.time() - start_time
            per_module_summary = ", ".join(
                f"{name}: mean={stats[name]['mean_reward']:.3f}, valid={stats[name]['valid_rate']:.2f}"
                for name in stats
                if name != "combined"
            )
            print(
                f"[step {step_idx:6d}] reward={mean_reward:.4f} valid={valid_rate:.3f} "
                f"baseline={ema.current():.4f} time={elapsed/60:.2f}m | {per_module_summary}"
            )

        if step_idx % args.save_every == 0:
            ckpt_dir = os.path.join(dirs["checkpoints"], f"step_{step_idx:06d}")
            save_checkpoint(s4, ckpt_dir)

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

            if patience_counter >= args.patience:
                print("Early stopping triggered by patience.")
                break
            if epoch_idx >= args.epochs:
                print("Reached maximum epoch budget.")
                break

    save_history(history, args.output_dir)

    if best_state is not None:
        s4.s4_model.load_state_dict(best_state["model"])
        print(
            f"Best model restored from step {best_state['step']} "
            f"(epoch {best_state['epoch']}), mean reward {best_state['mean_reward']:.4f}."
        )
    else:
        print("Warning: no improvement observed; saving current state as reinforced S4 model.")

    s4.save(args.output_dir)

    metadata = {
        "version": "linear",
        "best_step": best_state["step"] if best_state else None,
        "best_epoch": best_state["epoch"] if best_state else None,
        "best_epoch_mean_reward": best_state["mean_reward"] if best_state else None,
        "channels_enabled": enabled,
        "temperature": args.temp_sample,
        "base_model": args.base_model,
        "weights": weights,
        "mic_sig_center": MIC_SIG_CENTER,
        "mic_sig_sharpness": MIC_SIG_SHARPNESS,
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("=" * 70)
    print(f"Linear RL baseline saved to {args.output_dir}")
    print("=" * 70)
    print("Training complete!")


if __name__ == "__main__":
    main()
