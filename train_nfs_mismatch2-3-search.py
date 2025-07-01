# train_nfs_refactored.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from nflows.transforms.normalization import BatchNorm
from nflows.transforms.permutations import RandomPermutation
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform
)
from nflows.transforms.permutations import RandomPermutation
from nflows.distributions.mixture import MADEMoG
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from torch.distributions import Categorical

from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f"logs/{timestamp}"
os.makedirs(log_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

# Support for Apple Silicon MPS
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")
# else:
# device = torch.device("cpu")

print(f"Using device: {device}")

def save_metadata(log_dir: str, model_config: dict, training_config: dict, data_config: dict):
    """Save training metadata to log directory."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "model_config": model_config,
        "training_config": training_config,
        "data_config": data_config
    }
    
    metadata_path = os.path.join(log_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✔️  Metadata saved to {metadata_path}")
    return metadata_path

def load_dataset(x_path: Path | str, y_path: Path | str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read CSVs, align length mismatches, and return **float32** tensors."""
    x_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)

    if len(x_df) != len(y_df):
        print(
            f"⚠️  Length mismatch – x: {len(x_df)} rows, y: {len(y_df)} rows. "
            "Truncating to smallest."
        )
    n = min(len(x_df), len(y_df))
    x_tensor = torch.tensor(x_df.iloc[:n].values, dtype=torch.float32)
    y_tensor = torch.tensor(y_df.iloc[:n].values.reshape(-1, 1), dtype=torch.float32)
    return x_tensor, y_tensor


def _init_identity(transform):
    # assume transform has an internal autoregressive net
    for module in transform.autoregressive_net.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.xavier_uniform_(module.weight)

def build_nfs_model(context_features, flow_features=1, hidden_features=16,
                    num_layers=3, num_bins=15, num_mixture_components: int = 3):
    transforms = []
    for _ in range(num_layers):
        # spline block
        spline = MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=flow_features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_bins=num_bins,
            tails="linear",
            tail_bound=4.0,
        )
        _init_identity(spline)
        transforms.append(spline)
        transforms.append(BatchNorm(features=flow_features))

        # affine block
        affine = MaskedAffineAutoregressiveTransform(
            features=flow_features,
            hidden_features=hidden_features,
            context_features=context_features,
        )
        _init_identity(affine)
        transforms.append(affine)
        transforms.append(RandomPermutation(features=flow_features))

    base_dist = MADEMoG(
        features=flow_features,
        hidden_features=hidden_features,
        context_features=context_features,
        num_mixture_components=num_mixture_components,
    )
    
    flow = Flow(CompositeTransform(transforms), base_dist).float()
    
    # Return model config for metadata
    model_config = {
        "model_type": "MaskedPiecewiseRationalQuadraticAutoregressiveTransform",
        "context_features": context_features,
        "flow_features": flow_features,
        "hidden_features": hidden_features,
        "num_layers": num_layers,
        "num_bins": num_bins,
        "base_distribution": "StandardNormal",
        "total_parameters": sum(p.numel() for p in flow.parameters())
    }
    
    return flow, { **model_config, "num_mixture_components": num_mixture_components }

def train(model: Flow, train_loader: DataLoader, test_loader: DataLoader = None, 
          *, epochs: int = 200, lr: float = 5e-3) -> dict:
    """Single‑loop optimiser with test loss tracking and training plots."""
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Track losses
    train_losses = []
    test_losses = []
    
    print(f"🚀 Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        ep_train_loss = 0.0
        for cx, y in train_loader:
            cx, y = cx.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = -model.log_prob(inputs=y, context=cx).mean()
            loss.backward()
            opt.step()
            ep_train_loss += loss.item()
        ep_train_loss /= len(train_loader)
        train_losses.append(ep_train_loss)
        
        # Test phase (if test_loader provided)
        ep_test_loss = None
        if test_loader is not None:
            model.eval()
            ep_test_loss = 0.0
            with torch.no_grad():
                for cx, y in test_loader:
                    cx, y = cx.to(device), y.to(device)
                    loss = -model.log_prob(inputs=y, context=cx).mean()
                    ep_test_loss += loss.item()
                ep_test_loss /= len(test_loader)
                test_losses.append(ep_test_loss)

        # Logging
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            log_msg = f"Epoch {epoch:3d}/{epochs} | train nll {ep_train_loss:.4f}"
            if ep_test_loss is not None:
                log_msg += f" | test nll {ep_test_loss:.4f}"
            print(log_msg)

    # Save model
    torch.save(model.state_dict(), os.path.join(log_dir, "trained_flow.pt"))
    print(f"✔️  Training complete – model saved to {log_dir}/trained_flow.pt")
    
    # Create and save training plot
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
    if test_losses:
        plt.plot(epochs_range, test_losses, 'r-', label='Test Loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('Training and Test Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(log_dir, "training_curves.pdf")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"✔️  Training plot saved to {plot_path}")
    
    # Save loss data
    loss_data = {
        'epoch': list(epochs_range),
        'train_loss': train_losses,
        'test_loss': test_losses if test_losses else [None] * epochs
    }
    loss_df = pd.DataFrame(loss_data)
    loss_csv_path = os.path.join(log_dir, "training_losses.csv")
    loss_df.to_csv(loss_csv_path, index=False)
    print(f"✔️  Loss data saved to {loss_csv_path}")
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1] if test_losses else None
    }


def evaluate(
    model: Flow,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    samples_per_cond: int = 100,
    eval_subset: int | None = None,
    batch_size: int = 512,
    save_path: str = "images/evaluation.pdf",
    error_csv_path: str = "errors/setup_errors.csv"
) -> torch.Tensor:
    """Enhanced evaluation with more metrics."""
    model.eval()
    eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(eval_device)
    # 从测试集中选择4个不同的setup进行可视化
    test_setup_cols = x_test[:, :3]  # 前3列是setup特征
    test_unique_setups, test_indices = torch.unique(test_setup_cols, dim=0, return_inverse=True)
    
    # 选择4个setup用于画图
    n_viz_setups = min(4, len(test_unique_setups))
    selected_setups = torch.randperm(len(test_unique_setups))[:n_viz_setups]
    
    # 对选中的4个setup画图
    for i, setup_idx in enumerate(selected_setups):
        setup_mask = test_indices == setup_idx
        x_setup = x_test[setup_mask]
        y_setup = y_test[setup_mask]
        
        if len(x_setup) == 0:
            continue
            
        print(f"\n🎨 Visualizing Setup {i+1}/4 (Setup ID: {setup_idx}) - {len(x_setup)} samples")
        print(f"Setup parameters: {test_unique_setups[setup_idx].tolist()}")
        
        # 为该setup生成可视化数据
        empirical = []
        generated = []
        log_probs = []
        
        with torch.no_grad():
            for start in range(0, len(x_setup), batch_size):
                cx = x_setup[start : start + batch_size].to(eval_device)
                y = y_setup[start : start + batch_size]
                
                try:
                    # ---- GPU‐only sampling ----
                    B = cx.size(0)
                    # draw noise on GPU
                    z = torch.randn(samples_per_cond, B, 1, device=eval_device)
                    # flatten so we can call inverse
                    z_flat = z.view(-1, 1)
                    # replicate context
                    ctx_rep = cx.unsqueeze(0).expand(samples_per_cond, -1, -1) \
                                 .reshape(-1, cx.size(-1))
                    # inverse‐transform back to data‐space
                    x_flat, _ = model._transform.inverse(z_flat, context=ctx_rep)
                    batch_samples = x_flat.view(samples_per_cond, B, -1).cpu()

                    # log_probs as before (this one is safe: we .to(eval_device) first)
                    batch_log_probs = model.log_prob(
                        inputs=y.to(eval_device),
                        context=cx
                    ).cpu()
                    
                    generated.append(batch_samples)
                    empirical.append(y.repeat(samples_per_cond, 1))
                    log_probs.append(batch_log_probs)
                except Exception as e:
                    print(f"⚠️ Error in visualization batch: {e}")
                    continue
        
        if generated:
            y_emp = torch.cat(empirical).numpy().flatten()
            y_gen = torch.cat(generated).numpy().flatten()
            all_log_probs = torch.cat(log_probs).numpy()
            
            # 为每个setup创建单独的图
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            sns.kdeplot(y_emp, label="Empirical", fill=True, alpha=0.5)
            sns.kdeplot(y_gen, label="Generated", fill=True, alpha=0.5)
            plt.title(f"Setup {i+1} Distribution Overlap")
            plt.xlim([-0.1, 1.1])
            plt.legend()
            
            plt.subplot(1, 3, 2)
            percs = np.linspace(1, 99, 99)
            plt.scatter(
                np.percentile(y_emp, percs),
                np.percentile(y_gen, percs),
                s=8, alpha=0.7
            )
            lims = [y_emp.min(), y_emp.max()]
            plt.plot(lims, lims, "r--", alpha=0.8)
            plt.title(f"Setup {i+1} Q–Q Plot")
            plt.xlabel("Empirical Quantiles")
            plt.ylabel("Generated Quantiles")
            
            plt.subplot(1, 3, 3)
            plt.hist(all_log_probs, bins=50, alpha=0.7, density=True)
            plt.title(f"Setup {i+1} Log-Likelihood")
            plt.xlabel("Log Probability")
            plt.ylabel("Density")
            
            plt.tight_layout()
            plt.savefig(f"{log_dir}/evaluation_setup_{i+1}.pdf", bbox_inches='tight')
            plt.show()

    # ✅ 计算所有测试集setup的平均误差
    print(f"\n📊 Computing errors for all {len(test_unique_setups)} test setups...")
    model.to(device)  # ← ADD THIS LINE TOO

    all_setup_errors = []
    setup_error_details = []
    
    for setup_idx in range(len(test_unique_setups)):
        setup_mask = test_indices == setup_idx
        x_setup = x_test[setup_mask]
        y_setup = y_test[setup_mask]
        
        if len(x_setup) == 0:
            continue
            
        # 计算该setup的真实均值
        true_setup_mean = y_setup.mean().item()
        
        # 生成预测样本并计算均值
        generated_samples = []
        with torch.no_grad():
            for start in range(0, len(x_setup), batch_size):
                # cx = x_setup[start : start + batch_size].to(eval_device)
                # batch_samples = model.sample(samples_per_cond, context=cx).cpu()
                cx = x_setup[start : start + batch_size].to(eval_device)
                # same GPU‐sampling hack as above
                B = cx.size(0)
                z = torch.randn(samples_per_cond, B, 1, device=eval_device)
                z_flat = z.view(-1, 1)
                ctx_rep = cx.unsqueeze(0).expand(samples_per_cond, -1, -1) \
                             .reshape(-1, cx.size(-1))
                x_flat, _ = model._transform.inverse(z_flat, context=ctx_rep)
                batch_samples = x_flat.view(samples_per_cond, B, -1).cpu()                
                
                generated_samples.append(batch_samples)
        
        if generated_samples:
            # first flatten each [samples_per_cond, B, 1] → [samples_per_cond * B]
            flattened = [b.reshape(-1) for b in generated_samples]
            # now concatenate end-to-end
            all_generated = torch.cat(flattened, dim=0)
            pred_setup_mean = all_generated.mean().item()
            
            # 计算相对误差
            setup_relative_error = abs(pred_setup_mean - true_setup_mean) / (true_setup_mean + 1e-8)
            
            all_setup_errors.append(setup_relative_error)
            setup_error_details.append({
                'setup_id': setup_idx,
                'setup_params': test_unique_setups[setup_idx].tolist(),
                'true_mean': true_setup_mean,
                'pred_mean': pred_setup_mean,
                'relative_error': setup_relative_error,
                'n_samples': len(x_setup)
            })
    
    # 计算平均误差
    if all_setup_errors:
        mean_error = np.mean(all_setup_errors)
        std_error = np.std(all_setup_errors)
        
        print(f"\n🎯 Overall Results:")
        print(f"Number of test setups: {len(all_setup_errors)}")
        print(f"Average relative error across all setups: {mean_error:.4f} ± {std_error:.4f}")
        
        # 保存详细误差数据
        error_df = pd.DataFrame(setup_error_details)
        error_df.to_csv(error_csv_path, index=False)
        
        print(f"Detailed error data saved to: {error_csv_path}")
        
        return torch.tensor(all_setup_errors)
    else:
        print("❌ No valid setup errors computed!")
        return torch.tensor([])

# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_csv", type=str, default="data/x_data.csv", help="Path to x CSV file")
    parser.add_argument("--y_csv", type=str, default="data/y_data.csv", help="Path to y CSV file")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--hidden_features", type=int, default=16, help="Hidden features in each layer")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of flow layers")
    parser.add_argument("--num_bins", type=int, default=15, help="Number of bins for spline transforms")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Fraction of data held out for testing")
    parser.add_argument("--samples_per_cond", type=int, default=100, help="Samples per test condition during evaluation")
    parser.add_argument("--eval_subset", type=int, default=10000, help="Random subset of test rows to evaluate (None = all)")
    args = parser.parse_args()

    # ——— Load + split ———
    x, y = load_dataset(args.x_csv, args.y_csv)
    # set y all positive and set all elements < 0 to 0
    y = torch.clamp(y, min=0.0)
    print(f"Dataset loaded – {len(x)} rows, {x.shape[1]} features ➜ target dim 1")
    print(f"Y data range: min={y.min():.4f}, max={y.max():.4f}")
    print(f"Y data statistics: mean={y.mean():.4f}, std={y.std():.4f}")
    
    # 按setup分组划分（假设前6列是mf, mf1, mf2, gamma1, gamma2, T_coh）
    setup_cols = x[:, :6]  # 提取setup列
    unique_setups, indices = torch.unique(setup_cols, dim=0, return_inverse=True)
    n_setups = len(unique_setups)

    # 按setup划分训练测试集
    torch.manual_seed(42)
    setup_perm = torch.randperm(n_setups)
    n_test_setups = int(args.test_ratio * n_setups)
    test_setup_indices = setup_perm[:n_test_setups]
    train_setup_indices = setup_perm[n_test_setups:]

    # 创建训练测试mask
    test_mask = torch.isin(indices, test_setup_indices)
    train_mask = ~test_mask

    # 分割数据
    x_train, y_train = x[train_mask], y[train_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)

    print(f"Split by setup → train: {len(train_ds)} samples ({len(train_setup_indices)} setups) | test: {len(test_ds)} samples ({len(test_setup_indices)} setups)")
    
    # ——— DataLoaders ———
    batch_size = min(1024, len(train_ds))
    test_batch_size = min(512, len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)

    # ——— Model + Save metadata ———
    flow, model_config = build_nfs_model(
        context_features=x.shape[1],
        hidden_features=args.hidden_features,
        num_layers=args.num_layers,
        num_bins=args.num_bins
    )
    
    # Prepare metadata
    training_config = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": batch_size,
        "test_batch_size": test_batch_size,
        "optimizer": "Adam"
    }
    
    data_config = {
        "x_csv_path": args.x_csv,
        "y_csv_path": args.y_csv,
        "total_samples": len(x),
        "input_features": x.shape[1],
        "output_features": y.shape[1],
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "train_setups": len(train_setup_indices),
        "test_setups": len(test_setup_indices),
        "test_ratio": args.test_ratio,
        "y_data_stats": {
            "min": float(y.min()),
            "max": float(y.max()),
            "mean": float(y.mean()),
            "std": float(y.std())
        }
    }
    
    # Save metadata
    save_metadata(log_dir, model_config, training_config, data_config)
    
    # ——— Training ———
    training_results = train(
        flow, 
        train_loader, 
        test_loader,
        epochs=args.epochs, 
        lr=args.lr
    )

    # ——— Reload best weights & evaluate on all test setups ———
    flow.load_state_dict(torch.load(os.path.join(log_dir, "trained_flow.pt"), map_location=device))

    # 直接对整个测试集进行评估
    print(f"\n🔍 Evaluating all test setups...")
    
    all_errors = evaluate(
        flow,
        x_test,
        y_test,
        samples_per_cond=args.samples_per_cond,
        eval_subset=None if args.eval_subset <= 0 else args.eval_subset,
        save_path=f"{log_dir}/evaluation_overview.pdf",
        error_csv_path=f"{log_dir}/all_setup_errors.csv"
    )
    
    # 保存最终统计
    if len(all_errors) > 0:
        overall_df = pd.DataFrame({
            'setup_relative_errors': all_errors.numpy()
        })
        overall_df.to_csv(f"{log_dir}/overall_error_summary.csv", index=False)
        print(f"Final summary saved to: {log_dir}/overall_error_summary.csv")
    
    # Add training results to final summary
    final_summary = {
        "training_completed": True,
        "log_directory": log_dir,
        "final_train_loss": training_results['final_train_loss'],
        "final_test_loss": training_results['final_test_loss'],
        "mean_setup_error": float(all_errors.mean()) if len(all_errors) > 0 else None,
        "std_setup_error": float(all_errors.std()) if len(all_errors) > 0 else None
    }
    
    with open(os.path.join(log_dir, "final_summary.json"), 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    print(f"\n🎉 All results saved to: {log_dir}")


if __name__ == "__main__":
    main()