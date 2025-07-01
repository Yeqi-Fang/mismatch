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
import torch.nn as nn

from nflows.distributions.base import Distribution
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform
)
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.coupling import (
    PiecewiseRationalQuadraticCouplingTransform,
    AffineCouplingTransform
)
from nflows.transforms.normalization import BatchNorm
from torch.distributions import Beta
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


class BetaDistribution(Distribution):
    """Beta base distribution for [0,1] support"""
    
    def __init__(self, shape, alpha=1.0, beta=1.0):
        super().__init__()
        self._shape = torch.Size(shape)
        self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('beta', torch.tensor(beta))
        self._beta_dist = Beta(self.alpha, self.beta)
    
    def _log_prob(self, inputs, context):
        return self._beta_dist.log_prob(inputs).sum(-1)
    
    def _sample(self, num_samples, context):
        return self._beta_dist.sample((num_samples,) + self._shape)



# Fix 1: More aggressive gradient clipping and better training
def train_flow_stable_fixed(model, train_loader, test_loader=None, epochs=200, lr=5e-3):
    """Training with much more aggressive stability measures"""
    model.to(device)
    
    # Ensure all components are on device
    for param in model.parameters():
        param.data = param.data.to(device)
    for buffer in model.buffers():
        buffer.data = buffer.data.to(device)
    
    # Lower learning rate and more aggressive weight decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.7, min_lr=1e-6)
    
    train_losses = []
    test_losses = []
    
    print(f"🚀 Starting stable training for {epochs} epochs...")
    
    for epoch in range(epochs + 1):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # More aggressive clamping
            batch_y = torch.clamp(batch_y, 1e-5, 1-1e-5)
            
            optimizer.zero_grad()
            
            try:
                log_prob = model.log_prob(inputs=batch_y, context=batch_x)
                loss = -log_prob.mean()
                
                # Check for problematic losses
                if torch.isnan(loss) or torch.isinf(loss) or loss > 1000:
                    print(f"⚠️ Problematic loss {loss:.2f} at epoch {epoch}, batch {batch_idx}, skipping")
                    continue
                
                loss.backward()
                
                # Much more aggressive gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                # Check for gradient explosion
                if grad_norm > 10.0:
                    print(f"⚠️ Large gradient norm {grad_norm:.2f} at epoch {epoch}, skipping step")
                    continue
                
                optimizer.step()
                epoch_loss += loss.item()
                valid_batches += 1
                
            except RuntimeError as e:
                print(f"Runtime error at epoch {epoch}, batch {batch_idx}: {e}")
                continue
        
        if valid_batches > 0:
            epoch_loss /= valid_batches
        else:
            print(f"⚠️ No valid batches in epoch {epoch}")
            epoch_loss = float('inf')
        
        train_losses.append(epoch_loss)
        
        # Validation with early stopping check
        if test_loader:
            model.eval()
            test_loss = 0.0
            test_valid_batches = 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    batch_y = torch.clamp(batch_y, 1e-5, 1-1e-5)
                    
                    try:
                        log_prob = model.log_prob(inputs=batch_y, context=batch_x)
                        batch_test_loss = -log_prob.mean().item()
                        
                        # Skip problematic test losses
                        if not (torch.isnan(torch.tensor(batch_test_loss)) or torch.isinf(torch.tensor(batch_test_loss))):
                            test_loss += batch_test_loss
                            test_valid_batches += 1
                    except Exception as e:
                        continue
            
            if test_valid_batches > 0:
                test_loss /= test_valid_batches
                test_losses.append(test_loss)
                scheduler.step(test_loss)
            else:
                test_losses.append(float('inf'))
        
        # Early stopping if training becomes unstable
        if epoch_loss > 1000:
            print(f"🛑 Training unstable (loss > 1000), stopping early at epoch {epoch}")
            break
        
        # Logging with stability warnings
        if epoch % 10 == 0 or epoch == 1:
            log_msg = f"Epoch {epoch:3d}/{epochs} | train nll {epoch_loss:.4f}"
            if test_loader and test_losses:
                log_msg += f" | test nll {test_losses[-1]:.4f}"
            print(log_msg)
            
            # Warn about instability
            if epoch_loss > 100:
                print(f"⚠️ Training loss is high ({epoch_loss:.2f}) - potential instability")
    
    # Save model
    torch.save(model.state_dict(), os.path.join(log_dir, "trained_flow.pt"))
    print(f"✔️  Training complete – model saved to {log_dir}/trained_flow.pt")
    
    # Create training plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
    if test_losses:
        plt.plot(epochs_range, test_losses[:len(train_losses)], 'r-', label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log scale plot to see the explosion better
    plt.subplot(1, 2, 2)
    plt.semilogy(epochs_range, train_losses, 'b-', label='Training Loss (log scale)', linewidth=2)
    if test_losses:
        # Only plot positive test losses on log scale
        positive_test_losses = [max(1e-10, loss) for loss in test_losses[:len(train_losses)]]
        plt.semilogy(epochs_range, positive_test_losses, 'r-', label='Test Loss (log scale)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log-Likelihood (log scale)')
    plt.title('Training Curves (Log Scale)')
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
        'test_loss': test_losses[:len(train_losses)] if test_losses else [None] * len(train_losses)
    }
    loss_df = pd.DataFrame(loss_data)
    loss_csv_path = os.path.join(log_dir, "training_losses.csv")
    loss_df.to_csv(loss_csv_path, index=False)
    print(f"✔️  Loss data saved to {loss_csv_path}")
    
    return train_losses, test_losses

# Fix 2: More conservative model architecture
def build_conservative_flow(context_features: int, hidden_features: int = 32, 
                           num_layers: int = 4, num_bins: int = 8):
    """More conservative flow architecture for stability"""
    transforms = []
    
    for i in range(num_layers):
        if i % 2 == 0:
            # Smaller tail bounds and fewer bins for stability
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=1,
                    hidden_features=hidden_features,
                    context_features=context_features,
                    num_bins=num_bins,
                    tails="linear",
                    tail_bound=2.0,  # Reduced from 4.0
                )
            )
        else:
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=1,
                    hidden_features=hidden_features,
                    context_features=context_features,
                )
            )
    
    base_dist = StandardNormal([1])
    flow = Flow(CompositeTransform(transforms), base_dist)
    
    # Move to device
    if torch.cuda.is_available():
        flow = flow.to(device)
        for param in flow.parameters():
            param.data = param.data.to(device)
        for buffer in flow.buffers():
            buffer.data = buffer.data.to(device)
    
    return flow

# Update build_nfs_model for stability
def build_nfs_model_stable(context_features: int, flow_features: int = 1, 
                          hidden_features: int = 16, num_layers: int = 3, 
                          num_bins: int = 15) -> Tuple[Flow, dict]:
    """Build more stable flow model"""
    
    # Use more conservative architecture
    flow = build_conservative_flow(
        context_features=context_features,
        hidden_features=hidden_features * 2,  # Reduced from 4x
        num_layers=num_layers,  # Don't double the layers
        num_bins=max(8, num_bins // 2)  # Fewer bins for stability
    )
    
    model_config = {
        "model_type": "ConservativeAutoregressiveFlow",
        "context_features": context_features,
        "hidden_features": hidden_features * 2,
        "num_layers": num_layers,
        "num_bins": max(8, num_bins // 2),
        "base_distribution": "StandardNormal",
        "stability_measures": "conservative_architecture_aggressive_clipping",
        "total_parameters": sum(p.numel() for p in flow.parameters())
    }
    
    return flow, model_config



def build_autoregressive_flow_for_01(context_features: int, hidden_features: int = 64,
                                   num_layers: int = 8, num_bins: int = 32):
    """
    Improved autoregressive flow for [0,1] data with Normal base distribution
    """
    transforms = []
    
    for i in range(num_layers):
        if i % 3 == 0:
            # Rational quadratic spline
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=1,
                    hidden_features=hidden_features,
                    context_features=context_features,
                    num_bins=num_bins,
                    tails="linear",
                    tail_bound=4.0,
                )
            )
        elif i % 3 == 1:
            # Affine transform
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=1,
                    hidden_features=hidden_features,
                    context_features=context_features,
                )
            )
        else:
            # Add random permutation for better mixing
            transforms.append(RandomPermutation(features=1))
    
    # Use Standard Normal base distribution (safer for flows)
    base_dist = StandardNormal([1])
    flow = Flow(CompositeTransform(transforms), base_dist)
    
    return flow


# Better option: Use logit transformation for proper [0,1] handling
def build_logit_flow_for_01(context_features: int, hidden_features: int = 64, 
                           num_layers: int = 8, num_bins: int = 32):
    """
    Flow with logit transformation - maps [0,1] to (-∞,∞) properly
    This is the RECOMMENDED approach for [0,1] bounded data
    """
    from nflows.transforms.nonlinearities import Sigmoid
    
    transforms = []
    
    # First apply logit transformation to map [0,1] to (-∞,∞)
    # This handles the boundary constraints properly
    transforms.append(Sigmoid())
    
    # Then apply normalizing flows in unbounded space
    for i in range(num_layers):
        if i % 2 == 0:
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=1,
                    hidden_features=hidden_features,
                    context_features=context_features,
                    num_bins=num_bins,
                    tails="linear",
                    tail_bound=5.0,
                )
            )
        else:
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=1,
                    hidden_features=hidden_features,
                    context_features=context_features,
                )
            )
    
    # Use standard normal base distribution
    base_dist = StandardNormal([1])
    flow = Flow(CompositeTransform(transforms), base_dist)
    
    return flow

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
                cx = x_setup[start : start + batch_size].to(device)
                y = y_setup[start : start + batch_size]
                
                try:
                    batch_samples = model.sample(samples_per_cond, context=cx).cpu()
                    batch_log_probs = model.log_prob(inputs=y.to(device), context=cx).cpu()
                    
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
                cx = x_setup[start : start + batch_size].to(device)
                batch_samples = model.sample(samples_per_cond, context=cx).cpu()
                generated_samples.append(batch_samples)
        
        if generated_samples:
            all_generated = torch.cat(generated_samples)
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
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
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
    batch_size = args.batch_size
    test_batch_size = batch_size

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)

    # ——— Model + Save metadata ———
    flow, model_config = build_nfs_model_stable(
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
    training_results = train_flow_stable_fixed(
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
        "final_train_loss": training_results[0][-1] if training_results[0] else None,  # Last training loss
        "final_test_loss": training_results[1][-1] if training_results[1] else None,   # Last test loss
        "mean_setup_error": float(all_errors.mean()) if len(all_errors) > 0 else None,
        "std_setup_error": float(all_errors.std()) if len(all_errors) > 0 else None
    }
    
    with open(os.path.join(log_dir, "final_summary.json"), 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    print(f"\n🎉 All results saved to: {log_dir}")


if __name__ == "__main__":
    main()