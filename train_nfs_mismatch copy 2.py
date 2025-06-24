# train_nfs_refactored.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from datetime import datetime

# Create main log directory
main_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
main_log_dir = f"logs/grid_search_{main_timestamp}"
os.makedirs(main_log_dir, exist_ok=True)

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform
)
from nflows.transforms.permutations import RandomPermutation

# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

# Support for Apple Silicon MPS
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

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


def build_nfs_model(context_features: int, flow_features: int = 1, 
                   hidden_features: int = 64, num_layers: int = 6, 
                   num_bins: int = 20) -> Tuple[Flow, dict]:
    """Factory: a shallow MAF‑style conditional normalising flow."""
    transforms: List = []
    base_dist = StandardNormal([flow_features])

    for _ in range(num_layers):
        transforms.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=flow_features,
                hidden_features=hidden_features, 
                context_features=context_features,
                num_bins=num_bins,  
                min_bin_width=1e-3,
                min_bin_height=1e-3,
                min_derivative=1e-3,
                tails="linear",        # ← extend outside the bound
                tail_bound=4.0,        # ← big enough for ~99.7 % of N(0,1)
            )
        )
    flow =  Flow(
            CompositeTransform(transforms), 
            base_dist
        )
    
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
    
    return flow.float(), model_config

def train_single_config(model: Flow, train_loader: DataLoader, test_loader: DataLoader, 
                       log_dir: str, *, epochs: int = 200, lr: float = 3e-2) -> dict:
    """Single‑loop optimiser with test loss tracking and training plots."""
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    
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

        # Logging (less frequent to avoid spam during grid search)
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            log_msg = f"Epoch {epoch:3d}/{epochs} | train nll {ep_train_loss:.4f}"
            if ep_test_loss is not None:
                log_msg += f" | test nll {ep_test_loss:.4f}"
            print(log_msg)

    # Save model
    torch.save(model.state_dict(), os.path.join(log_dir, "trained_flow.pt"))
    
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
    plt.close()  # Close figure to save memory during grid search
    
    # Save loss data
    loss_data = {
        'epoch': list(epochs_range),
        'train_loss': train_losses,
        'test_loss': test_losses if test_losses else [None] * epochs
    }
    loss_df = pd.DataFrame(loss_data)
    loss_csv_path = os.path.join(log_dir, "training_losses.csv")
    loss_df.to_csv(loss_csv_path, index=False)
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1] if test_losses else None
    }


def evaluate_single_config(
    model: Flow,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    log_dir: str,
    *,
    samples_per_cond: int = 100,
    batch_size: int = 512,
) -> Dict[str, float]:
    """Simplified evaluation that returns key metrics for grid search."""
    model.eval()

    # Get setup information
    test_setup_cols = x_test[:, :3]  # 前3列是setup特征
    test_unique_setups, test_indices = torch.unique(test_setup_cols, dim=0, return_inverse=True)
    
    print(f"📊 Computing errors for all {len(test_unique_setups)} test setups...")
    
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
                try:
                    batch_samples = model.sample(samples_per_cond, context=cx).cpu()
                    generated_samples.append(batch_samples)
                except Exception as e:
                    print(f"⚠️ Error in sampling: {e}")
                    continue
        
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
    
    # 计算并保存误差统计
    if all_setup_errors:
        mean_error = np.mean(all_setup_errors)
        std_error = np.std(all_setup_errors)
        median_error = np.median(all_setup_errors)
        
        # 保存详细误差数据
        error_df = pd.DataFrame(setup_error_details)
        error_csv_path = os.path.join(log_dir, "all_setup_errors.csv")
        error_df.to_csv(error_csv_path, index=False)
        
        return {
            'mean_error': mean_error,
            'std_error': std_error,
            'median_error': median_error,
            'num_setups': len(all_setup_errors)
        }
    else:
        return {
            'mean_error': float('inf'),
            'std_error': float('inf'),
            'median_error': float('inf'),
            'num_setups': 0
        }


def run_grid_search(
    param_grid: Dict[str, List],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    context_features: int,
    epochs: int = 100,
    batch_size: int = 1024,
    test_batch_size: int = 512
) -> Tuple[Dict, List[Dict]]:
    """Run grid search over hyperparameters."""
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(itertools.product(*param_values))
    
    print(f"🔍 Starting grid search with {len(all_combinations)} combinations...")
    print(f"Parameters to tune: {param_names}")
    
    # Prepare data loaders
    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)
    
    results = []
    best_config = None
    best_score = float('inf')
    
    for i, combination in enumerate(all_combinations):
        # Create parameter dictionary
        params = dict(zip(param_names, combination))
        
        print(f"\n{'='*60}")
        print(f"🔧 Configuration {i+1}/{len(all_combinations)}: {params}")
        print(f"{'='*60}")
        
        # Create experiment directory
        exp_name = "_".join([f"{k}_{v}" for k, v in params.items()])
        exp_dir = os.path.join(main_log_dir, f"exp_{i+1:03d}_{exp_name}")
        os.makedirs(exp_dir, exist_ok=True)
        
        try:
            # Build model with current parameters
            flow, model_config = build_nfs_model(
                context_features=context_features,
                hidden_features=params['hidden_features'],
                num_layers=params['num_layers'],
                num_bins=params['num_bins']
            )
            
            # Train model
            training_results = train_single_config(
                flow, train_loader, test_loader, exp_dir,
                epochs=epochs, lr=params['lr']
            )
            
            # Evaluate model
            eval_results = evaluate_single_config(
                flow, x_test, y_test, exp_dir
            )
            
            # Combine results
            experiment_result = {
                'experiment_id': i + 1,
                'experiment_dir': exp_dir,
                'parameters': params,
                'model_config': model_config,
                'training_results': training_results,
                'evaluation_results': eval_results,
                'primary_score': eval_results['mean_error']  # Lower is better
            }
            
            results.append(experiment_result)
            
            # Check if this is the best configuration
            if eval_results['mean_error'] < best_score:
                best_score = eval_results['mean_error']
                best_config = experiment_result
            
            print(f"✅ Completed - Mean Error: {eval_results['mean_error']:.4f}")
            
            # Save individual experiment results
            with open(os.path.join(exp_dir, "experiment_results.json"), 'w') as f:
                json.dump(experiment_result, f, indent=2, default=str)
                
        except Exception as e:
            print(f"❌ Configuration failed: {e}")
            error_result = {
                'experiment_id': i + 1,
                'experiment_dir': exp_dir,
                'parameters': params,
                'error': str(e),
                'primary_score': float('inf')
            }
            results.append(error_result)
    
    return best_config, results


def save_grid_search_summary(best_config: Dict, all_results: List[Dict], summary_dir: str):
    """Save comprehensive grid search summary."""
    
    # Create summary DataFrame
    summary_data = []
    for result in all_results:
        row = result['parameters'].copy()
        if 'evaluation_results' in result:
            row.update({
                'mean_error': result['evaluation_results']['mean_error'],
                'std_error': result['evaluation_results']['std_error'],
                'median_error': result['evaluation_results']['median_error'],
                'final_train_loss': result['training_results']['final_train_loss'],
                'final_test_loss': result['training_results']['final_test_loss'],
                'total_parameters': result['model_config']['total_parameters']
            })
        else:
            row.update({
                'mean_error': float('inf'),
                'std_error': float('inf'),
                'median_error': float('inf'),
                'final_train_loss': float('inf'),
                'final_test_loss': float('inf'),
                'total_parameters': 0
            })
        row['experiment_id'] = result['experiment_id']
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('mean_error')
    
    # Save summary CSV
    summary_csv_path = os.path.join(summary_dir, "grid_search_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"📊 Grid search summary saved to: {summary_csv_path}")
    
    # Save best configuration
    best_config_path = os.path.join(summary_dir, "best_configuration.json")
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2, default=str)
    print(f"🏆 Best configuration saved to: {best_config_path}")
    
    # Create visualization
    create_grid_search_visualization(summary_df, summary_dir)
    
    return summary_df


def create_grid_search_visualization(summary_df: pd.DataFrame, summary_dir: str):
    """Create visualizations for grid search results."""
    
    # Plot 1: Parameter vs Performance
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    params = ['lr', 'hidden_features', 'num_layers', 'num_bins']
    
    for i, param in enumerate(params):
        ax = axes[i//2, i%2]
        
        # Group by parameter and compute statistics
        grouped = summary_df.groupby(param)['mean_error'].agg(['mean', 'std', 'min']).reset_index()
        
        ax.errorbar(grouped[param], grouped['mean'], yerr=grouped['std'], 
                   marker='o', capsize=5, capthick=2)
        ax.scatter(grouped[param], grouped['min'], color='red', marker='x', s=50, label='Best')
        ax.set_xlabel(param)
        ax.set_ylabel('Mean Relative Error')
        ax.set_title(f'Performance vs {param}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(summary_dir, "grid_search_analysis.pdf")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot 2: Top 10 configurations
    plt.figure(figsize=(12, 8))
    top_10 = summary_df.head(10)
    
    bars = plt.bar(range(len(top_10)), top_10['mean_error'])
    plt.xlabel('Configuration Rank')
    plt.ylabel('Mean Relative Error')
    plt.title('Top 10 Configurations by Performance')
    plt.xticks(range(len(top_10)), [f"Config {i+1}" for i in range(len(top_10))])
    
    # Add parameter labels
    for i, (idx, row) in enumerate(top_10.iterrows()):
        plt.text(i, row['mean_error'] + 0.001, 
                f"lr:{row['lr']}\nh:{row['hidden_features']}\nl:{row['num_layers']}\nb:{row['num_bins']}", 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    top10_path = os.path.join(summary_dir, "top_10_configurations.pdf")
    plt.savefig(top10_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"📈 Visualizations saved to: {summary_dir}")


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_csv", type=str, default="data/x_data.csv", help="Path to x CSV file")
    parser.add_argument("--y_csv", type=str, default="data/y_data.csv", help="Path to y CSV file")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Fraction of data held out for testing")
    parser.add_argument("--samples_per_cond", type=int, default=100, help="Samples per test condition during evaluation")
    parser.add_argument("--grid_search", action="store_true", help="Run grid search for hyperparameter tuning")
    
    # Grid search specific arguments
    parser.add_argument("--lr_values", nargs="+", type=float, default=[1e-3, 3e-3, 1e-2, 3e-2], 
                       help="Learning rate values for grid search")
    parser.add_argument("--hidden_features_values", nargs="+", type=int, default=[32, 64, 128], 
                       help="Hidden features values for grid search")
    parser.add_argument("--num_layers_values", nargs="+", type=int, default=[3, 6, 9], 
                       help="Number of layers values for grid search")
    parser.add_argument("--num_bins_values", nargs="+", type=int, default=[10, 20, 30], 
                       help="Number of bins values for grid search")
    
    args = parser.parse_args()

    # ——— Load + split ———
    x, y = load_dataset(args.x_csv, args.y_csv)
    y = torch.clamp(y, min=0.0)
    print(f"Dataset loaded – {len(x)} rows, {x.shape[1]} features ➜ target dim 1")
    print(f"Y data range: min={y.min():.4f}, max={y.max():.4f}")
    print(f"Y data statistics: mean={y.mean():.4f}, std={y.std():.4f}")
    
    # Split by setup
    setup_cols = x[:, :3]
    unique_setups, indices = torch.unique(setup_cols, dim=0, return_inverse=True)
    n_setups = len(unique_setups)

    torch.manual_seed(42)
    setup_perm = torch.randperm(n_setups)
    n_test_setups = int(args.test_ratio * n_setups)
    test_setup_indices = setup_perm[:n_test_setups]
    train_setup_indices = setup_perm[n_test_setups:]

    test_mask = torch.isin(indices, test_setup_indices)
    train_mask = ~test_mask

    x_train, y_train = x[train_mask], y[train_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    print(f"Split by setup → train: {len(x_train)} samples ({len(train_setup_indices)} setups) | test: {len(x_test)} samples ({len(test_setup_indices)} setups)")
    
    if args.grid_search:
        # Define parameter grid
        param_grid = {
            'lr': args.lr_values,
            'hidden_features': args.hidden_features_values,
            'num_layers': args.num_layers_values,
            'num_bins': args.num_bins_values
        }
        
        print(f"🔍 Grid search configuration:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        # Run grid search
        best_config, all_results = run_grid_search(
            param_grid=param_grid,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            context_features=x.shape[1],
            epochs=args.epochs
        )
        
        # Save comprehensive summary
        summary_df = save_grid_search_summary(best_config, all_results, main_log_dir)
        
        # Print results
        print(f"\n{'='*80}")
        print(f"🏆 GRID SEARCH COMPLETED")
        print(f"{'='*80}")
        print(f"Best configuration:")
        if best_config:
            for param, value in best_config['parameters'].items():
                print(f"  {param}: {value}")
            print(f"Best mean error: {best_config['evaluation_results']['mean_error']:.4f}")
        
        print(f"\nTop 5 configurations:")
        for i, (idx, row) in enumerate(summary_df.head(5).iterrows()):
            print(f"  {i+1}. lr={row['lr']}, h={row['hidden_features']}, l={row['num_layers']}, b={row['num_bins']} -> Error: {row['mean_error']:.4f}")
        
        print(f"\n📁 All results saved to: {main_log_dir}")
        
    else:
        # Run single configuration with updated defaults
        print("Running single configuration with default parameters...")
        # This would run the original single training loop
        # For now, suggest using grid search mode
        print("💡 Tip: Use --grid_search flag for hyperparameter tuning!")


if __name__ == "__main__":
    main()