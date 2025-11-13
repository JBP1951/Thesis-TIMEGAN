"""
evaluation_TGAN.py
------------------
Quantitative evaluation metrics for TimeGAN synthetic vs real time-series data.

Includes:
- MSE, MAE, MSLE, MAPE
- R² (coefficient of determination)
- Pearson correlation (optional)
- Distributional KL-divergence check (optional)
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_log_error,
    r2_score
)
from scipy.stats import entropy, pearsonr


def flatten_sequences(real_seqs, synth_seqs, max_samples=None):
    """
    Convert lists of sequences (3D) into comparable 2D arrays for metric computation.
    
    Args:
        real_seqs, synth_seqs: list of np.ndarray, shape [seq_len, features]
        max_samples: limit number of sequences for fair comparison
    """
    n = min(len(real_seqs), len(synth_seqs))
    if max_samples is not None:
        n = min(n, max_samples)
    
    real_flat = np.concatenate(real_seqs[:n]).reshape(-1, real_seqs[0].shape[1])
    synth_flat = np.concatenate(synth_seqs[:n]).reshape(-1, synth_seqs[0].shape[1])
    
    # make sure lengths match
    min_len = min(len(real_flat), len(synth_flat))
    return real_flat[:min_len], synth_flat[:min_len]


def evaluate_timegan(real_seqs, synth_seqs, max_samples=1000, verbose=True):
    """
    Compute quantitative similarity metrics between real and synthetic sequences.

    Args:
        real_seqs: list of np.ndarray (real data)
        synth_seqs: list of np.ndarray (synthetic data)
        max_samples: limit number of sequences for comparison
        verbose: print results

    Returns:
        dict of metric_name -> value
    """
    real_flat, synth_flat = flatten_sequences(real_seqs, synth_seqs, max_samples)

    # basic metrics
    mse = mean_squared_error(real_flat, synth_flat)
    mae = mean_absolute_error(real_flat, synth_flat)
    mape = mean_absolute_percentage_error(real_flat + 1e-8, synth_flat + 1e-8)
        # Safe MSLE (skip if negative values)
    if np.any(real_flat <= -1) or np.any(synth_flat <= -1):
        msle = np.nan
    else:
        msle = mean_squared_log_error(real_flat + 1e-8, np.abs(synth_flat) + 1e-8)

    r2 = r2_score(real_flat, synth_flat)

    # optional correlation check
    corr_vals = []
    for i in range(real_flat.shape[1]):
        corr, _ = pearsonr(real_flat[:, i], synth_flat[:, i])
        corr_vals.append(corr)
    mean_corr = np.nanmean(corr_vals)

    # optional KL-divergence of distributions
    p_hist, _ = np.histogram(real_flat, bins=50, density=True)
    q_hist, _ = np.histogram(synth_flat, bins=50, density=True)
    kl_div = entropy(p_hist + 1e-8, q_hist + 1e-8)

    results = {
        "MSE": mse,
        "MAE": mae,
        "MAPE": mape,
        "MSLE": msle,
        "R2": r2,
        "Mean_Corr": mean_corr,
        "KL_Divergence": kl_div
    }

    if verbose:
        print("\n📊 [TimeGAN Evaluation Metrics]")
        for k, v in results.items():
            print(f"{k:>15s}: {v:.6f}")
        print("-----------------------------------")

    return results
