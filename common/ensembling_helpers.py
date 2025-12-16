import os
import re

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def plot_ensemble_score_summary_to_png(
    scores,
    title,
    out_dir,
    percentiles=(10, 30, 70, 90),
    figsize=(8, 5),
    pad_frac=0.05,
    dpi=200,
):
    """
    Make a summary plot (mean + given percentiles) of ensemble scores and
    save it as PNG:  <out_dir>/<title>.png

    Parameters
    ----------
    scores : sequence of sequences
        scores[i] – iterable of scores for ensemble of (i+1) models.
    title : str
        Plot title; also used as the base of the filename.
    out_dir : str
        Directory where the PNG will be saved.
    percentiles : iterable of float, optional
        Percentiles in [0, 100] to plot, e.g. (10, 30, 70, 90).
    figsize : tuple, optional
        Matplotlib figure size.
    pad_frac : float, optional
        Extra padding around the min/max of all curves for y-limits.
    dpi : int, optional
        DPI for the saved PNG.

    Returns
    -------
    path : str
        Full path to the saved PNG file.
    """
    if not title:
        raise ValueError("title must be a non-empty string.")

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Sanitize title for filename
    safe_title = re.sub(r"[^\w\-]+", "_", title).strip("_")
    filename = f"{safe_title}.png"
    path = os.path.join(out_dir, filename)

    scores = [np.asarray(s, dtype=float) for s in scores]
    n_models = len(scores)
    if n_models == 0:
        raise ValueError("scores must contain at least one list.")

    x = np.arange(1, n_models + 1)

    # Mean for each ensemble size
    means = np.array([s.mean() if s.size else np.nan for s in scores])

    # Percentiles for each ensemble size
    percentiles = list(percentiles)
    perc_vals = {
        p: np.array([np.percentile(s, p) if s.size else np.nan for s in scores])
        for p in percentiles
    }

    fig, ax = plt.subplots(figsize=figsize)

    # Mean line
    ax.plot(x, means, marker="o", linewidth=2, label="mean")

    # Percentile lines
    for p in sorted(percentiles):
        ax.plot(x, perc_vals[p], marker="o", linestyle="--", label=f"p{p}")

    # --- always zoom to data (use all curves) ---
    all_series = [means] + list(perc_vals.values())
    all_vals = np.concatenate([v[~np.isnan(v)] for v in all_series])
    y_min, y_max = float(all_vals.min()), float(all_vals.max())
    if y_max == y_min:
        pad = abs(y_max) * 0.01 if y_max != 0 else 1e-3
    else:
        pad = (y_max - y_min) * pad_frac
    ax.set_ylim(y_min - pad, y_max + pad)

    # X axis & labels
    ax.set_xlim(0.5, n_models + 0.5)
    ax.set_xticks(x)
    ax.set_xlabel("Number of models in ensemble")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    # Save to PNG and close figure
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return path
