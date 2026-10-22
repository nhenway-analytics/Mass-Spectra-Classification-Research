from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------
# Poster visual theme
# ---------------------------------------------------------------
BG = "white"
TEXT = "#1f1f1f"
MUTED = "#5f6368"
GRID = "#d9dde3"
ACCENT = "#145a44"
ACCENT_LIGHT = "#e8f2ee"
COSINE = "#f28e2b"
DELTA = "#4e79a7"
DELTA_PRIME = "#2ca02c"
RANDOM_LINE = "#d62728"
SAME_FILL = "#2f7d62"
DIFF_FILL = "#9aa3ad"

# Stronger contrast for spectral overlay replicates
REP1_COLOR = "#8c564b"   # brown
REP2_COLOR = "#17becf"   # cyan


METHOD_ORDER = ["cosine_best", "delta", "delta_prime"]
METHOD_LABELS = {
    "cosine_best": "Cosine Similarity",
    "delta": "δ",
    "delta_prime": "δ′",
}
METHOD_SHORT = {
    "cosine_best": "Cosine",
    "delta": "δ",
    "delta_prime": "δ′",
}
METHOD_COLORS = {
    "cosine_best": COSINE,
    "delta": DELTA,
    "delta_prime": DELTA_PRIME,
}
METHOD_HIGHER_IS_SAME = {
    "cosine_best": True,
    "delta": False,
    "delta_prime": False,
}


# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------

def newest_file(out_dir: Path, pattern: str) -> Path:
    out_dir = Path(out_dir)
    files = sorted(out_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No files found matching {pattern} in {out_dir}")
    return files[0]


def apply_clean_style(ax, add_grid=True, grid_axis="y"):
    ax.set_facecolor(BG)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=TEXT, labelsize=11)
    if add_grid:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)


def zoom_limits(series_a, series_b, threshold: float, higher_is_same: bool, pad=0.08):
    a = pd.Series(series_a).dropna().astype(float)
    b = pd.Series(series_b).dropna().astype(float)
    x = pd.concat([a, b])
    xmin = float(x.min())
    xmax = float(x.max())

    if higher_is_same:
        lo = min(threshold, float(np.quantile(a, 0.005)), float(np.quantile(b, 0.97)))
        hi = max(float(np.quantile(a, 0.999)), threshold)
        span = hi - lo
        lo = max(xmin, lo - max(0.01, pad * max(span, 0.02)))
        hi = min(xmax, hi + max(0.004, 0.25 * pad * max(span, 0.02)))
    else:
        lo = min(float(np.quantile(a, 0.001)), float(np.quantile(b, 0.001)), threshold)
        hi = max(threshold, float(np.quantile(a, 0.999)), float(np.quantile(b, 0.03)))
        span = hi - lo
        lo = max(xmin, lo - max(0.002, 0.30 * pad * max(span, 0.01)))
        hi = min(xmax, hi + max(0.01, 0.75 * pad * max(span, 0.01)))

    if hi <= lo:
        span = max(1e-4, 0.08 * (xmax - xmin))
        lo = max(xmin, threshold - span)
        hi = min(xmax, threshold + span)
    return lo, hi


def make_bins(same: pd.Series, diff: pd.Series, zoom=False, xlim=None, nbins_full=45, nbins_zoom=28):
    same = pd.Series(same).dropna().astype(float)
    diff = pd.Series(diff).dropna().astype(float)

    if zoom and xlim is not None:
        lo, hi = xlim
        zoom_same = same[(same >= lo) & (same <= hi)]
        zoom_diff = diff[(diff >= lo) & (diff <= hi)]
        zoom_data = pd.concat([zoom_same, zoom_diff])
        if len(zoom_data) >= 4 and float(zoom_data.max()) > float(zoom_data.min()):
            return np.linspace(float(zoom_data.min()), float(zoom_data.max()), nbins_zoom)

    combined = pd.concat([same, diff])
    return np.histogram_bin_edges(combined, bins=nbins_full)


def best_threshold_from_seed_summary(df_seed: pd.DataFrame, method: str) -> float:
    sub = df_seed[df_seed["method"] == method]
    if sub.empty:
        raise KeyError(f"Method '{method}' not found in summary_by_seed data.")
    return float(sub["threshold"].mean())


def score_xlabel(method: str) -> str:
    if method == "cosine_best":
        return "Cosine Similarity Score"
    if method == "delta":
        return "Min–Max Score (δ)"
    if method == "delta_prime":
        return "Robust Min–Max Score (δ′)"
    return method


def score_title(method: str) -> str:
    if method == "cosine_best":
        return "Cosine Similarity"
    if method == "delta":
        return "Min–Max Statistic (δ)"
    if method == "delta_prime":
        return "Robust Min–Max Statistic (δ′)"
    return method


def compute_roc(scores: np.ndarray, y: np.ndarray, higher_is_same: bool):
    smin, smax = float(np.min(scores)), float(np.max(scores))
    thresholds = np.array([smin]) if smin == smax else np.linspace(smin, smax, 500)

    tprs = []
    fprs = []
    for t in thresholds:
        preds = (scores >= t).astype(int) if higher_is_same else (scores <= t).astype(int)
        tp = int(((preds == 1) & (y == 1)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)

    fprs = np.array(fprs, dtype=float)
    tprs = np.array(tprs, dtype=float)
    order = np.argsort(fprs)
    fprs = fprs[order]
    tprs = tprs[order]
    auc = float(np.trapz(tprs, fprs))
    return fprs, tprs, auc


def _metric_value(df_stats: pd.DataFrame, method: str, column: str):
    sub = df_stats.loc[df_stats["method"] == method, column]
    if sub.empty:
        return np.nan
    return float(sub.iloc[0])


# ---------------------------------------------------------------
# Main figures
# ---------------------------------------------------------------

def make_histogram_plot(
    same: pd.Series,
    diff: pd.Series,
    method: str,
    threshold: float,
    out_path: Path,
    zoom: bool = False,
) -> None:
    same = same.dropna().astype(float)
    diff = diff.dropna().astype(float)

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=BG)
    apply_clean_style(ax)

    xlim = None
    suffix = " (Full Range)"

    if zoom:
        xlim = zoom_limits(
            same,
            diff,
            threshold=threshold,
            higher_is_same=METHOD_HIGHER_IS_SAME[method],
            pad=0.06,
        )
        suffix = " (Zoomed)"

    bins = make_bins(same, diff, zoom=zoom, xlim=xlim)

    ax.hist(diff, bins=bins, alpha=0.40, density=True, color=DIFF_FILL,
            histtype="stepfilled", edgecolor=DIFF_FILL, linewidth=1.0, label="Different compound")
    ax.hist(same, bins=bins, alpha=0.40, density=True, color=SAME_FILL,
            histtype="stepfilled", edgecolor=SAME_FILL, linewidth=1.0, label="Same compound")
    ax.hist(diff, bins=bins, density=True, histtype="step", linewidth=1.8, color=DIFF_FILL)
    ax.hist(same, bins=bins, density=True, histtype="step", linewidth=1.8, color=SAME_FILL)

    ax.axvline(
        threshold,
        linestyle="--",
        linewidth=2.2,
        color=TEXT,
        label=f"Decision threshold = {threshold:.4f}",
    )

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.set_xlabel(score_xlabel(method), fontsize=13, color=TEXT)
    ax.set_ylabel("Density", fontsize=13, color=TEXT)
    ax.set_title(f"Score Distribution: {score_title(method)}{suffix}", fontsize=16, color=TEXT, pad=10)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left", bbox_to_anchor=(0.02, 0.98))

    fig.tight_layout()
    fig.savefig(out_path, dpi=350, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def make_accuracy_plot(df_stats: pd.DataFrame, fig_dir: Path) -> None:
    plot_df = df_stats.copy()
    desired_order = [m for m in METHOD_ORDER if m in plot_df["method"].tolist()]
    remaining = [m for m in plot_df["method"].tolist() if m not in desired_order]
    plot_df["method"] = pd.Categorical(plot_df["method"], categories=desired_order + remaining, ordered=True)
    plot_df = plot_df.sort_values("method").reset_index(drop=True)

    methods = plot_df["method"].tolist()
    labels = [METHOD_SHORT.get(m, m) for m in methods]
    acc_mean = plot_df["acc_mean"].to_numpy(dtype=float)
    acc_std = plot_df["acc_std"].to_numpy(dtype=float)
    colors = [METHOD_COLORS.get(m, ACCENT) for m in methods]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
    apply_clean_style(ax)

    ax.bar(x, acc_mean, yerr=acc_std, capsize=6, color=colors, alpha=0.92, width=0.62)

    ax.set_xticks(x, labels, fontsize=13)
    ax.set_ylabel("Accuracy (mean ± SD)", fontsize=14, color=TEXT)
    ax.set_xlabel("Method", fontsize=14, color=TEXT)
    ax.set_title("Classification Accuracy Across Similarity Metrics", fontsize=18, color=TEXT, pad=18)

    lower = max(0.985, float(acc_mean.min() - max(0.0015, np.nanmax(acc_std) * 1.3)))
    upper = min(1.0005, float(acc_mean.max() + max(0.0015, np.nanmax(acc_std) * 1.5)))
    ax.set_ylim(lower, upper)

    for i, v in enumerate(acc_mean):
        offset = (upper - lower) * 0.010
        ax.text(i, v + offset, f"{v:.4f}", ha="center", fontsize=11, color=TEXT, fontweight="bold")

    fig.tight_layout()
    fig.savefig(fig_dir / "accuracy.png", dpi=400, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_false_positive_rates(df_seed: pd.DataFrame, fig_dir: Path) -> None:
    methods = [m for m in METHOD_ORDER if m in df_seed["method"].unique()]
    labels = [METHOD_SHORT[m] for m in methods]
    values = [float(df_seed.loc[df_seed["method"] == m, "fpr"].mean()) for m in methods]

    fig, ax = plt.subplots(figsize=(7.4, 4.3), facecolor=BG)
    apply_clean_style(ax)

    ypos = np.arange(len(labels))
    ax.barh(ypos, values, color=[METHOD_COLORS[m] for m in methods], alpha=0.9, height=0.56)
    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("False Positive Rate", fontsize=13, color=TEXT)
    ax.set_title("False Positive Rate Comparison", fontsize=16, color=TEXT, pad=10)

    for y, v in zip(ypos, values):
        ax.text(v + max(values) * 0.02, y, f"{v:.4f}", va="center", fontsize=11, color=TEXT)

    fig.tight_layout()
    fig.savefig(fig_dir / "false_positive.png", dpi=380, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_roc_comparison(df_pairs: pd.DataFrame, fig_dir: Path) -> None:
    y = df_pairs["label"].to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(8.6, 6.5), facecolor=BG)
    apply_clean_style(ax, add_grid=False)

    roc_results = []
    for method in METHOD_ORDER:
        if method not in df_pairs.columns:
            raise KeyError(f"[plots] ROC: column '{method}' missing. Available: {list(df_pairs.columns)}")
        scores = df_pairs[method].to_numpy(dtype=float)
        fprs, tprs, auc = compute_roc(scores, y, METHOD_HIGHER_IS_SAME[method])
        roc_results.append((method, fprs, tprs, auc))

    best_method, _, _, _ = max(roc_results, key=lambda x: x[3])

    for method, fprs, tprs, auc in roc_results:
        linewidth = 3.8 if method == best_method else 2.4
        alpha = 1.0 if method == best_method else 0.95
        ax.plot(
            fprs,
            tprs,
            linewidth=linewidth,
            alpha=alpha,
            color=METHOD_COLORS[method],
            label=f"{METHOD_LABELS[method]} (AUC = {auc:.4f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2.0, color=RANDOM_LINE, label="Random classifier")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=15, color=TEXT)
    ax.set_ylabel("True Positive Rate", fontsize=15, color=TEXT)
    ax.set_title("ROC Comparison of Similarity Metrics for Forensic Drug Classification", fontsize=20, color=TEXT, pad=12)
    ax.legend(fontsize=12, frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(Path(fig_dir) / "figure_2_roc.png", dpi=400, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def make_threshold_variability_plot(df_seed: pd.DataFrame, fig_dir: Path) -> None:
    methods = [m for m in METHOD_ORDER if m in df_seed["method"].unique()]
    fig, axes = plt.subplots(len(methods), 1, figsize=(8.6, 7.8), facecolor=BG, sharex=True)

    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        apply_clean_style(ax)
        sub = df_seed[df_seed["method"] == method].sort_values("seed")
        ax.plot(
            sub["seed"],
            sub["threshold"],
            marker="o",
            markersize=5,
            linewidth=2.0,
            color=METHOD_COLORS[method],
        )
        thr_min = float(sub["threshold"].min())
        thr_max = float(sub["threshold"].max())
        span = thr_max - thr_min
        pad = max(0.001, span * 0.25)
        ax.set_ylim(thr_min - pad, thr_max + pad)
        ax.set_ylabel(f"{METHOD_SHORT[method]}\nthreshold", fontsize=12, color=TEXT, rotation=0, labelpad=35, va="center")
        ax.text(
            0.98,
            0.85,
            f"mean = {sub['threshold'].mean():.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10.2,
            color=TEXT,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=GRID, linewidth=1.0, alpha=0.92),
        )

    axes[0].set_title("Threshold Stability Across Random Seeds", fontsize=16, color=TEXT, pad=10)
    axes[-1].set_xlabel("Seed", fontsize=13, color=TEXT)

    fig.tight_layout()
    fig.savefig(fig_dir / "figure_3_thresholds.png", dpi=350, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def make_variability_boxplot(df_seed: pd.DataFrame, fig_dir: Path) -> None:
    methods = [m for m in METHOD_ORDER if m in df_seed["method"].unique()]
    data = [df_seed.loc[df_seed["method"] == m, "acc"].to_numpy(dtype=float) for m in methods]
    labels = [METHOD_SHORT[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor=BG)
    apply_clean_style(ax)

    positions = np.arange(1, len(methods) + 1)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.65,
        patch_artist=True,
        labels=labels,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(TEXT)
        patch.set_linewidth(1.5)

    for median in bp["medians"]:
        median.set_color(TEXT)
        median.set_linewidth(2)

    for whisker in bp["whiskers"]:
        whisker.set_color(TEXT)
        whisker.set_linewidth(1.3)

    for cap in bp["caps"]:
        cap.set_color(TEXT)
        cap.set_linewidth(1.3)

    ax.set_xlim(0.5, len(methods) + 0.5)
    ax.margins(x=0.02)

    ax.set_xlabel("Method", fontsize=13, color=TEXT)
    ax.set_ylabel("Accuracy Across Seeds", fontsize=13, color=TEXT)
    ax.set_title("Accuracy Variability Across Random Seeds", fontsize=16, color=TEXT, pad=10)

    fig.tight_layout()
    fig.savefig(fig_dir / "figure_3_accuracy_boxplot.png", dpi=350, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_spectral_examples(fig_dir: Path, groups, mz_min: int = 40) -> None:
    fig_dir = Path(fig_dir)
    keys = list(groups.keys())
    if len(keys) < 2:
        print("[plots] Not enough chemicals to plot Figure 4.")
        return

    k_same = None
    for k in keys:
        if len(groups[k]) >= 2:
            k_same = k
            break
    if k_same is None:
        print("[plots] No chemical with >=2 replicates found for SAME overlay.")
        return

    vecs_same = groups[k_same]
    v1 = vecs_same[0]
    v2 = vecs_same[1]

    k_diff2 = next((k for k in keys if k != k_same), None)
    if k_diff2 is None:
        print("[plots] Could not find a different chemical for DIFF overlay.")
        return
    v3 = groups[k_diff2][0]

    mz_axis = np.arange(mz_min, mz_min + len(v1))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.8), facecolor=BG, sharex=True)
    for ax in axes:
        apply_clean_style(ax)

    axes[0].plot(mz_axis, v1, color=REP1_COLOR, linewidth=2.0, label="Replicate 1")
    axes[0].plot(mz_axis, v2, color=REP2_COLOR, linewidth=1.8, alpha=0.95, label="Replicate 2")
    axes[0].set_title("Spectral Overlay: Same Compound Replicates", fontsize=15, color=TEXT, pad=8)
    axes[0].set_ylabel("Normalized Intensity", fontsize=12, color=TEXT)
    axes[0].legend(frameon=False, fontsize=9.5, loc="upper right")

    axes[1].plot(mz_axis, v1, color=ACCENT, linewidth=2.0, label="Compound A")
    axes[1].plot(mz_axis, v3, color=COSINE, linewidth=1.8, alpha=0.9, label="Compound B")
    axes[1].set_title("Spectral Overlay: Different Compounds", fontsize=15, color=TEXT, pad=8)
    axes[1].set_xlabel("m/z (integer bins)", fontsize=12, color=TEXT)
    axes[1].set_ylabel("Normalized Intensity", fontsize=12, color=TEXT)
    axes[1].legend(frameon=False, fontsize=9.5, loc="upper right")

    fig.tight_layout()
    fig.savefig(fig_dir / "figure_4_spectral_overlay.png", dpi=350, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------
# Flowchart and summary panels
# ---------------------------------------------------------------

def add_step_box(ax, x, y, w, h, title, subtitle=None, facecolor="white", edgecolor=ACCENT):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=15, color=TEXT, fontweight="bold")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.33, subtitle, ha="center", va="center", fontsize=11.5, color=MUTED)


def make_workflow_flowchart(fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 10.8), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    steps = [
        ("Raw Mass Spectra", "replicate measurements from compounds"),
        ("Preprocessing", "normalization, filtering, alignment"),
        ("Feature Representation", "integer-binned m/z intensity vectors"),
        ("Similarity Computation", "Cosine, δ, and δ′ metrics"),
        ("Classification", "same vs different compounds"),
        ("Evaluation", "ROC, accuracy, and false positives"),
    ]

    x = 0.16
    w = 0.68
    h = 0.105
    ys = [0.84, 0.69, 0.54, 0.39, 0.24, 0.09]

    for i, ((title, subtitle), y) in enumerate(zip(steps, ys)):
        face = ACCENT_LIGHT if i in {0, len(steps) - 1} else "white"
        add_step_box(ax, x, y, w, h, title, subtitle, facecolor=face)
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(0.50, ys[i + 1] + h + 0.01),
                xytext=(0.50, y - 0.01),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color=ACCENT),
            )

    ax.text(
        0.5,
        0.985,
        "Workflow for Replicate-Informed Forensic Drug Classification",
        ha="center",
        va="center",
        fontsize=20,
        color=TEXT,
        fontweight="bold",
    )

    ax.text(
        0.5,
        0.025,
        "Takeaway: replicate-aware similarity metrics preserve useful variability information for better classification.",
        ha="center",
        va="center",
        fontsize=11.5,
        color=TEXT,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=ACCENT_LIGHT, edgecolor=ACCENT, linewidth=1.2),
    )

    fig.subplots_adjust(top=0.97, bottom=0.03, left=0.05, right=0.95)
    fig.savefig(Path(fig_dir) / "workflow_flowchart.png", dpi=400, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def make_summary_table_figure(df_stats: pd.DataFrame, df_pairs: pd.DataFrame, df_seed: pd.DataFrame, fig_dir: Path) -> None:
    methods = [m for m in METHOD_ORDER if m in df_stats["method"].tolist() and m in df_pairs.columns]
    y = df_pairs["label"].to_numpy(dtype=int)

    rows = []
    aucs = {}
    fprs = {}
    for method in methods:
        scores = df_pairs[method].to_numpy(dtype=float)
        _, _, auc = compute_roc(scores, y, METHOD_HIGHER_IS_SAME[method])
        aucs[method] = auc
        fprs[method] = float(df_seed.loc[df_seed["method"] == method, "fpr"].mean())
        rows.append([
            METHOD_SHORT[method],
            _metric_value(df_stats, method, "acc_mean"),
            auc,
            fprs[method],
        ])

    table_df = pd.DataFrame(rows, columns=["Method", "Accuracy", "AUC", "False Positive Rate"])

    fig, ax = plt.subplots(figsize=(8.6, 3.6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    cell_text = []
    for _, row in table_df.iterrows():
        cell_text.append([
            row["Method"],
            f"{row['Accuracy']:.4f}",
            f"{row['AUC']:.4f}",
            f"{row['False Positive Rate']:.4f}",
        ])

    table = ax.table(
        cellText=cell_text,
        colLabels=["Method", "Accuracy", "AUC", "False Positive Rate"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.08, 1.95)

    best_acc_method = methods[int(np.argmax([_metric_value(df_stats, m, "acc_mean") for m in methods]))]
    best_auc_method = methods[int(np.argmax([aucs[m] for m in methods]))]
    best_fpr_method = methods[int(np.argmin([fprs[m] for m in methods]))]

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(1.1)
        if row == 0:
            cell.set_facecolor(ACCENT)
            cell.set_text_props(weight="bold", color="white")
        else:
            method = methods[row - 1]
            cell.set_facecolor("white")
            if col == 0:
                cell.set_text_props(weight="bold", color=TEXT)
            if (
                (col == 1 and method == best_acc_method)
                or (col == 2 and method == best_auc_method)
                or (col == 3 and method == best_fpr_method)
            ):
                cell.set_facecolor(ACCENT_LIGHT)
                cell.set_text_props(weight="bold", color=TEXT)

    ax.set_title("Performance Summary Across Similarity Metrics", fontsize=17, color=TEXT, pad=10)
    ax.text(
        0.5,
        0.06,
        "Highlighted cells show the strongest result in each column.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        color=MUTED,
    )

    fig.tight_layout()
    fig.savefig(Path(fig_dir) / "summary_table.png", dpi=400, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def make_key_findings_panel(df_stats: pd.DataFrame, df_pairs: pd.DataFrame, df_seed: pd.DataFrame, fig_dir: Path) -> None:
    methods = [m for m in METHOD_ORDER if m in df_stats["method"].tolist() and m in df_pairs.columns]
    y = df_pairs["label"].to_numpy(dtype=int)
    aucs = {m: compute_roc(df_pairs[m].to_numpy(dtype=float), y, METHOD_HIGHER_IS_SAME[m])[2] for m in methods}
    fprs = {m: float(df_seed.loc[df_seed["method"] == m, "fpr"].mean()) for m in methods}
    best_acc_method = max(methods, key=lambda m: _metric_value(df_stats, m, "acc_mean"))
    best_auc_method = max(methods, key=lambda m: aucs[m])
    best_fpr_method = min(methods, key=lambda m: fprs[m])

    fig, ax = plt.subplots(figsize=(8.2, 2.2), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    bullets = [
        f"Highest accuracy: {METHOD_SHORT[best_acc_method]} ({_metric_value(df_stats, best_acc_method, 'acc_mean'):.4f})",
        f"Highest AUC: {METHOD_SHORT[best_auc_method]} ({aucs[best_auc_method]:.4f})",
        f"Lowest false positive rate: {METHOD_SHORT[best_fpr_method]} ({fprs[best_fpr_method]:.4f})",
    ]

    panel = FancyBboxPatch(
        (0.04, 0.12),
        0.92,
        0.72,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.8,
        edgecolor=ACCENT,
        facecolor=ACCENT_LIGHT,
    )
    ax.add_patch(panel)
    ax.text(0.08, 0.76, "Key Findings", fontsize=16, fontweight="bold", color=TEXT, transform=ax.transAxes)
    for i, bullet in enumerate(bullets):
        ax.text(0.10, 0.56 - i * 0.19, f"• {bullet}", fontsize=12, color=TEXT, transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(Path(fig_dir) / "key_findings.png", dpi=400, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------
# Master function
# ---------------------------------------------------------------

def make_all_plots(out_dir: Path, groups=None, mz_min: int = 40) -> None:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = newest_file(out_dir, "pairs*.csv")
    stats_path = newest_file(out_dir, "summary_stats*.csv")
    seed_path = newest_file(out_dir, "summary_by_seed*.csv")

    df_pairs = pd.read_csv(pairs_path)
    df_stats = pd.read_csv(stats_path)
    df_seed = pd.read_csv(seed_path)

    print("[plots] Using newest files:")
    print(f"        pairs: {pairs_path.name}")
    print(f"        stats: {stats_path.name}")
    print(f"        seed : {seed_path.name}")

    for method in METHOD_ORDER:
        if method not in df_pairs.columns:
            raise KeyError(
                f"[plots] Column '{method}' not found in {pairs_path.name}. "
                f"Available columns: {list(df_pairs.columns)}"
            )

        threshold = best_threshold_from_seed_summary(df_seed, method)
        same = df_pairs.loc[df_pairs["label"] == 1, method]
        diff = df_pairs.loc[df_pairs["label"] == 0, method]

        make_histogram_plot(
            same=same,
            diff=diff,
            method=method,
            threshold=threshold,
            out_path=fig_dir / f"figure_1_dist_full_{method}.png",
            zoom=False,
        )
        make_histogram_plot(
            same=same,
            diff=diff,
            method=method,
            threshold=threshold,
            out_path=fig_dir / f"figure_1_dist_zoom_{method}.png",
            zoom=True,
        )

    make_accuracy_plot(df_stats, fig_dir)
    plot_false_positive_rates(df_seed, fig_dir)
    plot_roc_comparison(df_pairs, fig_dir)
    make_threshold_variability_plot(df_seed, fig_dir)
    make_variability_boxplot(df_seed, fig_dir)
    make_workflow_flowchart(fig_dir)
    make_summary_table_figure(df_stats, df_pairs, df_seed, fig_dir)
    make_key_findings_panel(df_stats, df_pairs, df_seed, fig_dir)

    if groups is not None:
        plot_spectral_examples(fig_dir, groups, mz_min=mz_min)
    else:
        print("[plots] Skipping Figure 4 (spectral overlay) because groups=None")

    print(f"[plots] Figures saved to: {fig_dir}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    make_all_plots(script_dir / "results")