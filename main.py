import re
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# USER CONFIG
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT
MZ_MIN = 40
MZ_MAX = 500
SEEDS = list(range(20))
N_SAME = 300
N_DIFF_CROSS = 300
N_DIFF_WITHIN = 300
N_GRID = 250


def safe_to_csv(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    """
    Write CSV safely on Windows even when the target file may be locked.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=index)

    try:
        tmp.replace(path)
        return path
    except PermissionError:
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        alt = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        tmp.replace(alt)
        print(f"[WARN] Could not overwrite (locked?): {path}")
        print(f"[WARN] Wrote instead: {alt}")
        return alt


def read_spectrum_csv(path: Path) -> pd.DataFrame:
    """
    Read one spectrum from a CSV-like file that may be comma, tab, or whitespace delimited,
    and may contain comment lines starting with '#'.

    Returns DataFrame with columns: mz, intensity
    """
    df = pd.read_csv(
        path,
        comment="#",
        sep=r"\s+|,|\t",
        engine="python",
        header=0
    )

    cols = [str(c).strip() for c in df.columns]
    numeric_like = 0
    for c in cols:
        try:
            float(c)
            numeric_like += 1
        except Exception:
            pass

    if numeric_like >= 2:
        df = pd.read_csv(
            path,
            comment="#",
            sep=r"\s+|,|\t",
            engine="python",
            header=None
        )

        if df.shape[1] < 3:
            raise ValueError(
                f"{path} has {df.shape[1]} columns; expected at least 3 (Point, mz, intensity)."
            )

        out = df.iloc[:, [1, 2]].copy()
        out.columns = ["mz", "intensity"]

    else:
        mz_col = next(
            (
                c for c in df.columns
                if "thompson" in str(c).lower()
                or str(c).lower().startswith("x(")
                or str(c).lower() in {"mz", "m/z", "x"}
            ),
            None,
        )
        int_col = next(
            (
                c for c in df.columns
                if "count" in str(c).lower()
                or str(c).lower().startswith("y(")
                or "intensity" in str(c).lower()
                or str(c).lower() in {"y"}
            ),
            None,
        )

        if mz_col is None or int_col is None:
            raise ValueError(
                f"Could not find m/z & intensity columns in {path}. "
                f"Columns found: {list(df.columns)}"
            )

        out = df[[mz_col, int_col]].copy()
        out.columns = ["mz", "intensity"]

    out["mz"] = pd.to_numeric(out["mz"], errors="coerce")
    out["intensity"] = pd.to_numeric(out["intensity"], errors="coerce")
    out = out.dropna()
    out = out[out["intensity"] > 0]
    out = out.sort_values("mz").reset_index(drop=True)

    return out


def spectrum_to_vector(peaks_df: pd.DataFrame, mz_min: int, mz_max: int) -> np.ndarray:
    """
    Convert peak list to fixed-length integer m/z binned vector.
    """
    df = peaks_df.copy()
    df["mz_int"] = df["mz"].round().astype(int)

    df = df[(df["mz_int"] >= mz_min) & (df["mz_int"] <= mz_max)]
    grouped = df.groupby("mz_int")["intensity"].max()

    v = np.zeros(mz_max - mz_min + 1, dtype=float)
    for mz_val, inten in grouped.items():
        v[mz_val - mz_min] = float(inten)

    maxv = v.max()
    if maxv > 0:
        v = v / maxv

    return v


def parse_filename_chem_rep(filename: str) -> Tuple[int, int]:
    stem = filename.rsplit(".", 1)[0]
    m = re.match(r"^\s*(\d{1,3})-(\d{1,3})\s*$", stem)
    if not m:
        raise ValueError(f"Filename '{filename}' does not match 'NN-NN' pattern.")
    return int(m.group(1)), int(m.group(2))


def resolve_data_root(root: Path, explicit_root: Optional[Path] = None) -> Path:
    """
    Resolve the directory that directly contains CM1, CM2, CM3, and Isomers.
    """
    candidates = []

    if explicit_root is not None:
        candidates.append(Path(explicit_root))

    candidates.extend([
        root / "Dataset",
        root / "Dataset" / "minmax_SourceAndData",
        root / "Dataset" / "minmax_SourceAndData" / "C-ComputingIndices",
        root / "Dataset" / "minmax_SourceAndData" / "C-ComputingIndices" / "CosineSimilarity",
        root / "Dataset" / "minmax_SourceAndData" / "SourceAndData",
        root / "Dataset" / "minmax_SourceAndData" / "Data",
    ])

    required = {"CM1", "CM2", "CM3", "Isomers"}

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            names = {p.name for p in candidate.iterdir()}
            print("Checking candidate root:", candidate)
            print("Contents:", sorted(names))
            if required.issubset(names):
                return candidate

    raise FileNotFoundError(
        "Could not find the data folder containing CM1, CM2, CM3, and Isomers.\n"
        f"Checked these locations:\n" + "\n".join(str(p) for p in candidates)
    )


def load_all_spectra(root_dir: Path, mz_min: int = 40, mz_max: int = 500):
    families = ["CM1", "CM2", "CM3", "Isomers"]
    records = []

    print("Using spectra root:", root_dir.resolve())

    for fam in families:
        folder = root_dir / fam
        print("Checking folder (full path):", folder.resolve())

        if not folder.exists():
            available = [p.name for p in root_dir.iterdir()] if root_dir.exists() else []
            raise FileNotFoundError(
                f"Missing folder: {folder}\n"
                f"Available items in {root_dir}: {available}"
            )

        csv_files = sorted(folder.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files in {fam}")

        for f in csv_files:
            chem_id, rep_id = parse_filename_chem_rep(f.name)
            peaks = read_spectrum_csv(f)
            vec = spectrum_to_vector(peaks, mz_min=mz_min, mz_max=mz_max)

            records.append({
                "family": fam,
                "chemical_id": chem_id,
                "replicate_id": rep_id,
                "path": str(f),
                "vector": vec,
            })

    return records


def group_replicates(records: List[Dict[str, Any]]) -> Dict[Tuple[str, int], List[np.ndarray]]:
    groups: Dict[Tuple[str, int], List[np.ndarray]] = defaultdict(list)
    for r in records:
        groups[(r["family"], r["chemical_id"])].append(r["vector"])
    return groups


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def max_pairwise_cosine(A: List[np.ndarray], B: List[np.ndarray]) -> float:
    return float(max(cosine_sim(a, b) for a in A for b in B))


def minmax_delta(
    vectors_A: List[np.ndarray],
    vectors_B: List[np.ndarray],
    robust: bool = False,
    q_low: float = 0.10,
    q_high: float = 0.90
) -> Optional[float]:
    """
    Min-Max delta based on cosine similarities.
    """
    if len(vectors_A) < 2 or len(vectors_B) < 2:
        return None

    S11 = [
        cosine_sim(vectors_A[i], vectors_A[j])
        for i in range(len(vectors_A)) for j in range(i + 1, len(vectors_A))
    ]
    S22 = [
        cosine_sim(vectors_B[i], vectors_B[j])
        for i in range(len(vectors_B)) for j in range(i + 1, len(vectors_B))
    ]
    S12 = [cosine_sim(a, b) for a in vectors_A for b in vectors_B]

    if len(S11) == 0 or len(S22) == 0 or len(S12) == 0:
        return None

    if robust:
        left = min(np.quantile(S11, q_low), np.quantile(S22, q_low))
        right = np.quantile(S12, q_high)
    else:
        left = min(min(S11), min(S22))
        right = max(S12)

    return float(left - right)


def split_replicates(vecs: List[np.ndarray], n_first: int = 5) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    vecs = list(vecs)
    random.shuffle(vecs)

    if len(vecs) <= n_first:
        mid = max(1, len(vecs) // 2)
        return vecs[:mid], vecs[mid:]
    return vecs[:n_first], vecs[n_first:]


def build_pair_dataset(
    groups: Dict[Tuple[str, int], List[np.ndarray]],
    n_same: int = 300,
    n_diff_cross: int = 300,
    n_diff_within: int = 300,
    seed: int = 42
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    keys = list(groups.keys())
    by_family: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for k in keys:
        by_family[k[0]].append(k)

    rows: List[Dict[str, Any]] = []

    same_candidates = [k for k in keys if len(groups[k]) >= 4]
    if not same_candidates:
        raise ValueError("No chemicals have at least 4 replicates, so SAME-pair construction cannot proceed.")

    for _ in range(n_same):
        k = random.choice(same_candidates)
        vecs = groups[k]
        A, B = split_replicates(vecs, n_first=min(5, max(2, len(vecs) // 2)))
        if len(A) < 2 or len(B) < 2:
            continue

        cos_best = max_pairwise_cosine(A, B)
        delta = minmax_delta(A, B, robust=False)
        delta_prime = minmax_delta(A, B, robust=True)

        rows.append({
            "label": 1,
            "type": "same",
            "fam_A": k[0], "chem_A": k[1],
            "fam_B": k[0], "chem_B": k[1],
            "cosine_best": cos_best,
            "delta": delta,
            "delta_prime": delta_prime,
        })

    fams = [f for f in by_family.keys() if len(by_family[f]) > 0]
    if len(fams) < 2:
        raise ValueError("Need at least two families to build diff_cross pairs.")

    for _ in range(n_diff_cross):
        famA, famB = random.sample(fams, 2)
        kA = random.choice(by_family[famA])
        kB = random.choice(by_family[famB])
        A = groups[kA]
        B = groups[kB]
        if len(A) < 2 or len(B) < 2:
            continue

        A_sub = random.sample(A, min(5, len(A)))
        B_sub = random.sample(B, min(5, len(B)))

        cos_best = max_pairwise_cosine(A_sub, B_sub)
        delta = minmax_delta(A_sub, B_sub, robust=False)
        delta_prime = minmax_delta(A_sub, B_sub, robust=True)

        rows.append({
            "label": 0,
            "type": "diff_cross",
            "fam_A": kA[0], "chem_A": kA[1],
            "fam_B": kB[0], "chem_B": kB[1],
            "cosine_best": cos_best,
            "delta": delta,
            "delta_prime": delta_prime,
        })

    for _ in range(n_diff_within):
        fam = random.choice(fams)
        if len(by_family[fam]) < 2:
            continue
        kA, kB = random.sample(by_family[fam], 2)
        A = groups[kA]
        B = groups[kB]
        if len(A) < 2 or len(B) < 2:
            continue

        A_sub = random.sample(A, min(5, len(A)))
        B_sub = random.sample(B, min(5, len(B)))

        cos_best = max_pairwise_cosine(A_sub, B_sub)
        delta = minmax_delta(A_sub, B_sub, robust=False)
        delta_prime = minmax_delta(A_sub, B_sub, robust=True)

        rows.append({
            "label": 0,
            "type": "diff_within",
            "fam_A": kA[0], "chem_A": kA[1],
            "fam_B": kB[0], "chem_B": kB[1],
            "cosine_best": cos_best,
            "delta": delta,
            "delta_prime": delta_prime,
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["delta", "delta_prime", "cosine_best"]).reset_index(drop=True)
    return df


def evaluate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    higher_is_same: bool = True
) -> Dict[str, float]:
    if higher_is_same:
        preds = (scores >= threshold).astype(int)
    else:
        preds = (scores <= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    tpr = tp / max(1, (tp + fn))
    fpr = fp / max(1, (fp + tn))
    return {"acc": acc, "tpr": tpr, "fpr": fpr, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def find_best_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    n_grid: int = 200,
    higher_is_same: bool = True
) -> Dict[str, Any]:
    lo, hi = float(np.nanmin(scores)), float(np.nanmax(scores))
    if lo == hi:
        m = evaluate_threshold(scores, labels, threshold=lo, higher_is_same=higher_is_same)
        return {
            "best_tpr_minus_fpr": {"threshold": lo, "metric": m, "obj": m["tpr"] - m["fpr"]},
            "best_acc": {"threshold": lo, "metric": m},
        }

    best_tpr_fpr = None
    best_acc = None

    for t in np.linspace(lo, hi, n_grid):
        m = evaluate_threshold(scores, labels, threshold=float(t), higher_is_same=higher_is_same)
        obj = m["tpr"] - m["fpr"]

        if best_tpr_fpr is None or obj > best_tpr_fpr["obj"]:
            best_tpr_fpr = {"threshold": float(t), "metric": m, "obj": float(obj)}

        if best_acc is None or m["acc"] > best_acc["metric"]["acc"]:
            best_acc = {"threshold": float(t), "metric": m}

    return {"best_tpr_minus_fpr": best_tpr_fpr, "best_acc": best_acc}


def metrics_from_best(best: Dict[str, Any]) -> Dict[str, float]:
    m = best["best_tpr_minus_fpr"]["metric"]
    return {
        "threshold": float(best["best_tpr_minus_fpr"]["threshold"]),
        "acc": float(m["acc"]),
        "tpr": float(m["tpr"]),
        "fpr": float(m["fpr"]),
        "tp": float(m["tp"]),
        "tn": float(m["tn"]),
        "fp": float(m["fp"]),
        "fn": float(m["fn"]),
        "obj_tpr_minus_fpr": float(best["best_tpr_minus_fpr"]["obj"]),
    }


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    d = a - b
    n = len(d)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(np.mean(d[idx])))
    boots = np.array(boots)
    return float(np.mean(d)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def paired_significance_table(df_seed: pd.DataFrame, metric: str = "acc") -> pd.DataFrame:
    methods = sorted(df_seed["method"].unique())
    pivot = df_seed.pivot_table(index="seed", columns="method", values=metric, aggfunc="mean")

    rows = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1, m2 = methods[i], methods[j]
            a = pivot[m1].to_numpy(dtype=float)
            b = pivot[m2].to_numpy(dtype=float)

            mask = np.isfinite(a) & np.isfinite(b)
            a = a[mask]
            b = b[mask]
            if len(a) < 3:
                continue

            mean_diff, ci_low, ci_high = bootstrap_mean_diff(a, b, n_boot=5000, seed=0)

            pval = np.nan
            try:
                from scipy import stats
                t = stats.ttest_rel(a, b)
                pval = float(t.pvalue)
            except Exception:
                pass

            rows.append({
                "metric": metric,
                "method_A": m1,
                "method_B": m2,
                "mean_A": float(np.mean(a)),
                "mean_B": float(np.mean(b)),
                "mean_diff_A_minus_B": float(mean_diff),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "p_value_paired_ttest": pval,
                "n_seeds_used": int(len(a)),
            })

    return pd.DataFrame(rows)


def run_multiseed_experiment(
    groups: Dict[Tuple[str, int], List[np.ndarray]],
    out_dir: Path,
    seeds: List[int],
    n_same: int = 300,
    n_diff_cross: int = 300,
    n_diff_within: int = 300,
    n_grid: int = 250,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []

    for seed in seeds:
        df_pairs = build_pair_dataset(
            groups,
            n_same=n_same,
            n_diff_cross=n_diff_cross,
            n_diff_within=n_diff_within,
            seed=seed
        )
        y = df_pairs["label"].to_numpy(dtype=int)

        scores = df_pairs["cosine_best"].to_numpy(dtype=float)
        best = find_best_threshold(scores, y, n_grid=n_grid, higher_is_same=True)
        rows.append({"seed": seed, "method": "cosine_best", **metrics_from_best(best)})

        for col in ["delta", "delta_prime"]:
            scores = df_pairs[col].to_numpy(dtype=float)
            best = find_best_threshold(scores, y, n_grid=n_grid, higher_is_same=False)
            rows.append({"seed": seed, "method": col, **metrics_from_best(best)})

        print(f"[seed={seed}] done. pairs={len(df_pairs)}")

    df_seed = pd.DataFrame(rows)

    agg = df_seed.groupby("method").agg({
        "threshold": ["mean", "std"],
        "acc": ["mean", "std"],
        "tpr": ["mean", "std"],
        "fpr": ["mean", "std"],
        "obj_tpr_minus_fpr": ["mean", "std"],
    })
    agg.columns = [f"{a}_{b}" for (a, b) in agg.columns]
    agg = agg.reset_index()

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    p1 = safe_to_csv(df_seed, out_dir / "summary_by_seed.csv", index=False)
    p2 = safe_to_csv(agg, out_dir / "summary_stats.csv", index=False)

    sig_acc = paired_significance_table(df_seed, metric="acc")
    sig_obj = paired_significance_table(df_seed, metric="obj_tpr_minus_fpr")
    sig = pd.concat([sig_acc, sig_obj], ignore_index=True)
    p3 = safe_to_csv(sig, out_dir / "significance_tests.csv", index=False)

    print("\nSaved:")
    print(" ", p1)
    print(" ", p2)
    print(" ", p3)

    return df_seed, agg, sig


def print_summary_table(df_stats: pd.DataFrame) -> None:
    show_cols = [
        "method",
        "acc_mean", "acc_std",
        "tpr_mean", "tpr_std",
        "fpr_mean", "fpr_std",
        "threshold_mean", "threshold_std",
    ]
    print("\nSummary (mean ± std):")
    print(df_stats[show_cols].to_string(index=False))


def main():
    root = ROOT
    print("ROOT =", root)
    print("ROOT contents =", [p.name for p in root.iterdir()])

    mz_min, mz_max = MZ_MIN, MZ_MAX

    print("Loading spectra...")
    data_root = resolve_data_root(root, explicit_root=DEFAULT_DATA_ROOT)
    print("DATA ROOT =", data_root)
    print("DATA ROOT contents =", [p.name for p in data_root.iterdir()])

    records = load_all_spectra(data_root, mz_min=mz_min, mz_max=mz_max)
    print("Total spectra loaded:", len(records))

    groups = group_replicates(records)
    print("Total chemicals (family, id):", len(groups))

    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)

    print("\nRunning multi-seed evaluation...")
    _, df_stats, _ = run_multiseed_experiment(
        groups=groups,
        out_dir=out_dir,
        seeds=SEEDS,
        n_same=N_SAME,
        n_diff_cross=N_DIFF_CROSS,
        n_diff_within=N_DIFF_WITHIN,
        n_grid=N_GRID,
    )

    print_summary_table(df_stats)

    print("\nBuilding pair dataset for plots...")
    df_pairs = build_pair_dataset(
        groups,
        n_same=N_SAME,
        n_diff_cross=N_DIFF_CROSS,
        n_diff_within=N_DIFF_WITHIN,
        seed=42
    )
    print("Pairs built:", len(df_pairs))
    print(df_pairs["type"].value_counts())

    pairs_path = out_dir / "pairs.csv"
    pairs_written = safe_to_csv(df_pairs, pairs_path, index=False)
    print(f"\nSaved: {pairs_written}")

    from plots import make_all_plots
    make_all_plots(out_dir, groups, mz_min=mz_min)

    print("\nDONE. Check:")
    print(" - results/summary_by_seed.csv")
    print(" - results/summary_stats.csv")
    print(" - results/significance_tests.csv")
    print(" - results/pairs.csv")
    print(" - results/figures/")


if __name__ == "__main__":
    main()
