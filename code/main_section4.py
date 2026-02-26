import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from enhanced_kmeans import EnhancedKMeans
from alternate_kmeans import AlternateKMeans


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results_section4")

K_VALUES = [2, 5, 10, 20]
N_RUNS = 5
MAX_ITER = 100

ALGOS = {
    "sklearn":   "Standard k-means",
    "enhanced":  "Section 3 (Enhanced)",
    "alternate": "Section 4 (Alternate)",
}


# ----------------------------
# Data
# ----------------------------

def load_datasets():
    """Load all three processed CSV datasets from the script directory."""
    datasets = {}
    for fname in ["D1_Processed.csv", "D2_Processed.csv", "D3_Processed.csv"]:
        path = os.path.join(SCRIPT_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            datasets[fname.replace(".csv", "")] = df
            print(f"Loaded {fname}: {df.shape[0]} x {df.shape[1]}")
        else:
            print(f"Missing: {fname}")
    return datasets


def preprocess(df):
    """Standardise numeric columns after filling missing values with column means."""
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.fillna(num.mean(numeric_only=True))
    return StandardScaler().fit_transform(num.values)


# ----------------------------
# Fit wrappers
# ----------------------------

def fit_sklearn(X, k, seed):
    """Fit sklearn KMeans and normalise attribute names to match custom classes."""
    m = KMeans(n_clusters=k, init="random", n_init=1,
               max_iter=MAX_ITER, random_state=seed, algorithm="lloyd")
    m.fit(X)
    m.centroids = m.cluster_centers_
    m.inertia_ = float(m.inertia_)
    m.n_iter_ = int(m.n_iter_)
    m.sse_history_ = [m.inertia_]
    m.reassigned_history_ = [np.nan]
    return m


def fit_enhanced(X, k, seed):
    """Fit EnhancedKMeans, adding history attributes if the class omits them."""
    m = EnhancedKMeans(k=k, max_iter=MAX_ITER, random_state=seed)
    m.fit(X)
    if not hasattr(m, "sse_history_"):
        m.sse_history_ = [float(m.inertia_)]
    if not hasattr(m, "reassigned_history_"):
        m.reassigned_history_ = [np.nan]
    return m


def fit_alternate(X, k, seed):
    """Fit the Section 4 AlternateKMeans."""
    m = AlternateKMeans(k=k, max_iter=MAX_ITER, random_state=seed)
    m.fit(X)
    return m


FITTERS = {
    "sklearn":   fit_sklearn,
    "enhanced":  fit_enhanced,
    "alternate": fit_alternate,
}


# ----------------------------
# Experiment runner
# ----------------------------

def run_single(algo, X, k, seed):
    """Fit one model for a given algorithm, k, and seed. Returns a result dict."""
    t0 = time.time()
    model = FITTERS[algo](X, k, seed)
    elapsed = time.time() - t0
    return {
        "model":              model,
        "sse":                float(model.inertia_),
        "iters":              int(model.n_iter_),
        "time":               elapsed,
        "sse_history":        list(model.sse_history_),
        "reassigned_history": list(model.reassigned_history_),
    }


def aggregate_runs(algo, runs):
    """Summarise N seed-runs into best-run details and averaged metrics."""
    best = min(runs, key=lambda r: r["sse"])
    return {
        "label":                   ALGOS[algo],
        "best_model":              best["model"],
        "best_sse_history":        best["sse_history"],
        "best_reassigned_history": best["reassigned_history"],
        "avg_sse":   np.mean([r["sse"]   for r in runs]),
        "avg_iters": np.mean([r["iters"] for r in runs]),
        "avg_time":  np.mean([r["time"]  for r in runs]),
    }


def run_dataset(X, dataset_name):
    """Run all algorithms x all k values x N_RUNS seeds for one dataset."""
    print(f"\n{'='*60}\n{dataset_name} | shape={X.shape}\n{'='*60}")
    results = {k: {} for k in K_VALUES}

    for k in K_VALUES:
        print(f"\nk={k}")
        for algo in ALGOS:
            runs = []
            for seed in range(N_RUNS):
                try:
                    run = run_single(algo, X, k, seed)
                    runs.append(run)
                    print(f"  {algo:<9} run {seed+1}: SSE={run['sse']:.2f}, "
                          f"iters={run['iters']}, t={run['time']:.3f}s")
                except Exception as e:
                    print(f"  {algo:<9} run {seed+1}: FAILED ({e})")

            results[k][algo] = aggregate_runs(algo, runs) if runs else None

    return results


# ----------------------------
# Plotting
# ----------------------------

def project_2d(X):
    """Reduce X to 2 dimensions with PCA if needed."""
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        return pca.fit_transform(X), pca, "PC1", "PC2"
    return X, None, "Feature 1", "Feature 2"


def plot_clusters(X, dataset_name, results):
    """
    One figure per k value: side-by-side scatter plots for each algorithm,
    with centroids marked in red.
    """
    X2, pca, xlabel, ylabel = project_2d(X)

    for k in K_VALUES:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, algo in zip(axes, ALGOS):
            r = results[k][algo]
            if r is None:
                ax.axis("off")
                continue
            m = r["best_model"]
            C = np.asarray(m.centroids)
            C2 = pca.transform(C) if pca is not None else C
            ax.scatter(X2[:, 0], X2[:, 1], c=m.labels_, cmap="tab10", s=15, alpha=0.6)
            ax.scatter(C2[:, 0], C2[:, 1], marker="X", s=150, c="red", edgecolors="black")
            ax.set_title(f"{r['label']}\nSSE={r['avg_sse']:.2f}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        plt.suptitle(f"{dataset_name} — Cluster visualisation (k={k})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{dataset_name}_clusters_k{k}.png"), dpi=140)
        plt.close()


def plot_convergence(dataset_name, results):
    """
    One figure per dataset: 2x2 grid of SSE-vs-iteration subplots,
    one per k value, with all algorithms overlaid on each subplot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, k in zip(axes, K_VALUES):
        for algo in ALGOS:
            r = results[k][algo]
            if r is None or not r["best_sse_history"]:
                continue
            y = r["best_sse_history"]
            ax.plot(range(1, len(y) + 1), y, marker="o", label=r["label"])
        ax.set_title(f"k={k}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("SSE")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(f"{dataset_name} — SSE convergence", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{dataset_name}_convergence.png"), dpi=140)
    plt.close()


def plot_stability(dataset_name, results):
    """
    One figure per dataset: 2x2 grid of reassigned-points-per-iteration
    subplots, one per k value, with all algorithms overlaid on each subplot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, k in zip(axes, K_VALUES):
        plotted_any = False
        for algo in ALGOS:
            r = results[k][algo]
            if r is None:
                continue
            y = np.array(r["best_reassigned_history"], dtype=float)
            if np.all(np.isnan(y)):
                continue
            ax.plot(range(1, len(y) + 1), y, marker="o", label=r["label"])
            plotted_any = True

        ax.set_title(f"k={k}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("# reassigned points")
        ax.grid(alpha=0.3)
        if plotted_any:
            ax.legend(fontsize=8)

    plt.suptitle(f"{dataset_name} — Reassignments per iteration", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{dataset_name}_stability.png"), dpi=140)
    plt.close()


def plot_runtime(all_results):
    """One runtime-vs-k figure per dataset."""
    for dataset_name, results in all_results.items():
        plt.figure(figsize=(7, 4.5))
        for algo, label in ALGOS.items():
            y = [results[k][algo]["avg_time"] if results[k][algo] else np.nan
                 for k in K_VALUES]
            plt.plot(K_VALUES, y, marker="o", label=label)

        plt.title(f"{dataset_name} — Runtime vs k")
        plt.xlabel("k")
        plt.ylabel("Wall-clock runtime (s)")
        plt.xticks(K_VALUES)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{dataset_name}_runtime_vs_k.png"), dpi=140)
        plt.close()


# ----------------------------
# Summary report
# ----------------------------

def save_summary(all_results):
    """Print and save a text table of average SSE, iterations, and runtime."""
    lines = ["SECTION 4 SUMMARY", "=" * 60]
    for dataset_name, results in all_results.items():
        lines += ["", f"Dataset: {dataset_name}", "-" * 60]
        for k in K_VALUES:
            lines += [f"k={k}",
                      f"{'Algorithm':<24}{'Avg SSE':<14}{'Avg Iter':<12}{'Avg Time':<10}"]
            for algo, label in ALGOS.items():
                r = results[k][algo]
                if r is None:
                    lines.append(f"{label:<24}FAILED")
                else:
                    lines.append(f"{label:<24}{r['avg_sse']:<14.3f}"
                                 f"{r['avg_iters']:<12.2f}{r['avg_time']:<10.4f}")
            lines.append("")

    text = "\n".join(lines)
    out = os.path.join(OUTPUT_DIR, "section4_summary.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


# ----------------------------
# Main
# ----------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    datasets = load_datasets()
    if not datasets:
        print("No datasets found.")
        return

    all_results = {}
    for dataset_name, df in datasets.items():
        X = preprocess(df)
        results = run_dataset(X, dataset_name)
        all_results[dataset_name] = results

        plot_clusters(X, dataset_name, results)
        plot_convergence(dataset_name, results)
        plot_stability(dataset_name, results)

    plot_runtime(all_results)
    save_summary(all_results)
    print(f"\nDone. Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()