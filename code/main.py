import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from enhanced_kmeans import EnhancedKMeans

sns.set_style("whitegrid")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')
K_VALUES   = [2, 5, 10, 20]
N_RUNS     = 5

def load_datasets():
    datasets = {}
    for name in ['D1_Processed.csv', 'D2_Processed.csv', 'D3_Processed.csv']:
        path = os.path.join(SCRIPT_DIR, name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            datasets[name.replace('.csv', '')] = df
            print(f"Loaded {name}: {df.shape[0]} rows x {df.shape[1]} cols")
        else:
            print(f"Not found: {name}")
    return datasets


def preprocess(df):
    numeric = df.select_dtypes(include=[np.number]).fillna(df.mean(numeric_only=True))
    return StandardScaler().fit_transform(numeric.values)


def run_experiments(X, dataset_name):
    print(f"\n{'='*60}\n{dataset_name}  |  shape: {X.shape}\n{'='*60}")
    results = {}

    for k in K_VALUES:
        print(f"\n  k={k}")
        runs = []
        for run in range(N_RUNS):
            model = EnhancedKMeans(k=k, max_iter=100, random_state=run)
            t0 = time.time()
            model.fit(X)
            elapsed = time.time() - t0
            runs.append({'model': model, 'sse': model.inertia_,
                         'n_iter': model.n_iter_, 'time': elapsed})
            print(f"    run {run+1}: SSE={model.inertia_:.1f}  iters={model.n_iter_}  t={elapsed:.3f}s")

        best = runs[np.argmin([r['sse'] for r in runs])]
        results[k] = {
            'best_model':   best['model'],
            'avg_sse':      np.mean([r['sse']    for r in runs]),
            'avg_iterations': np.mean([r['n_iter'] for r in runs]),
            'avg_time':     np.mean([r['time']   for r in runs]),
        }

    return results

def plot_clusters(X, results, dataset_name):
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(X)
        var = pca.explained_variance_ratio_.sum()
        xlabel, ylabel = f"PC1 ({var*100:.1f}% var)", "PC2"
        def project(pts): return pca.transform(pts)
    else:
        X2 = X
        xlabel, ylabel = "Feature 1", "Feature 2"
        def project(pts): return pts

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, k in zip(axes.flatten(), K_VALUES):
        model  = results[k]['best_model']
        ax.scatter(X2[:, 0], X2[:, 1], c=model.labels_, cmap='tab10',
                   alpha=0.5, s=20, linewidths=0)
        c2 = project(model.centroids)
        ax.scatter(c2[:, 0], c2[:, 1], c='red', marker='X',
                   s=200, edgecolors='black', linewidths=1.5, zorder=5)
        ax.set_title(f"k={k}   SSE={results[k]['avg_sse']:.1f}", fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    plt.suptitle(f"Cluster Assignments — {dataset_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{dataset_name}_clusters.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_sse(results, dataset_name):
    sse   = [results[k]['avg_sse']        for k in K_VALUES]
    iters = [results[k]['avg_iterations'] for k in K_VALUES]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(K_VALUES, sse, marker='o', color='steelblue', linewidth=2)
    axes[0].set_title("SSE vs k", fontweight='bold')
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("SSE")
    axes[0].set_xticks(K_VALUES)
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(K_VALUES, iters, color='skyblue', edgecolor='black', alpha=0.75, width=1.5)
    axes[1].set_title("Avg Iterations to Convergence", fontweight='bold')
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Iterations")
    axes[1].set_xticks(K_VALUES)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle(f"SSE Analysis — {dataset_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{dataset_name}_sse.png"), dpi=150, bbox_inches='tight')
    plt.close()

def save_report(all_results):
    lines = [
        "=" * 70,
        "ENHANCED K-MEANS CLUSTERING — ANALYSIS REPORT",
        "=" * 70,
        "",
        "INITIALIZATION: Density-Aware Spread Initialization (DASI)",
        "-" * 70,
        "  First centroid : highest local density point (most representative)",
        "  Next centroids : maximise score = 0.5*spread + 0.3*density + 0.2*spread^2",
        "  Time complexity: O(n^2*d + k^2*n*d), approx O(n^2*d) for k << n",
        "",
        "ITERATIVE STEP",
        "-" * 70,
        "  Distance: vectorised via ||x-c||^2 = ||x||^2 + ||c||^2 - 2*x*c^T  O(n*k*d)",
        "  Update  : mean of assigned points per cluster                       O(n*d)",
        "  Stopping: early exit when max centroid shift < tol",
        "  Empty   : empty clusters reinitialised to a random point",
        "  Total   : O(n^2*d + T*n*k*d)  where T = iterations until convergence",
        "",
        "",
        "RESULTS",
        "=" * 70,
    ]

    for dataset_name, results in all_results.items():
        lines += [
            "",
            f"Dataset: {dataset_name}",
            "-" * 70,
            f"{'k':<6} {'Avg SSE':<16} {'Avg Iters':<12}",
            "-" * 70,
        ]
        for k in K_VALUES:
            r = results[k]
            lines.append(f"{k:<6} {r['avg_sse']:<16.2f} {r['avg_iterations']:<12.1f}")
        lines.append("")

    report = "\n".join(lines)
    path = os.path.join(OUTPUT_DIR, 'analysis_report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print("\n" + report)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    datasets = load_datasets()
    if not datasets:
        print(f"No datasets found. Place CSV files in: {SCRIPT_DIR}")
        return

    all_results = {}
    for name, df in datasets.items():
        X = preprocess(df)
        results = run_experiments(X, name)
        all_results[name] = results

        plot_clusters(X, results, name)
        plot_sse(results, name)
        print(f"Plots saved for {name}")

    save_report(all_results)
    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()