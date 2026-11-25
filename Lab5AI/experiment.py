import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.cluster import KMeans

from clustering import (
    ClusteringResult,
    run_kmeans,
    run_dbscan,
    run_agglomerative,
    run_gmm,
)

# Создаем папку для результатов, если её нет
RESULTS_DIR = "results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def evaluate_clustering(X: np.ndarray,
                        result: ClusteringResult) -> Dict[str, float]:
    """
    Считает внутренние метрики качества кластеризации.
    Шумовые точки DBSCAN (-1) исключаются из расчёта.
    """
    mask = result.labels != -1
    labels = result.labels[mask]
    X_eval = X[mask]

    n_clusters = len(set(labels))

    if n_clusters < 2:
        return {
            "silhouette": np.nan,
            "calinski": np.nan,
            "davies": np.nan,
            "n_eval": int(mask.sum()),
            "n_clusters": n_clusters,
        }

    return {
        "silhouette": float(silhouette_score(X_eval, labels)),
        "calinski": float(calinski_harabasz_score(X_eval, labels)),
        "davies": float(davies_bouldin_score(X_eval, labels)),
        "n_eval": int(mask.sum()),
        "n_clusters": n_clusters,
    }


def run_algorithm_sweeps(X: np.ndarray) -> Tuple[List[Tuple[ClusteringResult, Dict]], Dict[int, float]]:
    """
    Прогоняет все алгоритмы по сеткам параметров.
    Возвращает список пар (результат, метрики) и словарь инерций для метода локтя.
    """
    runs: List[Tuple[ClusteringResult, Dict]] = []
    inertias = {}

    # --- KMeans: подбор k по локтю / силуэту ---
    print("\n--- KMeans: перебор k ---")
    for k in range(2, 11):
        res = run_kmeans(X, k)
        metrics = evaluate_clustering(X, res)
        runs.append((res, metrics))

        inertias[k] = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_

        print(f"k={k:2d} | silhouette={metrics['silhouette']:.3f} "
              f"| CH={metrics['calinski']:.1f} "
              f"| DB={metrics['davies']:.3f}")

    # --- DBSCAN: сетка eps × min_samples ---
    print("\n--- DBSCAN: перебор eps / min_samples ---")
    for eps in [0.8, 1.0, 1.3, 1.6, 2.0]:
        for ms in [3, 5, 8, 10]:
            res = run_dbscan(X, eps, ms)
            metrics = evaluate_clustering(X, res)
            runs.append((res, metrics))
            if res.n_clusters >= 2:
                print(f"eps={eps:.1f} ms={ms:2d} | кластеров={res.n_clusters} "
                      f"шум={res.n_noise:3d} | sil={metrics['silhouette']:.3f}")

    # --- Agglomerative: сетка linkage × k ---
    print("\n--- Agglomerative: перебор linkage / k ---")
    for linkage in ["ward", "complete", "average"]:
        for k in [2, 3, 4, 5]:
            res = run_agglomerative(X, k, linkage)
            metrics = evaluate_clustering(X, res)
            runs.append((res, metrics))
            print(f"{linkage:8s} k={k} | sil={metrics['silhouette']:.3f} "
                  f"| CH={metrics['calinski']:.1f} "
                  f"| DB={metrics['davies']:.3f}")

    # --- GMM: перебор k ---
    print("\n--- GMM: перебор k ---")
    for k in [2, 3, 4, 5]:
        res = run_gmm(X, k)
        metrics = evaluate_clustering(X, res)
        runs.append((res, metrics))
        print(f"k={k} | sil={metrics['silhouette']:.3f} "
              f"| CH={metrics['calinski']:.1f} "
              f"| DB={metrics['davies']:.3f}")

    return runs, inertias


def pick_best(runs: List[Tuple[ClusteringResult, Dict]]) -> Tuple[ClusteringResult, Dict]:
    """
    Выбор лучшей модели по силуэту (выше — лучше),
    при равенстве — по Calinski-Harabasz (выше — лучше).
    """
    valid = [(r, m) for r, m in runs
             if not np.isnan(m["silhouette"]) and m["n_clusters"] >= 2]

    if not valid:
        raise ValueError("Не найдено ни одной валидной модели с >= 2 кластерами!")

    return max(valid, key=lambda x: (round(x[1]["silhouette"], 3),
                                     x[1]["calinski"]))


# ----------------------------- визуализация -----------------------------

def plot_elbow(inertias: Dict[int, float]):
    """Строит график метода локтя."""
    ks = sorted(inertias)
    plt.figure(figsize=(8, 5))
    plt.plot(ks, [inertias[k] for k in ks], "bo-")
    plt.title("Метод локтя (KMeans)")
    plt.xlabel("Число кластеров k")
    plt.ylabel("Сумма квадратов внутри кластера (inertia)")
    plt.xticks(ks)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "01_elbow_method.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[Сохранено] {path}")


def plot_clusters(X_pca: np.ndarray, result: ClusteringResult,
                  title: str, filename: str):
    """Визуализирует кластеры на 2D-проекции PCA."""
    plt.figure(figsize=(9, 6))
    labels = result.labels
    unique = sorted(set(labels))

    cmap = plt.get_cmap("viridis", max(len(unique), 2))

    for i, lab in enumerate(unique):
        mask = labels == lab
        color = "grey" if lab == -1 else cmap(i)
        name = "шум" if lab == -1 else f"кластер {lab}"
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    s=18, color=color, label=name, alpha=0.8)

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[Сохранено] {path}")


def plot_results_comparison(runs: List[Tuple[ClusteringResult, Dict]]):
    """
    Точечный график: по оси X — число кластеров,
    по оси Y — silhouette, цвет — алгоритм.
    """
    algos = sorted(set(r.algorithm for r, _ in runs))
    markers = ["o", "s", "^", "D", "v"]

    plt.figure(figsize=(10, 6))
    for i, algo in enumerate(algos):
        points = []
        for r, m in runs:
            if r.algorithm == algo and not np.isnan(m["silhouette"]):
                points.append((m["n_clusters"], m["silhouette"]))

        if not points:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.scatter(xs, ys, label=algo, marker=markers[i % len(markers)], s=60)

    plt.title("Сравнение алгоритмов кластеризации")
    plt.xlabel("Число кластеров")
    plt.ylabel("Silhouette score (выше — лучше)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "02_algorithms_comparison.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[Сохранено] {path}")


def print_best_summary(best_result: ClusteringResult, best_metrics: Dict):
    """Выводит сводку по лучшей модели в консоль."""
    print("\n=== Лучшая модель ===")
    print(f"Алгоритм: {best_result.algorithm}")
    print(f"Параметры: {best_result.params}")
    print(f"Кластеров: {best_result.n_clusters}")
    print(f"Шумовых точек: {best_result.n_noise}")
    print(f"Silhouette: {best_metrics['silhouette']:.4f}")
    print(f"Calinski-Harabasz: {best_metrics['calinski']:.2f}")
    print(f"Davies-Bouldin: {best_metrics['davies']:.4f}")