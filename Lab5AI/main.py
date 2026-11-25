import os
import numpy as np

from data import prepare_dataset
from experiment import (
    run_algorithm_sweeps,
    pick_best,
    plot_elbow,
    plot_clusters,
    plot_results_comparison,
    print_best_summary,
    evaluate_clustering,
    RESULTS_DIR
)
from clustering import run_agglomerative


def save_metrics_to_txt(runs, filename):
    """Сохраняет все метрики экспериментов в текстовый файл."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== МЕТРИКИ КЛАСТЕРИЗАЦИИ ===\n\n")
        f.write(f"{'Алгоритм':<22} | {'Параметры':<25} | {'Кластеров':<10} | {'Silhouette':<10} | {'CH':<10} | {'DB':<10}\n")
        f.write("-" * 100 + "\n")

        for res, metrics in runs:
            params_str = str(res.params)
            sil = f"{metrics['silhouette']:.4f}" if not np.isnan(metrics['silhouette']) else "NaN"
            ch = f"{metrics['calinski']:.1f}" if not np.isnan(metrics['calinski']) else "NaN"
            db = f"{metrics['davies']:.4f}" if not np.isnan(metrics['davies']) else "NaN"

            f.write(f"{res.algorithm:<22} | {params_str:<25} | {res.n_clusters:<10} | {sil:<10} | {ch:<10} | {db:<10}\n")

    print(f"\n[Сохранено] Метрики всех экспериментов: {path}")


def main():
    print("Лабораторная работа №5. Кластеризация: водоочистка")
    print("Датасет: water-treatment.csv (UCI ML Repository)")

    # 1. Подготовка данных
    X_scaled, X_pca, columns, info = prepare_dataset()

    # 2. Эксперименты
    runs, inertias = run_algorithm_sweeps(X_scaled)

    # 3. Сохраняем текстовый отчет с метриками
    save_metrics_to_txt(runs, "metrics_report.txt")

    # 4. График метода локтя
    plot_elbow(inertias)

    # 5. Сравнение алгоритмов
    plot_results_comparison(runs)

    # 6. Выбор лучшей модели
    best_result, best_metrics = pick_best(runs)
    print_best_summary(best_result, best_metrics)

    # 7. Визуализация лучшей модели
    plot_clusters(
        X_pca,
        best_result,
        f"Лучшая модель: {best_result.algorithm} "
        f"{best_result.params} (sil={best_metrics['silhouette']:.3f})",
        filename="03_best_model.png"
    )

    # 8. Интерпретируемое разбиение (Agglomerative, k=3)
    agglo3 = run_agglomerative(X_scaled, 3, "ward")
    plot_clusters(X_pca, agglo3,
                  "Agglomerative (ward, k=3) — 3 режима работы станции",
                  filename="04_agglomerative_k3.png")

    # 9. Интерпретация кластеров
    print("\n=== Интерпретация разбиения (k=3) ===")
    print(f"Размеры кластеров: "
          f"{np.bincount(agglo3.labels + 1)[1:].tolist()}")
    print("Средние масштабированные значения по кластерам:")
    for c in sorted(set(agglo3.labels)):
        mean_vec = X_scaled[agglo3.labels == c].mean(axis=0)
        top = np.argsort(np.abs(mean_vec))[-3:][::-1]
        feats = ", ".join(
            f"{columns[i]}: {mean_vec[i]:+.2f}" for i in top
        )
        print(f"  Кластер {c}: {feats}")


if __name__ == "__main__":
    main()