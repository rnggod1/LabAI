import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Путь к файлу относительно папки, где лежит data.py
FILE_PATH = Path(__file__).parent / "water-treatment.csv"

# Пропуски в файле обозначены символом '?'
NA_VALUES = "?"

# Доля пропусков, выше которой столбец удаляется
MISSING_THRESHOLD = 0.5

RANDOM_STATE = 42


def load_raw(path: str = FILE_PATH) -> pd.DataFrame:
    """
    Загрузка сырого набора данных.
    Первый столбец — дата (D-день/месяц/год), для кластеризации
    он не нужен и удаляется. Все остальные столбцы приводятся
    к числовому типу, '?' превращается в NaN.
    """
    df = pd.read_csv(path, header=None, na_values=NA_VALUES)
    # Первый столбец — дата, удаляем
    df = df.drop(columns=[0])
    # Принудительно к числовому типу
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def drop_sparse_columns(df: pd.DataFrame,
                        threshold: float = MISSING_THRESHOLD) -> pd.DataFrame:
    """
    Удаляем столбцы, в которых доля пропусков больше threshold.
    """
    missing_ratio = df.isna().mean()
    keep = missing_ratio[missing_ratio <= threshold].index
    dropped = missing_ratio[missing_ratio > threshold].index

    print(f"Удалено разреженных столбцов: {len(dropped)} "
          f"(порог пропусков > {threshold:.0%})")
    if len(dropped):
        print("  индексы (исходные):", list(dropped))

    return df[keep]


def impute_median(df: pd.DataFrame) -> np.ndarray:
    """
    Заполнение оставшихся пропусков медианой столбца.
    """
    imputer = SimpleImputer(strategy="median")
    return imputer.fit_transform(df)


def scale_features(X: np.ndarray) -> np.ndarray:
    """
    Z-нормализация признаков.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def reduce_for_plot(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """PCA до 2 компонент — только для визуализации."""
    return PCA(n_components=n_components,
               random_state=RANDOM_STATE).fit_transform(X)


def prepare_dataset(path: str = FILE_PATH):
    """
    Полный конвейер подготовки данных.
    """
    raw = load_raw(path)
    n_raw_cols = raw.shape[1]

    filtered = drop_sparse_columns(raw)
    n_kept_cols = filtered.shape[1]

    X_imputed = impute_median(filtered)
    X_scaled = scale_features(X_imputed)
    X_pca = reduce_for_plot(X_scaled)

    info = {
        "n_samples": X_scaled.shape[0],
        "n_features_raw": n_raw_cols,
        "n_features_used": n_kept_cols,
        "n_missing_filled": int(filtered.isna().sum().sum()),
    }

    print("\n--- Подготовка данных ---")
    print(f"Объектов: {info['n_samples']}")
    print(f"Признаков исходно: {info['n_features_raw']}")
    print(f"Признаков после фильтрации: {info['n_features_used']}")
    print(f"Заполнено пропусков медианой: {info['n_missing_filled']}")

    return X_scaled, X_pca, list(filtered.columns), info