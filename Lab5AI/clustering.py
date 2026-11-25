from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture


@dataclass
class ClusteringResult:
    """Результат работы одного алгоритма с заданными параметрами."""
    algorithm: str
    params: Dict
    labels: np.ndarray
    n_clusters: int
    n_noise: int


def _count_clusters(labels: np.ndarray) -> int:
    unique = set(labels)
    unique.discard(-1)
    return len(unique)


def run_kmeans(X: np.ndarray, k: int,
               random_state: int = 42) -> ClusteringResult:
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return ClusteringResult(
        algorithm="KMeans",
        params={"k": k},
        labels=labels,
        n_clusters=k,
        n_noise=0,
    )


def run_dbscan(X: np.ndarray, eps: float,
               min_samples: int) -> ClusteringResult:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return ClusteringResult(
        algorithm="DBSCAN",
        params={"eps": eps, "min_samples": min_samples},
        labels=labels,
        n_clusters=_count_clusters(labels),
        n_noise=int((labels == -1).sum()),
    )


def run_agglomerative(X: np.ndarray, k: int,
                      linkage: str = "ward") -> ClusteringResult:
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = model.fit_predict(X)
    return ClusteringResult(
        algorithm=f"Agglomerative({linkage})",
        params={"k": k, "linkage": linkage},
        labels=labels,
        n_clusters=k,
        n_noise=0,
    )


def run_gmm(X: np.ndarray, k: int,
            random_state: int = 42) -> ClusteringResult:
    model = GaussianMixture(n_components=k, random_state=random_state)
    labels = model.fit_predict(X)
    return ClusteringResult(
        algorithm="GMM",
        params={"k": k},
        labels=labels,
        n_clusters=k,
        n_noise=0,
    )