"""
Метрики ранжирования для списков рекомендаций (индексы каталога).

Для первого уровня главная метрика — Recall@K при большом K (50-200):
она показывает, какую долю будущих прочтений вообще увидит второй уровень.
Precision/NDCG/MAP при K=10 полезны для сравнения с notebooks/EDA.ipynb.
"""
import numpy as np
from scipy.sparse import csr_matrix

from ..candidates.base import PAD_ITEM, CandidateGenerator

DEFAULT_KS = (10, 50, 100, 200)


def hit_matrix(items: np.ndarray, target: csr_matrix, users: np.ndarray) -> np.ndarray:
    """
    Строит бинарную матрицу попаданий:
    bool (n_users, K): угадана ли книга на позиции r у пользователя users[u].
    """
    # n - количество юзеров, k - количество рекоммендаций
    n, k = items.shape
    valid = items != PAD_ITEM
    # Повторяем индекс пользователя столько раз, сколько для него рекоммендаций
    rows = np.repeat(np.asarray(users), k)
    # Невалидные рекоммендации заполняем 0
    cols = np.where(valid, items, 0).ravel()
    # Смотрим есть ли эти рекоммендации среди валидных книг
    found = np.asarray(target[rows, cols]).reshape(n, k) > 0
    return found & valid


def ranking_metrics(items: np.ndarray, target: csr_matrix, users: np.ndarray,
                     ks=(10,), n_items: int | None = None) -> dict[str, float]:
    """
    Усреднённые по пользователям Precision/Recall/HitRate/NDCG/MAP @K.

    items — (len(users), K_max) рекомендации, target — бинарная матрица релевантных книг.
    Пользователи без релевантных книг пропускаются. AP@K нормируется на min(K, число релевантных).
    """
    # Оставляем только тех пользователей, для которых есть информация в target
    users = np.asarray(users)
    n_relevant = np.diff(target.tocsr().indptr)[users]
    keep = n_relevant > 0
    users, items, n_relevant = users[keep], items[keep], n_relevant[keep]
    # Строим матрицу попаданий
    hits = hit_matrix(items, target, users)
    discount = 1.0 / np.log2(np.arange(2, items.shape[1] + 2))
    ideal_dcg = np.cumsum(discount)
    # Считаем метрики
    metrics: dict[str, float] = {"n_users": float(len(users))}
    for k in ks:
        h = hits[:, :k]
        n_hits = h.sum(axis=1)
        ranks = np.arange(1, h.shape[1] + 1)
        ap = (h * (np.cumsum(h, axis=1) / ranks)).sum(axis=1) / np.minimum(n_relevant, k)
        dcg = (h * discount[:h.shape[1]]).sum(axis=1)

        metrics[f"Precision@{k}"] = float(np.mean(n_hits / k))
        metrics[f"Recall@{k}"] = float(np.mean(n_hits / n_relevant))
        metrics[f"HitRate@{k}"] = float(np.mean(n_hits > 0))
        metrics[f"NDCG@{k}"] = float(np.mean(dcg / ideal_dcg[np.minimum(n_relevant, k) - 1]))
        metrics[f"MAP@{k}"] = float(np.mean(ap))
        if n_items:
            shown = items[:, :k]
            metrics[f"Coverage@{k}"] = len(np.unique(shown[shown != PAD_ITEM])) / n_items
    return metrics


def evaluate_generator(generator: CandidateGenerator, target: csr_matrix,
                        ks=DEFAULT_KS, users: np.ndarray | None = None) -> dict[str, float]:
    """Рекомендации генератора для пользователей с непустым target и метрики по ним."""
    target = target.tocsr()
    if users is None:
        users = np.flatnonzero(np.diff(target.indptr) > 0)
    items, _ = generator.recommend(users, k=max(ks))
    return ranking_metrics(items, target, users, ks, n_items=generator.n_items_)