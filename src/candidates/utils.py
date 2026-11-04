"""Общие функции для генераторов кандидатов."""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Индексы и значения k лучших элементов в каждой строке, по убыванию.

    scores — (n_rows, n_cols); k должно быть не больше n_cols.
    """
    if k < scores.shape[1]:
        idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    else:
        idx = np.tile(np.arange(scores.shape[1]), (scores.shape[0], 1))
    vals = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-vals, axis=1, kind="stable")
    return (
        np.take_along_axis(idx, order, axis=1), 
        np.take_along_axis(vals, order, axis=1)
    )


def keep_top_k(dense: np.ndarray, k: int) -> np.ndarray:
    """В каждой строке оставляет k наибольших значений, остальные обнуляет."""
    if k >= dense.shape[1]:
        return dense
    idx = np.argpartition(-dense, k - 1, axis=1)[:, :k]
    out = np.zeros_like(dense)
    np.put_along_axis(out, idx, np.take_along_axis(dense, idx, axis=1), axis=1)
    return out


def bm25_weight(item_users: csr_matrix, k1: float = 100.0, b: float = 0.8) -> csr_matrix:
    """
    BM25-взвешивание матрицы книга × пользователь (как в implicit.nearest_neighbours).

    Снижает вклад пользователей, которые прочитали очень много книг, и популярных книг.
    """
    x = coo_matrix(item_users, dtype=np.float32)
    n_cols = float(x.shape[1])
    idf = np.log(n_cols) - np.log1p(np.bincount(x.col, minlength=x.shape[1]))
    row_sums = np.asarray(x.sum(axis=1)).ravel()
    length_norm = (1.0 - b) + b * row_sums / max(row_sums.mean(), 1e-9)
    x.data = x.data * (k1 + 1.0) / (k1 * length_norm[x.row] + x.data) * idf[x.col]
    return x.tocsr()