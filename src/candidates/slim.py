import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.linear_model import ElasticNet

from .base import CandidateGenerator


def _fit_one_item(r: csc_matrix, j: int, params: dict, max_neighbors: int | None):
    """
    Регрессия столбца j на остальные столбцы:
    Ищем книги i, который лучше всего предсказывают взаимодействие 
    с книгой j.  
    возвращает (индексы книг, веса).
    """
    start, stop = r.indptr[j], r.indptr[j + 1]
    if start == stop:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    y = r[:, j].toarray().ravel()
    # Книгу j нельзя предсказывать по ней самой: зануляем её столбец (структура общая, данные копия)
    data = r.data.copy()
    data[start:stop] = 0.0
    # Массив индексов и границ можно не менять - просто получим явно сохранённые нули
    x = csc_matrix((data, r.indices, r.indptr), shape=r.shape)

    coef = ElasticNet(positive=True, fit_intercept=False, copy_X=False, **params).fit(x, y).coef_
    nonzero = np.flatnonzero(coef > 1e-8)
    # Ограничиваем количество связей (книг i) 
    if max_neighbors is not None and len(nonzero) > max_neighbors:
        nonzero = nonzero[np.argsort(coef[nonzero])[::-1][:max_neighbors]]
    return nonzero, coef[nonzero].astype(np.float32)


class SLIMCandidates(CandidateGenerator):
    """
    SLIM (Sparse Linear Methods): для каждой книги j учим Elastic Net регрессию

    R[:, j] ~= R[:, остальные] @ w_j,   w_j >= 0, w_jj = 0

    Оценка книги для пользователя: R[u] @ W.

    alpha / l1_ratio задают разреженность W: больше alpha — меньше ненулевых весов.
    max_neighbors ограничивает число книг-предикторов у каждой книги.
    """

    name = "slim"
    drop_nonpositive = True

    def __init__(self, alpha: float = 1e-4, l1_ratio: float = 0.5, max_iter: int = 100,
                 tol: float = 1e-4, max_neighbors: int | None = None, n_jobs: int = 1):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_neighbors = max_neighbors
        self.n_jobs = n_jobs

    def _fit(self, matrix: csr_matrix, catalog) -> None:
        r = matrix.tocsc().astype(np.float64)
        n_items = r.shape[1]
        params = dict(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=self.max_iter,
                      tol=self.tol, selection="cyclic")

        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_fit_one_item)(r, j, params, self.max_neighbors) for j in range(n_items)
        )

        rows = np.concatenate([idx for idx, _ in results]) if results else np.empty(0, np.int64)
        cols = np.concatenate([np.full(len(idx), j) for j, (idx, _) in enumerate(results)]) \
            if results else np.empty(0, np.int64)
        vals = np.concatenate([w for _, w in results]) if results else np.empty(0, np.float32)
        self.weights_ = csr_matrix((vals, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.matrix_[users] @ self.weights_).toarray()