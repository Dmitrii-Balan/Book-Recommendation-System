import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.preprocessing import normalize

from .base import CandidateGenerator
from .utils import bm25_weight, keep_top_k


class ItemKNNCandidates(CandidateGenerator):
    """
    Item-based CF: книга получает оценку, пропорциональную сходству с книгами из истории.

    score(u, j) = sum_{i in history(u)} sim(i, j),  i из k ближайших соседей книги j

    sim — косинус по столбцам матрицы; shrink в знаменателе занижает сходство книг
    с малым числом общих читателей (cos = co / (|i| * |j| + shrink)).
    bm25=True взвешивает матрицу BM25 перед подсчётом сходства.
    """

    name = "item_knn"
    drop_nonpositive = True

    def __init__(self, k_neighbors: int = 100, shrink: float = 0.0,
                 bm25: bool = False, chunk_size: int = 1024):
        self.k_neighbors = k_neighbors
        self.shrink = shrink
        self.bm25 = bm25
        self.chunk_size = chunk_size

    def _fit(self, matrix: csr_matrix, catalog) -> None:
        x = bm25_weight(matrix.T.tocsr()).T.tocsr() if self.bm25 else matrix
        xt = x.T.tocsr()                                        # книги × пользователи
        norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=0)).ravel())
        n_items = x.shape[1]

        blocks = []
        for start in range(0, n_items, self.chunk_size):
            stop = min(start + self.chunk_size, n_items)
            co = (xt[start:stop] @ x).toarray()    # co[i][j] = общее число читателей, прочитавших обе книги
            denom = norms[start:stop, None] * norms[None, :] + self.shrink
            sim = co / np.maximum(denom, 1e-9)
            sim[np.arange(stop - start), np.arange(start, stop)] = 0.0   # без самой себя
            blocks.append(csr_matrix(keep_top_k(sim, self.k_neighbors)))

        # строка j в top-k — соседи книги j; W[i, j] = sim(i, j), поэтому транспонируем
        self.sim_ = vstack(blocks, format="csr").T.tocsr().astype(np.float32)

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.matrix_[users] @ self.sim_).toarray()


class UserKNNCandidates(CandidateGenerator):
    """
    User-based CF: рекомендуем то, что читали k самых похожих пользователей.

        score(u, j) = sum_{v in neighbors(u)} sim(u, v) * R[v, j]

    sim — косинус между строками матрицы. Похожих пользователей ищем по train-матрице,
    поэтому новый пользователь без истории кандидатов не получит.
    """

    name = "user_knn"
    drop_nonpositive = True

    def __init__(self, k_neighbors: int = 50):
        self.k_neighbors = k_neighbors

    def _fit(self, matrix: csr_matrix, catalog) -> None:
        self.normalized_ = normalize(matrix, norm="l2", axis=1).tocsr()

    def score(self, users: np.ndarray) -> np.ndarray:
        sim = (self.normalized_[users] @ self.normalized_.T).toarray()   # (batch, n_users)
        sim[np.arange(len(users)), users] = 0.0                          # без самого себя
        neighbors = csr_matrix(keep_top_k(sim, self.k_neighbors))
        return (neighbors @ self.matrix_).toarray()
