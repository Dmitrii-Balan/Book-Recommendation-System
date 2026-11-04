"""
Интерфейс моделей первого уровня (генераторов кандидатов).

Задача первого уровня — из всего каталога быстро отобрать для каждого пользователя
~100-200 книг так, чтобы среди них как можно больше оказалось тех, что он прочитает
(метрика — Recall@K при большом K). Точный порядок внутри списка — дело второго уровня.

Все генераторы:
  * работают с индексами каталога (user_idx, item_idx), а не с исходными id;
  * учатся на матрице пользователь × книга формы (catalog.n_users, catalog.n_items),
    её строит src.data.matrices.build_interaction_matrix;
  * по умолчанию не рекомендуют книги, которые уже есть в матрице, на которой они учились.
    Поэтому перед тестом генераторы дообучают на train+val.

Использование:
    gen = ItemKNNCandidates().fit(train_matrix, catalog)
    items, scores = gen.recommend(user_idx, k=200)   # обе матрицы (n_users, k)
    # если кандидатов меньше k, хвост заполнен PAD_ITEM = -1
"""
from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import csr_matrix

from .utils import top_k

PAD_ITEM = -1


class CandidateGenerator(ABC):
    name: str = "base"
    # True: оценки <= 0 означают «нет сигнала» и не должны попадать в кандидаты
    # (kNN, SLIM, content). Для ALS и популярности оценка 0 — обычное значение.
    drop_nonpositive: bool = False

    def fit(self, matrix: csr_matrix, catalog=None) -> "CandidateGenerator":
        self.matrix_ = csr_matrix(matrix, dtype=np.float32)
        self.n_users_, self.n_items_ = self.matrix_.shape
        self._fit(self.matrix_, catalog)
        return self

    @abstractmethod
    def _fit(self, matrix: csr_matrix, catalog) -> None:
        """Обучение на матрице пользователь × книга."""

    @abstractmethod
    def score(self, users: np.ndarray) -> np.ndarray:
        """Оценки всех книг для users: dense float32 (len(users), n_items)"""

    def recommend(self, users, k: int = 100, filter_seen: bool = True, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """Топ-k книг для каждого пользователя из users (индексы каталога)."""
        users = np.asarray(users, dtype=np.int64)
        k_eff = min(k, self.n_items_)
        items = np.full((len(users), k), PAD_ITEM, dtype=np.int32)
        scores = np.full((len(users), k), -np.inf, dtype=np.float32)

        for start in range(0, len(users), batch_size):
            batch = users[start:start + batch_size]
            s = np.array(self.score(batch), dtype=np.float32)  # копия: ниже правим на месте
            if filter_seen:
                seen = self.matrix_[batch].tocoo()
                s[seen.row, seen.col] = -np.inf
            if self.drop_nonpositive:
                s[s <= 0] = -np.inf
            top_idx, top_scores = top_k(s, k_eff)
            rows = slice(start, start + len(batch))
            items[rows, :k_eff] = top_idx
            scores[rows, :k_eff] = top_scores

        items[scores < 0] = PAD_ITEM
        return items, scores