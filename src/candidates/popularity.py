import numpy as np

from .base import CandidateGenerator


class PopularityCandidates(CandidateGenerator):
    """
    Глобальная популярность: сумма весов взаимодействий по книге.

    Одинаковый список для всех, но это обязательный бейзлайн: любая персональная
    модель должна его обыгрывать. Как источник кандидатов закрывает пользователей
    с короткой историей, для которых остальные модели ничего не находят.
    """

    name = "popular"

    def _fit(self, matrix, catalog) -> None:
        self.popularity_ = np.asarray(matrix.sum(axis=0), dtype=np.float32).ravel()

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.tile(self.popularity_, (len(users), 1))
