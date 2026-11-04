import numpy as np

from .base import CandidateGenerator
from implicit.als import AlternatingLeastSquares


def _to_numpy(factors) -> np.ndarray:
    """implicit на GPU отдаёт собственный тип матрицы; на CPU это уже numpy."""
    if hasattr(factors, "to_numpy"):
        factors = factors.to_numpy()
    return np.asarray(factors, dtype=np.float32)


class ALSCandidates(CandidateGenerator):
    """
    Implicit ALS (библиотека implicit): латентные векторы пользователей и книг,
    оценка — скалярное произведение.

    Значения матрицы — уверенность в том, что книга понравилась (обычно веса
    "progress_rating" из build_interaction_matrix), implicit умножает их на alpha.
    Оценки считаем сами по факторам, чтобы пакетно обрабатывать пользователей и
    не зависеть от версии API implicit.recommend.
    """

    name = "als"

    def __init__(self, factors: int = 100, regularization: float = 0.05, alpha: float = 10.0,
                 iterations: int = 70, random_state: int = 42, num_threads: int = 0):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.random_state = random_state
        self.num_threads = num_threads

    def _fit(self, matrix, catalog) -> None:

        model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            alpha=self.alpha,
            iterations=self.iterations,
            random_state=self.random_state,
            num_threads=self.num_threads,
            use_gpu=False,
        )
        model.fit(matrix, show_progress=False)
        self.user_factors_ = _to_numpy(model.user_factors)
        self.item_factors_ = _to_numpy(model.item_factors)

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.user_factors_[users] @ self.item_factors_.T