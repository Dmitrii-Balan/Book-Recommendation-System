"""
Набор генераторов по умолчанию: имя -> (генератор, как взвешивать матрицу).

CF-модели, которым важен сам факт чтения (kNN, SLIM), учатся на бинарной матрице;
ALS и content используют силу сигнала (прогресс чтения и оценку).
Гиперпараметры — разумные стартовые значения, их стоит подбирать на val.
"""
from .als import ALSCandidates
from .base import CandidateGenerator
from .content import ContentCandidates
from .knn import ItemKNNCandidates, UserKNNCandidates
from .popularity import PopularityCandidates
from .slim import SLIMCandidates


def build_generators(names: list[str] | None = None) -> dict[str, tuple[CandidateGenerator, str]]:
    all_generators = {
        "popular": (PopularityCandidates(), "binary"),
        "item_knn": (ItemKNNCandidates(k_neighbors=100), "binary"),
        "item_knn_bm25": (ItemKNNCandidates(k_neighbors=100, bm25=True), "binary"),
        "user_knn": (UserKNNCandidates(k_neighbors=50), "binary"),
        "slim": (SLIMCandidates(alpha=1e-3, l1_ratio=0.5, max_neighbors=200, n_jobs=2), "binary"),
        "als": (ALSCandidates(factors=100, regularization=0.05, alpha=10, iterations=70), "progress_rating"),
        "content": (ContentCandidates(), "progress_rating"),
    }
    if names is None:
        return all_generators
    unknown = set(names) - set(all_generators)
    if unknown:
        raise ValueError(f"Неизвестные генераторы: {sorted(unknown)}. Доступны: {sorted(all_generators)}")
    return {name: all_generators[name] for name in names}
