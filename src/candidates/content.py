import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .base import CandidateGenerator


def _split_commas(text: str) -> list[str]:
    """«Фэнтези, Приключения» -> ["фэнтези", "приключения"]."""
    return [part.strip().lower() for part in text.split(",") if part.strip()]


class ContentCandidates(CandidateGenerator):
    """
    Content-based: профиль пользователя — взвешенная сумма векторов его книг,
    оценка — косинусное сходство профиля с каждой книгой.

    Вектор книги — склейка TF-IDF по названию, жанрам и авторам (целые
    значения через запятую), каждый блок умножается на свой вес.

    По метрикам слабее CF, но не зависит от того, сколько у книги читателей, и
    добавляет разнообразие в общий пул кандидатов.
    Нужен catalog: строка catalog.meta[i] описывает книгу с индексом i.
    """

    name = "content"
    drop_nonpositive = True

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or {"title": 0.5, "genres": 1.0, "authors": 1.0}

    def _vectorizers(self) -> dict[str, TfidfVectorizer]:
        return {
            "title": TfidfVectorizer(sublinear_tf=True, token_pattern=r"(?u)\b\w\w+\b"),
            "genres": TfidfVectorizer(tokenizer=_split_commas, token_pattern=None, lowercase=False),
            "authors": TfidfVectorizer(tokenizer=_split_commas, token_pattern=None, lowercase=False),
        }

    def _fit(self, matrix, catalog) -> None:
        if catalog is None:
            raise ValueError("ContentCandidates нужен catalog с метаданными книг")
        meta = catalog.meta
        if len(meta) != matrix.shape[1]:
            raise ValueError("catalog.meta не согласован с числом книг в матрице")
    
        blocks = []
        for column, vectorizer in self._vectorizers().items():
            weight = self.weights.get(column, 0.0)
            if weight <= 0 or column not in meta.columns:
                continue
            try:
                block = vectorizer.fit_transform(meta[column].fillna("").astype(str))
            except ValueError:   # во всех книгах поле пустое -> пустой словарь
                continue
            blocks.append(normalize(block) * weight)
        if not blocks:
            raise ValueError("Не удалось построить ни одного текстового признака книг")

        self.item_vectors_ = normalize(hstack(blocks, format="csr")).astype(np.float32)

    def score(self, users: np.ndarray) -> np.ndarray:
        profile = normalize(self.matrix_[users] @ self.item_vectors_)
        return (profile @ self.item_vectors_.T).toarray()
