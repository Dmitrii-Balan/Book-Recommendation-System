"""
Матрица пользователь × книга для CF-моделей (ALS, SLIM, item-kNN, U2U).

Строки — user_idx, столбцы — item_idx из Catalog, форма всегда (n_users, n_items)
каталога, поэтому матрицы train / val / test сопоставимы между собой.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .catalog import Catalog
from .schema import ITEM_IDX_COL, PROGRESS_COL, RATING_COL, USER_IDX_COL

WEIGHTINGS = ("binary", "progress", "progress_rating")


def interaction_weights(
        progress: pd.Series, 
        rating: pd.Series, 
        weighting: str = "progress_rating", 
        rating_bonus: float = 1, 
        min_weight: float = 0.05
    ) -> np.ndarray:
    """
    Преобразуем взаимодействия в веса - уверенность в том, что книга понравилась.

    binary           — 1.0 для всех;
    progress         — progress / 100, не меньше min_weight;
    progress_rating  — progress / 100 + rating_bonus * (rating - 1) / 4
                       (пропуск рейтинга = 0 бонуса).
    """
    if weighting == "binary":
        return np.ones(len(progress), dtype=np.float32)

    base = progress.to_numpy(dtype=np.float32) / 100.0
    if weighting == "progress":
        weights = base
    elif weighting == "progress_rating":
        rating_norm = ((rating.to_numpy(dtype=np.float32) - 1.0) / 4.0)
        rating_norm = np.nan_to_num(rating_norm, nan=0.0)
        weights = base + rating_bonus * rating_norm
    else:
        raise ValueError(f"weighting должен быть одним из {WEIGHTINGS}, а не {weighting!r}")

    return np.clip(weights, min_weight, None).astype(np.float32)


def build_interaction_matrix(
        df: pd.DataFrame, 
        catalog: Catalog, 
        weighting: str = "binary", 
        rating_bonus: float = 0.5, 
        min_weight: float = 0.05
    ) -> csr_matrix:
    """
    df (user_id, item_id, progress, rating]) -> csr (n_users, n_items).

    Пары с пользователем или книгой вне каталога отбрасываются.
    Повторы пары схлопываются: progress — максимум, rating — среднее.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting должен быть одним из {WEIGHTINGS}, а не {weighting!r}")

    enc = catalog.encode(df)
    keys = [USER_IDX_COL, ITEM_IDX_COL]

    if weighting == "binary":
        enc = enc.drop_duplicates(keys)
    else:
        if enc.duplicated(keys).any():
            enc = enc.groupby(keys, as_index=False).agg(
                {PROGRESS_COL: "max", RATING_COL: "mean"}
            )

    weights = interaction_weights(
        enc[PROGRESS_COL], enc[RATING_COL], weighting, rating_bonus, min_weight
    )

    return csr_matrix(
        (weights, (enc[USER_IDX_COL].to_numpy(), enc[ITEM_IDX_COL].to_numpy())),
        shape=(catalog.n_users, catalog.n_items),
    )