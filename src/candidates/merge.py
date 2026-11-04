"""
Объединение кандидатов от нескольких генераторов в один пул для второго уровня.

Пул — таблица (user_idx, item_idx) с признаками для ранкера: у каждого источника
оценка и позиция в его выдаче (NaN, если источник книгу не предложил), число
источников, предложивших пару, и сводный балл RRF.
"""
from functools import reduce

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ..data.schema import ITEM_IDX_COL, USER_IDX_COL
from .base import PAD_ITEM

RRF_COL = "rrf"
N_SOURCES_COL = "n_sources"


def _long_format(name: str, users: np.ndarray, items: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    n, k = items.shape
    frame = pd.DataFrame({
        USER_IDX_COL: np.repeat(users, k),
        ITEM_IDX_COL: items.ravel(),
        f"{name}_score": scores.ravel(),
        f"{name}_rank": np.tile(np.arange(1, k + 1, dtype=np.float32), n),
    })
    return frame[frame[ITEM_IDX_COL] != PAD_ITEM]


def merge_candidates(
    users,
    recommendations: dict[str, tuple[np.ndarray, np.ndarray]],
    max_per_user: int | None = None,
    rrf_k: int = 60,
) -> pd.DataFrame:
    """
    users — индексы пользователей (строки массивов recommendations);
    recommendations — {имя генератора: (items, scores)} из CandidateGenerator.recommend.

    RRF (reciprocal rank fusion) = sum 1 / (rrf_k + rank) по источникам; не зависит от
    масштаба оценок моделей. max_per_user оставляет столько лучших по RRF пар на пользователя.
    """
    users = np.asarray(users, dtype=np.int64)
    frames = [_long_format(name, users, items, scores) for name, (items, scores) in recommendations.items()]
    pool = reduce(lambda a, b: a.merge(b, on=[USER_IDX_COL, ITEM_IDX_COL], how="outer"), frames)

    rank_cols = [f"{name}_rank" for name in recommendations]
    pool[N_SOURCES_COL] = pool[rank_cols].notna().sum(axis=1).astype(np.int8)
    pool[RRF_COL] = (1.0 / (rrf_k + pool[rank_cols])).sum(axis=1)  # NaN-ранги не дают вклада

    pool = pool.sort_values([USER_IDX_COL, RRF_COL], ascending=[True, False], kind="stable")
    if max_per_user is not None:
        pool = pool.groupby(USER_IDX_COL, sort=False).head(max_per_user)
    return pool.reset_index(drop=True)


def pool_recall(pool: pd.DataFrame, target: csr_matrix, users=None) -> float:
    """
    Доля релевантных пар (target), попавших в пул, — потолок Recall для второго уровня.

    users — по каким пользователям считать (по умолчанию все, у кого есть target).
    """
    target = target.tocsr()
    users = np.flatnonzero(np.diff(target.indptr) > 0) if users is None else np.asarray(users)
    sub = target[users].tocoo()
    wanted = set(zip(users[sub.row].tolist(), sub.col.tolist()))
    if not wanted:
        return float("nan")
    in_pool = pool[pool[USER_IDX_COL].isin(users)]
    found = wanted & set(zip(in_pool[USER_IDX_COL].tolist(), in_pool[ITEM_IDX_COL].tolist()))
    return len(found) / len(wanted)
