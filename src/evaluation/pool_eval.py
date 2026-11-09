"""
Оценка разных способов упорядочить один и тот же пул кандидатов (второй уровень против бейзлайнов).

Один пул — несколько «способов ранжирования»: оценка ранкера, порядок RRF, порядок любого из
генераторов первого уровня. Все они оцениваются теми же метриками (src.evaluation.metrics),
на тех же пользователях и на той же цели, поэтому сравнение честное.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ..candidates.base import PAD_ITEM
from ..data.schema import ITEM_IDX_COL, USER_IDX_COL
from .metrics import ranking_metrics


def topk_from_scores(pool: pd.DataFrame, scores, users: np.ndarray, k: int) -> np.ndarray:
    """
    Топ-k книг на пользователя по оценке пары (чем выше, тем лучше).

    pool — пул (user_idx, item_idx); scores — оценка каждой строки; строки с NaN/-inf игнорируются.
    Возвращает (len(users), k) с PAD_ITEM в хвосте, если кандидатов меньше k.
    """
    frame = pd.DataFrame({
        "u": pool[USER_IDX_COL].to_numpy(), "i": pool[ITEM_IDX_COL].to_numpy(),
        "s": np.asarray(scores, dtype=np.float64),
    })
    frame = frame[np.isfinite(frame["s"])]
    frame = frame.sort_values(["u", "s"], ascending=[True, False], kind="stable")
    frame["pos"] = frame.groupby("u", sort=False).cumcount()
    frame = frame[frame["pos"] < k]

    items = np.full((len(users), k), PAD_ITEM, dtype=np.int32)
    rows = pd.Index(users).get_indexer(frame["u"].to_numpy())
    ok = rows >= 0
    items[rows[ok], frame["pos"].to_numpy()[ok]] = frame["i"].to_numpy()[ok]
    return items


def evaluate_methods(pool: pd.DataFrame, methods: dict[str, np.ndarray], target: csr_matrix,
                     users: np.ndarray, ks=(10, 20), n_items: int | None = None) -> pd.DataFrame:
    """Таблица метрик: строка — способ ранжирования, колонки — Precision/Recall/HitRate/NDCG/MAP/Coverage @K."""
    rows = []
    for name, scores in methods.items():
        items = topk_from_scores(pool, scores, users, max(ks))
        metrics = ranking_metrics(items, target, users, ks=tuple(ks), n_items=n_items)
        metrics.pop("n_users", None)
        rows.append({"method": name, **metrics})
    return pd.DataFrame(rows)