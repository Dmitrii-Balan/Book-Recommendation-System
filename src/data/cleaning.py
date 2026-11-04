"""
Очистка взаимодействий. Входной датафрейм не меняется.

Порядок в пайплайне (см. prepare.py):
    deduplicate_interactions -> filter_positive -> filter_users   (до сплита)
    iterative_core_filter                                         (только на train!)
"""
import logging

import pandas as pd

from .schema import DATE_COL, ITEM_COL, PROGRESS_COL, RATING_COL, USER_COL

logger = logging.getLogger(__name__)


def clear_deduplicate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Одна строка на пару (user, item).

    start_date — самая ранняя, progress — максимальный, rating — средний.
    После этого пара (user, item) встречается ровно один раз, поэтому
    «утечка по паре» между train/val/test невозможна по построению.
    """
    if not df.duplicated([USER_COL, ITEM_COL]).any():
        return df.copy()

    out = df.groupby([USER_COL, ITEM_COL], as_index=False, sort=False).agg(
        {PROGRESS_COL: "max", RATING_COL: "mean", DATE_COL: "min"}
    )
    logger.info("Удаление повторных взаимодействий: %d -> %d строк", len(df), len(out))
    return out[list(df.columns)]


def filter_positive(df: pd.DataFrame, min_rating: float = 4.0,
                    min_progress: float = 70, drop_below_rating: float | None = 1.5) -> pd.DataFrame:
    """
    Оставляет «положительные» взаимодействия: rating >= min_rating ИЛИ progress >= min_progress.

    drop_below_rating — если задан, выбрасывает книги с оценкой ниже порога,
    даже когда они дочитаны (прочёл до конца и поставил 1 — это не positive).
    """
    mask = (df[RATING_COL] >= min_rating) | (df[PROGRESS_COL] >= min_progress)
    if drop_below_rating is not None:
        mask &= ~(df[RATING_COL] < drop_below_rating)  # NaN < x == False, значит NaN остаются
    out = df[mask].copy()
    logger.info("Положительных взаимодействий: %d из %d", len(out), len(df))
    return out


def filter_users(df: pd.DataFrame, min_items: int) -> pd.DataFrame:
    """Оставляет пользователей, у которых не меньше min_items разных книг."""
    counts = df.groupby(USER_COL)[ITEM_COL].nunique()
    keep = counts[counts >= min_items].index
    return df[df[USER_COL].isin(keep)].copy()


def filter_items(df: pd.DataFrame, min_users: int) -> pd.DataFrame:
    """Оставляет книги, которые прочитали не меньше min_users разных пользователей."""
    counts = df.groupby(ITEM_COL)[USER_COL].nunique()
    keep = counts[counts >= min_users].index
    return df[df[ITEM_COL].isin(keep)].copy()


def iterative_core_filter(df: pd.DataFrame, min_item_users: int = 20,
                          min_user_items: int = 10, max_iter: int = 100) -> pd.DataFrame:
    """
    k-core: повторяет filter_items/filter_users, пока набор строк не перестанет меняться.

    Вызывать только на train, поскольку порог по числу читателей, посчитанный на всём
    периоде, использует будущую популярность книг.
    """
    out = df
    for _ in range(max_iter):
        n_before = len(out)
        out = filter_users(filter_items(out, min_item_users), min_user_items)
        if len(out) == n_before:
            break
    else:
        logger.warning("iterative_core_filter не сошёлся за %d итераций", max_iter)
    return out.reset_index(drop=True)