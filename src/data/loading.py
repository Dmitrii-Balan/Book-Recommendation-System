import logging
from pathlib import Path

import pandas as pd

from .schema import (
    DATE_COL,
    INTERACTION_COLUMNS,
    ITEM_COL,
    ITEM_COLUMNS,
    PROGRESS_COL,
    RATING_COL,
    TEXT_COLUMNS,
    USER_COL,
    YEAR_COL,
)

logger = logging.getLogger(__name__)


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    """
    Проверяем, есть ли необходимые колонки в pd.DataFrame 
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"В {name} нет колонок: {missing}. Есть: {list(df.columns)}")


def load_interactions(path: str | Path) -> pd.DataFrame:
    """
    Загружает interactions.csv и делает предобработку:

    - start_date -> datetime
    - progress -> [0, 100] (пропуск = 0),
    - строки без user_id / item_id / даты удаляются
    """
    raw = pd.read_csv(path)
    _require_columns(raw, INTERACTION_COLUMNS, "interactions")

    df = raw[INTERACTION_COLUMNS].copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[PROGRESS_COL] = pd.to_numeric(df[PROGRESS_COL], errors="coerce").fillna(0).clip(0, 100)

    rating = pd.to_numeric(df[RATING_COL], errors="coerce")
    df[RATING_COL] = rating.where(rating.between(1, 5))

    n_before = len(df)
    df = df.dropna(subset=[USER_COL, ITEM_COL, DATE_COL])
    df = df.astype({USER_COL: "int64", ITEM_COL: "int64"})
    if len(df) < n_before:
        logger.warning("Удалено строк с пропусками в user/item/date: %d", n_before - len(df))

    logger.info("interactions: %d строк, %d пользователей, %d книг",
                len(df), df[USER_COL].nunique(), df[ITEM_COL].nunique())
    return df.reset_index(drop=True)


def load_items(path: str | Path) -> pd.DataFrame:
    """
    Загружает items.csv и делает предобработку: 

    - колонка `id` переименовывается в `item_id`;
    - дубли по item_id: остаётся первая строка;
    - пропуски в title/genres/authors -> "", 
    - year -> число или NaN.
    """
    raw = pd.read_csv(path)
    if ITEM_COL not in raw.columns and "id" in raw.columns:
        raw = raw.rename(columns={"id": ITEM_COL})
    _require_columns(raw, ITEM_COLUMNS, "items")

    items = raw.dropna(subset=[ITEM_COL]).drop_duplicates(ITEM_COL, keep="first").copy()
    items[ITEM_COL] = items[ITEM_COL].astype("int64")
    for col in TEXT_COLUMNS:
        items[col] = items[col].fillna("").astype(str)
    items[YEAR_COL] = pd.to_numeric(items[YEAR_COL], errors="coerce")

    logger.info("items: %d книг", len(items))
    return items.reset_index(drop=True)