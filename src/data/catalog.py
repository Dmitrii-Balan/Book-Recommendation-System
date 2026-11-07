"""
Catalog — единый словарь пользователей и книг для всех моделей.

Каталог строится только по train. Все генераторы кандидатов и метрики работают
с индексами 0..N-1 из этого каталога, поэтому модель физически не может
порекомендовать книгу, которой нет в каталоге оценки.
"""
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import ITEM_COL, ITEM_IDX_COL, TEXT_COLUMNS, USER_COL, USER_IDX_COL

logger = logging.getLogger(__name__)


class IdMap:
    """Двунаправленное отображение «исходный id <-> индекс 0..N-1»"""

    def __init__(self, ids):
        self.ids = np.asarray(pd.unique(ids))
        self._index = pd.Index(self.ids)

    def __len__(self):
        return len(self.ids)

    def encode(self, ids) -> np.ndarray:
        """id -> индекс; для неизвестных id возвращает -1"""
        return self._index.get_indexer(np.asarray(ids))

    def decode(self, idx) -> np.ndarray:
        """индекс -> id"""
        return self.ids[np.asarray(idx)]


@dataclass
class Catalog:
    """Строим единую нумерацию книг и пользователей"""
    users: IdMap
    items: IdMap
    meta: pd.DataFrame  # строка i описывает книгу с индексом i (item_id, title, ...)

    @classmethod
    def from_train(cls, train: pd.DataFrame, items_meta: pd.DataFrame) -> "Catalog":
        """
        Альтернативный конструктор класса Catalog. 
        train - взаимодействия пользователей с книгами
        items_meta - информация о каждой книге
        """
        # Находим все уникальные киги и пользователей
        user_ids = np.sort(train[USER_COL].unique())
        item_ids = np.sort(train[ITEM_COL].unique())
        # Удаляем информацию о книгах, для которых нет взаимодействий 
        meta = pd.DataFrame({ITEM_COL: item_ids}).merge(
            items_meta.drop_duplicates(ITEM_COL), on=ITEM_COL, how="left"
        )
        # Проверяем о скольки книгах у нас отсутствует информация в meta
        if "title" in meta.columns:
            n_missing = int(meta["title"].isna().sum())
            if n_missing:
                logger.warning("Книг из train без метаданных в items.csv: %d", n_missing)
        # Обрабатываем пропуски
        for col in TEXT_COLUMNS:
            if col in meta.columns:
                meta[col] = meta[col].fillna("")
        # Считаем количество читателей книг
        readers = train.groupby(ITEM_COL)[USER_COL].nunique()
        meta["n_readers"] = meta[ITEM_COL].map(readers).astype("int64")
        # Возвращаем готовый Catalog 
        return cls(IdMap(user_ids), IdMap(item_ids), meta)

    @property
    def users_len(self):
        return len(self.users)
    
    @property
    def items_len(self):
        return len(self.items)

    def filter_known(self, df: pd.DataFrame) -> pd.DataFrame:
        """Оставляет взаимодействия, где и пользователь, и книга есть в каталоге."""
        known = df[USER_COL].isin(self.users._index) & df[ITEM_COL].isin(self.items._index)
        return df[known].reset_index(drop=True)

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Переводит user_id и item_id в сжатые индексы.

        Возвращает копию df с новыми столбцами user_idx и item_idx (исходные
        user_id и item_id сохраняются). Строки, где пользователь или книга
        отсутствуют в каталоге, удаляются.
        """
        out = df.copy()
        out[USER_IDX_COL] = self.users.encode(out[USER_COL])
        out[ITEM_IDX_COL] = self.items.encode(out[ITEM_COL])
        known = (out[USER_IDX_COL] >= 0) & (out[ITEM_IDX_COL] >= 0)
        return out[known].reset_index(drop=True)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "user_ids.npy", self.users.ids)
        np.save(directory / "item_ids.npy", self.items.ids)
        self.meta.to_csv(directory / "items_meta.csv", index=False)

    @classmethod
    def load(cls, directory: str | Path) -> "Catalog":
        directory = Path(directory)
        users = IdMap(np.load(directory / "user_ids.npy"))
        items = IdMap(np.load(directory / "item_ids.npy"))
        meta = pd.read_csv(directory / "items_meta.csv")
        for col in TEXT_COLUMNS:
            if col in meta.columns:
                meta[col] = meta[col].fillna("").astype(str)
        if not np.array_equal(meta[ITEM_COL].to_numpy(), items.ids):
            raise ValueError("items_meta.csv не согласован с item_ids.npy")
        return cls(users, items, meta)