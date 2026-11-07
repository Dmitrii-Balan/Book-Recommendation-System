"""
Разбиение на train / val / test по времени.

Две схемы:

  * "global_time" — общая дата отсечки для всех. Так в проде:
    знаем всё до «сегодня», предсказываем будущее. Пользователи, которых нет
    в train, при оценке пропускаются (холодный старт — отдельная задача).

  * "per_user" — у каждого пользователя последние доли его истории уходят
    в val/test. Удобно, когда мало данных, но модель «видит» будущее других
    пользователей.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import DATE_COL, ITEM_COL, USER_COL

_EPS = 1e-9  # защита от 6.999999 при floor(n * 0.7)


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    meta: dict = field(default_factory=dict)

    @property
    def train_val(self) -> pd.DataFrame:
        return pd.concat([self.train, self.val], ignore_index=True)

    def history(self, part: str) -> pd.DataFrame:
        """
        Что модель вправе знать при оценке на `part`: 
        - для val — train
        - для test — train+val
        """
        if part == "val":
            return self.train
        if part == "test":
            return self.train_val
        raise ValueError("history() определена только для 'val' и 'test'")

    def target(self, part: str) -> pd.DataFrame:
        """
        Что нужно угадать: 
        val/test без пар, уже встречавшихся в истории.
        """
        if part not in ("val", "test"):
            raise ValueError("target() определена только для 'val' и 'test'")
        frame = self.val if part == "val" else self.test
        return _drop_seen(frame, self.history(part))

    def summary(self) -> pd.DataFrame:
        """
        Для проверки временного разбиения 
        """
        rows = []
        for name in ("train", "val", "test"):
            part = getattr(self, name)
            rows.append(
                {
                    "part": name,
                    "rows": len(part),
                    "users": part[USER_COL].nunique(),
                    "items": part[ITEM_COL].nunique(),
                    "date_min": part[DATE_COL].min() if len(part) else pd.NaT,
                    "date_max": part[DATE_COL].max() if len(part) else pd.NaT,
                }
            )
        return pd.DataFrame(rows)

def _drop_seen(target: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Убирает из target пары (user, item), которые уже есть в history."""
    seen = history[[USER_COL, ITEM_COL]].drop_duplicates().assign(_seen=True)
    merged = target.merge(seen, how="left", on=[USER_COL, ITEM_COL])
    return merged[merged["_seen"].isna()].drop(columns="_seen").reset_index(drop=True)

def _sorted(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return df.sort_values(keys, kind="stable").reset_index(drop=True)

def split_global_time(df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1) -> Splits:
    """Общая дата отсечки"""
    data = _sorted(df, [DATE_COL, USER_COL, ITEM_COL])
    n = len(data)
    dates = data[DATE_COL]

    test_start = dates.iloc[min(int(n * (1 - test_size)), n - 1)]
    val_start = dates.iloc[min(int(n * (1 - test_size - val_size)), n - 1)]
    val_start, test_start = pd.Timestamp(val_start), pd.Timestamp(test_start)

    if val_size > 0 and val_start >= test_start:
        raise ValueError(
            f"val_start ({val_start.date()}) должен быть раньше test_start ({test_start.date()}). "
        )

    if val_size == 0:
        val_start = test_start

    train = data[dates < val_start].reset_index(drop=True)
    val = data[(dates >= val_start) & (dates < test_start)].reset_index(drop=True)
    test = data[dates >= test_start].reset_index(drop=True)

    meta = {
        "scheme": "global_time",
        "val_start": val_start.isoformat(),
        "test_start": test_start.isoformat(),
        "real_fracs": {
            "train": len(train) / n,
            "val": len(val) / n,
            "test": len(test) / n,
        }
    }
    return Splits(train, val, test, meta)


def split_per_user_time(df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.2) -> Splits:
    """У каждого пользователя своя дата (делим по объёму прочитанных книг)"""
    data = _sorted(df, [USER_COL, DATE_COL, ITEM_COL])

    reading_order = data.groupby(USER_COL).cumcount().to_numpy()
    group_size = data.groupby(USER_COL)[USER_COL].transform("size").to_numpy()

    train_end = np.floor(group_size * (1 - val_size - test_size) + _EPS).astype(int)
    val_end = np.floor(group_size * (1 - test_size) + _EPS).astype(int)

    train = data[reading_order < train_end].reset_index(drop=True)
    val = data[(reading_order >= train_end) & (reading_order < val_end)].reset_index(drop=True)
    test = data[reading_order >= val_end].reset_index(drop=True)

    n = len(data)
    meta = {
        "scheme": "per_user",
        "real_fracs": {
            "train": len(train) / n,
            "val": len(val) / n,
            "test": len(test) / n,
        },
    }
    return Splits(train, val, test, meta)


def make_splits(df: pd.DataFrame, scheme: str = "global_time", **kwargs) -> Splits:
    if scheme == "global_time":
        return split_global_time(df, **kwargs)
    if scheme == "per_user":
        return split_per_user_time(df, **kwargs)
    raise ValueError(f"Неизвестная схема сплита: {scheme!r}")