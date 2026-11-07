"""
Сборка данных целиком: сырые csv -> Splits + Catalog.

Порядок шагов (важен для отсутствия утечек):

 1. дедупликация (user, item)                       — до сплита
 2. положительные взаимодействия                    — до сплита
 3. фильтр пользователей (>= min_user_items)        — до сплита: определяет
    «популяцию», а не каталог
 4. сплит по времени
 5. k-core фильтр на TRAIN (книги >= min_item_users читателей в train)
    — каталог определяется только прошлым
 6. Catalog из train
 7. val/test: оставляем только известных пользователей и книги каталога
"""
import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd

from .loading import load_interactions, load_items
from .catalog import Catalog
from .cleaning import (
    clear_deduplicate_interactions,
    filter_positive,
    filter_users,
    iterative_core_filter,
)
from .schema import DATE_COL, ITEM_COL, USER_COL
from .split import Splits, make_splits

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    # Файлы
    raw_dir: str = "data/raw"
    interactions_file: str = "interactions.csv"
    items_file: str = "items.csv"
    # Что считать положительным взаимодействием
    min_rating: float = 4.0
    min_progress: float = 80
    drop_below_rating: float | None = 1.5
    # Фильтры
    min_user_items: int = 10     # до сплита
    min_item_users: int = 20     # только по train
    min_train_items: int = 3     # минимум книг у пользователя в train
    # Сплит
    scheme: str = "global_time"  # или "per_user"
    val_size: float = 0.1
    test_size: float = 0.1

    @classmethod
    def from_dict(cls, params: dict) -> "DataConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(params) - known
        if unknown:
            raise ValueError(f"Неизвестные параметры DataConfig: {sorted(unknown)}")
        return cls(**params)


@dataclass
class PreparedData:
    splits: Splits
    catalog: Catalog
    report: dict

    def save(self, directory: str | Path) -> None:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        for name in ("train", "val", "test"):
            getattr(self.splits, name).to_csv(out / f"{name}.csv", index=False)
        self.catalog.save(out / "catalog")
        (out / "report.json").write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "PreparedData":
        src = Path(directory)
        parts = {
            name: pd.read_csv(src / f"{name}.csv", parse_dates=[DATE_COL])
            for name in ("train", "val", "test")
        }
        report = json.loads((src / "report.json").read_text(encoding="utf-8"))
        splits = Splits(**parts, meta=report.get("split", {}))
        return cls(splits, Catalog.load(src / "catalog"), report)


def prepare_data(cfg: DataConfig | None = None, *, interactions: pd.DataFrame | None = None,
                 items: pd.DataFrame | None = None) -> PreparedData:
    """
    Основная подготовка данных:
    - загрузка, удаление пропусков и простая предобработка 
    - удаление повторных взаимодействий 
    - разбиение по времени
    - создаём Catalog и удаляем книг и пользователей из val/test, которых уже нет в train
    """
    # Если interactions/items не переданы, они читаются из cfg.raw_dir
    cfg = cfg or DataConfig()
    # Загрузка данных, если не были переданы 
    if interactions is None or items is None:

        loaded_inter = load_interactions(Path(cfg.raw_dir) / cfg.interactions_file)
        loaded_items = load_items(Path(cfg.raw_dir) / cfg.items_file)

        interactions = loaded_inter if interactions is None else interactions
        items = loaded_items if items is None else items

    report = {"config": asdict(cfg), "rows_raw": int(len(interactions))}

    # Удаление дубликатов и отрицательных взаимодействий 
    data = clear_deduplicate_interactions(interactions)
    report["rows_deduplicated"] = int(len(data))
    data = filter_positive(data, cfg.min_rating, cfg.min_progress, cfg.drop_below_rating)
    report["rows_positive"] = int(len(data))

    # Удаляем пользователей, у который мало прочитанных книг
    data = filter_users(data, cfg.min_user_items)
    report["rows_after_user_filter"] = int(len(data))

    # Разбиение по времени 
    split_kwargs = {"val_size": cfg.val_size, "test_size": cfg.test_size}
    raw_splits = make_splits(data, cfg.scheme, **split_kwargs)

    # Удаляем книги и пользователей, о которых мало информации 
    train = iterative_core_filter(raw_splits.train, cfg.min_item_users, cfg.min_train_items)
    if train.empty:
        raise ValueError("После фильтров train пуст: ослабьте min_item_users / min_train_items")
    catalog = Catalog.from_train(train, items)

    # Оставляем val/test только в пределах каталога, построенного на train
    val = catalog.filter_known(raw_splits.val)
    test = catalog.filter_known(raw_splits.test)

    splits = Splits(train, val, test, raw_splits.meta)

    report["split"] = raw_splits.meta
    report["rows"] = {"train": len(train), "val": len(val), "test": len(test)}
    report["dropped_by_catalog"] = {
        "val": int(len(raw_splits.val) - len(val)),
        "test": int(len(raw_splits.test) - len(test)),
    }
    report["n_users"] = catalog.users_len
    report["n_items"] = catalog.items_len
    report["train_density"] = len(train) / (catalog.users_len * catalog.items_len)

    logger.info("Готово: %d пользователей, %d книг, train/val/test = %d/%d/%d",
                catalog.users_len, catalog.items_len, len(train), len(val), len(test))
    return PreparedData(splits, catalog, report)