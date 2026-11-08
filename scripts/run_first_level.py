"""
Прогон первого уровня: данные -> генераторы кандидатов -> пул кандидатов -> Recall по моделям.

Запуск из корня репозитория:

    python scripts/run_first_level.py                              # val, все генераторы
    python scripts/run_first_level.py --generators als slim item_knn --ablation
    python scripts/run_first_level.py --save-pool                  # сохранить пул + label для ранкера
    python scripts/run_first_level.py --part test                  # финальная оценка (train+val -> test)

Результаты (reports/metrics):
    first_level_<part>.csv   метрики каждой модели
    pool_<part>.csv          Recall пула при разном max_per_user (ограничение по RRF)
    ablation_<part>.csv      (с --ablation) Recall пула без каждого из генераторов
    run_<part>.json          параметры запуска и отчёт о данных

Два вида Recall:
    macro — Recall@K из ranking_metrics: доля релевантных книг пользователя в его топ-K,
            усреднённая по пользователям;
    micro — доля релевантных пар (пользователь, книга) среди всех пар цели; именно её
            считает pool_recall, это потолок для второго уровня.
"""
import os

# implicit рекомендует один BLAS-поток: иначе внутренний пул потоков сильно тормозит ALS.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import logging
import sys
from pathlib import Path 

import numpy as np  
import pandas as pd 

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.candidates import merge_candidates, pool_recall
from src.candidates.base import PAD_ITEM
from src.candidates.registry import build_generators
from src.data.matrices import build_interaction_matrix 
from src.data.prepare import DataConfig, prepare_data 
from src.data.schema import ITEM_IDX_COL, USER_IDX_COL  
from src.evaluation.metrics import hit_matrix, ranking_metrics 

log = logging.getLogger("first_level")


# --------------------------------------------------------------------------- аргументы

def parse_args() -> argparse.Namespace:
    d = DataConfig()
    p = argparse.ArgumentParser(
        description="Прогон первого уровня и оценка Recall",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--part", choices=["val", "test"], default="val",
                   help="на чём оцениваем: val (train -> val) или test (train+val -> test)")
    p.add_argument("--generators", nargs="+", choices=sorted(build_generators()), default=None,
                   help="какие генераторы запускать (по умолчанию все из registry)")
    p.add_argument("--ks", nargs="+", type=int, default=[10, 50, 100, 200],
                   help="K для метрик; каждый генератор отдаёт max(ks) кандидатов")
    p.add_argument("--pool-sizes", nargs="+", type=int, default=[100, 200, 300, 500],
                   help="сколько лучших по RRF кандидатов на пользователя оставлять в пуле")
    p.add_argument("--rrf-k", type=int, default=100, help="константа в Reciprocal Rank Fusion")
    p.add_argument("--ablation", action="store_true",
                   help="пересчитать Recall пула без каждого из генераторов по очереди")
    p.add_argument("--save-pool", action="store_true",
                   help="сохранить полный пул кандидатов с колонкой label (для обучения ранкера)")
    p.add_argument("--seed", type=int, default=42, help="seed для --max-users")
    p.add_argument("--out-dir", default=str(ROOT / "reports" / "metrics"))
    p.add_argument("--interim-dir", default=str(ROOT / "data" / "interim"),
                   help="куда писать пул при --save-pool")

    g = p.add_argument_group("данные (значения по умолчанию — из DataConfig)")
    g.add_argument("--raw-dir", default=str(ROOT / d.raw_dir))
    g.add_argument("--interactions-file", default=d.interactions_file)
    g.add_argument("--items-file", default=d.items_file)
    g.add_argument("--scheme", choices=["global_time", "per_user"], default=d.scheme)
    g.add_argument("--val-size", type=float, default=d.val_size)
    g.add_argument("--test-size", type=float, default=d.test_size)
    g.add_argument("--min-rating", type=float, default=d.min_rating)
    g.add_argument("--min-progress", type=float, default=d.min_progress)
    g.add_argument("--min-user-items", type=int, default=d.min_user_items)
    g.add_argument("--min-item-users", type=int, default=d.min_item_users)
    g.add_argument("--min-train-items", type=int, default=d.min_train_items)

    args = p.parse_args()
    args.ks = sorted(set(args.ks))
    args.pool_sizes = sorted(set(args.pool_sizes))
    return args

# --------------------------------------------------------------------------- метрики

def relevant_keys(target, users: np.ndarray, n_items: int) -> np.ndarray:
    """
    Создаёт массив закодированных релевантных пар users × книги.
    Кодируем формулой user_idx * n_items + item_idx.
    """
    sub = target[users].tocoo() # Coordinate format
    return users[sub.row].astype(np.int64) * n_items + sub.col.astype(np.int64)

def pool_scores(pool: pd.DataFrame, rel_keys: np.ndarray, users: np.ndarray,
                n_relevant: np.ndarray, n_users_total: int, n_items: int) -> dict:
    """
    pool - пары рекоммендаций (user, item)
    pool_recall: micro (по парам) и macro (среднее по пользователям)
    """
    pool_users = pool[USER_IDX_COL].to_numpy(dtype=np.int64)
    # Кодируем чтобы сравнивать пары взаимодействий (user_id, item_id)
    keys = pool_users * n_items + pool[ITEM_IDX_COL].to_numpy(dtype=np.int64)
    found = np.isin(keys, rel_keys)
    # Считаем повторения успешных рекоммендаций по пользователям
    per_user = np.bincount(pool_users[found], minlength=n_users_total)[users]
    return {
        "pool_recall": float(found.sum() / n_relevant.sum()),
        "pool_recall_macro": float(np.mean(per_user / n_relevant)),
        "avg_pool_size": len(pool) / len(users),
    }

def pool_table(pool: pd.DataFrame, sizes: list[int], **kw) -> pd.DataFrame:
    """Recall пула при разных max_per_user. pool уже отсортирован по RRF внутри пользователя."""
    rows = []
    for size in [*sizes, None]:
        sub = pool if size is None else pool.groupby(USER_IDX_COL, sort=False).head(size)
        rows.append({"max_per_user": "all" if size is None else size, **pool_scores(sub, **kw)})
    return pd.DataFrame(rows)

def show(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# --------------------------------------------------------------------------- основной прогон

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    ks, k_max, part = args.ks, max(args.ks), args.part

    # === 1. Подготовка данных ===
    # Запоминаем информацию о данных
    cfg = DataConfig(
        raw_dir=args.raw_dir, interactions_file=args.interactions_file, items_file=args.items_file,
        min_rating=args.min_rating, min_progress=args.min_progress,
        min_user_items=args.min_user_items, min_item_users=args.min_item_users,
        min_train_items=args.min_train_items,
        scheme=args.scheme, val_size=args.val_size, test_size=args.test_size,
    )
    # Загружаем, фильтруем и разбиваем  
    log.info("Готовлю данные из %s", cfg.raw_dir)
    data = prepare_data(cfg)
    splits, catalog = data.splits, data.catalog
    items_len = catalog.items_len
    print("\n=== Разбиение ===")
    print(splits.summary().to_string(index=False))
    log.info("Каталог: %d пользователей, %d книг, плотность train %.5f",
             catalog.users_len, items_len, data.report["train_density"])

    # === 2. История, цель и построение матрицы взаимодействий === 
    history = splits.history(part) # part = val/test 
    target = build_interaction_matrix(splits.target(part), catalog, weighting="binary")
    # Оставляем только тех пользователей, о которых есть информация в target
    users = np.flatnonzero(np.diff(target.indptr) > 0)
    if len(users) == 0:
        raise SystemExit(f"В target({part!r}) нет ни одного пользователя из каталога")
    # Считаем количество релевантных книг по пользователям
    n_relevant = np.diff(target.indptr)[users]
    total_relevant = int(n_relevant.sum())
    rel_keys = relevant_keys(target, users, items_len)
    log.info("Оценка на %s: %d пользователей, %d релевантных пар; история — %d строк",
             part, len(users), total_relevant, len(history))

    # === 3. Генераторы: fit -> recommend -> метрики ===
    # Создаём генераторы кандидатов
    generators = build_generators(args.generators)
    # Строим для них матрицы взаимодействий 
    matrices: dict[str, object] = {}
    for _, weighting in generators.values():
        if weighting not in matrices:
            matrices[weighting] = build_interaction_matrix(history, catalog, weighting=weighting)
    # Обучем генеравторы и получаем рекоммендации 
    recs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    hit_keys: dict[str, np.ndarray] = {}
    rows = []
    for i, (name, (gen, weighting)) in enumerate(generators.items(), start=1):
        log.info("[%d/%d] %s (матрица: %s) — обучение", i, len(generators), name, weighting)
        gen.fit(matrices[weighting], catalog)
        # Получаем рекоммендуемые книги и их порядок
        items, scores = gen.recommend(users, k=k_max)
        recs[name] = (items, scores)
        # Получаем оценки рекоммендаций 
        metrics = ranking_metrics(items, target, users, ks=tuple(ks), n_items=items_len)
        hits = hit_matrix(items, target, users)
        u_pos, r_pos = np.nonzero(hits)
        # Кодируем рекоммендуемые книги
        keys = users[u_pos].astype(np.int64) * items_len + items[u_pos, r_pos].astype(np.int64)
        hit_keys[name] = np.unique(keys)

        valid = items != PAD_ITEM
        rows.append({
            "model": name,
            "avg_cands": float(valid.sum(axis=1).mean()),
            "no_cands_share": float((valid.sum(axis=1) == 0).mean()),
            **{k: v for k, v in metrics.items() if k != "n_users"},
        })
        log.info("    Recall@%d = %.4f", k_max, metrics[f"Recall@{k_max}"])

    # === 4. Микро-Recall каждой модели и вклад, которого нет у остальных === 
    # Объединяем предсказания всех моделей 
    union = np.unique(np.concatenate(list(hit_keys.values())))
    # Проходимся по каждой модели и считаем её вклад в общий пул
    for row in rows:
        mine = hit_keys[row["model"]]
        # Предсказания других моделей 
        others = [v for name, v in hit_keys.items() if name != row["model"]]
        rest = np.unique(np.concatenate(others)) if others else np.empty(0, dtype=np.int64)
        # Метрика модели и её отдельный вклад в общую метрику  
        row[f"PairRecall@{k_max}"] = len(mine) / total_relevant
        row[f"UniqueRecall@{k_max}"] = len(np.setdiff1d(mine, rest, assume_unique=True)) / total_relevant
    # Recall по предсказаниям всех моделей
    union_recall = len(union) / total_relevant
    # Выводим расчитанные метрики
    models = pd.DataFrame(rows).sort_values(f"Recall@{k_max}", ascending=False).reset_index(drop=True)
    shown = (["model", "avg_cands"] + [f"Recall@{k}" for k in ks]
             + [f"NDCG@{ks[0]}", f"Coverage@{k_max}", f"PairRecall@{k_max}", f"UniqueRecall@{k_max}"])
    show(f"Модели ({part}), топ-{k_max} на пользователя", models[shown])

    # === 5. Подготовка пула кандидатов для второго уровня ===
    log.info("Собираю пул кандидатов (RRF, rrf_k=%d)", args.rrf_k)
    pool = merge_candidates(users, recs, rrf_k=args.rrf_k)
    pool_kw = dict(rel_keys=rel_keys, users=users, n_relevant=n_relevant,
                   n_users_total=target.shape[0], n_items=items_len)
    pools = pool_table(pool, args.pool_sizes, **pool_kw)
    show(f"Пул: Recall при разном max_per_user ({part})", pools)

    ref = pool_recall(pool, target, users)
    if np.isclose(ref, union_recall):
        log.info("Проверка пройдена: Recall объединения топ-%d (%.4f) совпал с pool_recall", k_max, ref)
    else:
        log.warning("Recall объединения (%.6f) не совпал с pool_recall (%.6f) — проверьте merge.py",
                    union_recall, ref)

    # === 6. вклад каждого генератора в пул ===
    ablation = None
    if args.ablation and len(recs) > 1:
        parts = []
        for name in recs:
            log.info("Абляция: пул без %s", name)
            subset = {k: v for k, v in recs.items() if k != name}
            sub_pool = merge_candidates(users, subset, rrf_k=args.rrf_k)
            parts.append(pool_table(sub_pool, args.pool_sizes, **pool_kw).assign(without=name))
        ablation = pd.concat(parts, ignore_index=True)
        full = pools.set_index("max_per_user")["pool_recall"]
        ablation["delta_vs_full"] = ablation["pool_recall"] - ablation["max_per_user"].map(full)
        view = ablation.pivot(index="without", columns="max_per_user", values="delta_vs_full")
        view = view[[*args.pool_sizes, "all"]].reset_index()
        view.columns = ["without", *[f"pool_{c}" for c in view.columns[1:]]]
        show("Абляция: изменение pool_recall (micro) без генератора "
             "(< 0 — генератор нужен, >= 0 — не добавляет Recall при таком размере пула)", view)

    # === 7. Сохранение ===
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models.to_csv(out / f"first_level_{part}.csv", index=False)
    pools.to_csv(out / f"pool_{part}.csv", index=False)
    if ablation is not None:
        ablation.to_csv(out / f"ablation_{part}.csv", index=False)
    (out / f"run_{part}.json").write_text(json.dumps({
        "args": vars(args),
        "data_report": data.report,
        "eval_users": int(len(users)),
        "relevant_pairs": total_relevant,
        "union_recall": union_recall,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log.info("Таблицы сохранены в %s", out)

    if args.save_pool:
        keys = (pool[USER_IDX_COL].to_numpy(dtype=np.int64) * items_len
                + pool[ITEM_IDX_COL].to_numpy(dtype=np.int64))
        labeled = pool.assign(label=np.isin(keys, rel_keys).astype(np.int8))
        interim = Path(args.interim_dir)
        interim.mkdir(parents=True, exist_ok=True)
        try:
            path = interim / f"pool_{part}.parquet"
            labeled.to_parquet(path, index=False)
        except ImportError:
            path = interim / f"pool_{part}.csv.gz"
            labeled.to_csv(path, index=False)
        log.info("Пул (%d строк, label=1 у %d) сохранён: %s", len(labeled), int(labeled["label"].sum()), path)

    print(f"\nЛучшая одиночная модель по micro-Recall@{k_max}: "
          f"{models.loc[models[f'PairRecall@{k_max}'].idxmax(), 'model']} "
          f"({models[f'PairRecall@{k_max}'].max():.4f}); "
          f"объединение всех: {union_recall:.4f}")
    print("Recall@K в таблице моделей — macro (по пользователям); PairRecall, UniqueRecall и "
          "pool_recall — micro (по парам).")


if __name__ == "__main__":
    main()