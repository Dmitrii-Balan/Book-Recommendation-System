"""
Оба уровня целиком и метрики итоговых рекомендаций.

    окно val  (первый уровень на train -> пул -> признаки -> label из val) -> обучение ранкера
    окно test (первый уровень на train+val -> пул -> признаки)             -> метрики на test

Ранкер видит только окно val; test используется один раз, для итоговой оценки: так цифры честные.
На одном и том же пуле test сравниваются:
    ranker   — итоговая выдача: пул, упорядоченный ранкером;
    rrf      — пул, упорядоченный только слиянием первого уровня (Reciprocal Rank Fusion);
    <модель> — выдача каждого генератора первого уровня по отдельности;
    oracle   — идеальный порядок внутри пула (потолок: выше него второй уровень подняться не может).

Запуск из корня репозитория:

    python scripts/run_pipeline.py                                  # все генераторы
    python scripts/run_pipeline.py --examples 3                     # показать выдачу для 3 пользователей

Результаты (reports/metrics): pipeline_test.csv (метрики) и pipeline_run.json (параметры и сведения о запуске).
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

from src.candidates.registry import build_generators
from src.data.prepare import DataConfig, prepare_data
from src.data.schema import DATE_COL, ITEM_COL, USER_COL
from src.evaluation.pool_eval import evaluate_methods, topk_from_scores
from src.models.second_layer import Window, build_window, fit_ranker

log = logging.getLogger("pipeline")


def parse_args() -> argparse.Namespace:
    d = DataConfig()
    p = argparse.ArgumentParser(description="Первый + второй уровень: обучение и метрики итоговых рекомендаций",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--generators", nargs="+", choices=sorted(build_generators()), default=None,
                   help="какие генераторы первого уровня использовать (по умолчанию все из registry)")
    p.add_argument("--k-per-generator", type=int, default=200, help="сколько кандидатов берём у каждого генератора")
    p.add_argument("--rrf-k", type=int, default=100, help="константа в Reciprocal Rank Fusion")
    p.add_argument("--pool-size", type=int, default=300,
                   help="сколько лучших по RRF кандидатов на пользователя остаётся в пуле (0 — весь пул)")
    p.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20], help="K для метрик итоговой выдачи")
    p.add_argument("--backend", choices=["auto", "catboost", "hgb"], default="auto", help="библиотека ранкера")
    p.add_argument("--iterations", type=int, default=500, help="максимум деревьев ранкера")
    p.add_argument("--examples", type=int, default=0, help="показать выдачу для N случайных пользователей test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=str(ROOT / "reports" / "metrics"))

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
    return args


def rankings(window: Window, ranker_scores: np.ndarray) -> dict[str, np.ndarray]:
    """Способы упорядочить пул окна: имя -> оценка каждой строки пула (чем больше, тем раньше в выдаче)."""
    # Записываем ранжирование первым уровнем и ранкером второго уровня 
    pool = window.pool
    methods = {"ranker": ranker_scores, "rrf": pool["rrf"].to_numpy()}
    # Записываем ранжирование отдельными генераторами 
    for name in window.candidates.generators:
        methods[name] = -pool[f"{name}_rank"].to_numpy()
    methods["oracle"] = window.y.astype(np.float64)
    return methods

def show_examples(window: Window, scores: np.ndarray, history: pd.DataFrame, catalog, n: int, k: int,
                  seed: int) -> None:
    """Выдача ранкера для n случайных пользователей: что читал раньше и что рекомендуем"""
    rng = np.random.default_rng(seed)
    users = np.sort(rng.choice(window.users, size=min(n, len(window.users)), replace=False))
    top = topk_from_scores(window.pool, scores, users, k)

    meta = catalog.meta
    titles = meta["title"].to_numpy()
    authors = meta["authors"].to_numpy()

    # Добавляем к истории названия книг и авторов.
    last_read = history.sort_values(DATE_COL).merge(
        meta[[ITEM_COL, "title", "authors"]],
        on=ITEM_COL,
        how="left"
    )

    for row, user in enumerate(users):
        user_id = catalog.users.decode(user)
        # Последние 5 прочитанных книг.
        read = last_read[last_read[USER_COL] == user_id].tail(5)
        print(f"\n--- пользователь {user_id}: в test прочитал новых книг: {window.target[user].nnz}")
        print("  Недавно читал:")
        if read.empty:
            print("    —")
        else:
            for i, (_, book) in enumerate(read.iterrows(), start=1):
                title = book["title"]
                author = book["authors"]
                if pd.isna(author):
                    author = "неизвестен"
                print(f"    {i}. {title} — {author}")
        # Рекомендации ранкера.
        print("  Рекомендации:")
        for rank, item in enumerate(top[row], start=1):
            if item < 0:
                break
            mark = "✓" if window.target[user, item] > 0 else " "
            author = authors[item]
            if pd.isna(author):
                author = "неизвестен"
            print(f"    {rank:>2}. {mark} {titles[item]} — {author}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    ks, k_main = args.ks, 10 if 10 in args.ks else args.ks[0]

    # === 1. Данные ===
    cfg = DataConfig(
        raw_dir=args.raw_dir, interactions_file=args.interactions_file, items_file=args.items_file,
        min_rating=args.min_rating, min_progress=args.min_progress,
        min_user_items=args.min_user_items, min_item_users=args.min_item_users,
        min_train_items=args.min_train_items,
        scheme=args.scheme, val_size=args.val_size, test_size=args.test_size,
    )
    log.info("Готовлю данные из %s", cfg.raw_dir)
    data = prepare_data(cfg)
    splits, catalog = data.splits, data.catalog
    print("\n=== Разбиение ===")
    print(splits.summary().to_string(index=False))

    window_kw = dict(
        candidates_kwargs=dict(generators=args.generators, k_per_generator=args.k_per_generator,
                               rrf_k=args.rrf_k),
        pool_size=args.pool_size or None
    )

    # === 2. Окно val: на нём учится ранкер ===
    # Обучаем генераторы 
    val = build_window("val", splits.history("val"), splits.target("val"), catalog, **window_kw)
    ranker, ranker_info = fit_ranker(val, seed=args.seed, backend=args.backend, iterations=args.iterations,
                                     random_state=args.seed)

    # === 3. Окно test: только оценка ===
    test = build_window("test", splits.history("test"), splits.target("test"), catalog, **window_kw)
    methods = rankings(test, ranker.predict(test.X))
    table = evaluate_methods(test.pool, methods, test.target, test.users, ks=tuple(ks), n_items=catalog.items_len)

    # === 4. Вывод ===
    shown = ["method"] + [f"{m}@{k_main}" for m in ("Precision", "Recall", "HitRate", "NDCG", "MAP", "Coverage")]
    shown += [f"{m}@{k}" for k in ks if k != k_main for m in ("Recall", "NDCG")]
    print(f"\n=== Итоговые метрики на test: {len(test.users)} пользователей, "
          f"Recall пула (потолок) {test.pool_recall:.4f} ===")
    print(table[shown].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    base = table.set_index("method")[f"NDCG@{k_main}"]
    print(f"\nNDCG@{k_main}: ранкер {base['ranker']:.4f} vs RRF {base['rrf']:.4f} "
          f"({(base['ranker'] / base['rrf'] - 1) * 100:+.1f}%), потолок пула {base['oracle']:.4f}")

    # === 5. Сохранение ===
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "pipeline_test.csv", index=False)
    (out / "pipeline_run.json").write_text(json.dumps({
        "args": vars(args), "data_report": data.report, "ranker": ranker_info,
        "val_pool_recall": val.pool_recall, "test_pool_recall": test.pool_recall,
        "test_users": int(len(test.users)),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log.info("Таблица сохранена в %s", out / "pipeline_test.csv")

    # === 6. Необязательный вывод (после сохранения: сбой здесь не должен терять результаты) ===
    importance = ranker.feature_importance()
    if importance is not None:
        print("\nВажнейшие признаки ранкера:")
        print(importance.head(10).round(2).to_string())

    if args.examples:
        show_examples(test, methods["ranker"], splits.history("test"), catalog, args.examples, k_main, args.seed)


if __name__ == "__main__":
    main()