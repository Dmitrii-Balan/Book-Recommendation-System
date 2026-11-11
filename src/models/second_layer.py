"""
Второй уровень: окна данных и обучение ранкера.

Окно = «что модель знала на момент X» + «что пользователи прочитали после».
Для каждого окна заново обучается первый уровень (новый объект Candidates), поэтому ни одно окно
не видит будущее относительно того, что предсказывает. Внутри окна:
    история -> Candidates.fit -> пул кандидатов -> признаки (FeatureBuilder) -> X
    цель    -> label = 1 стоит у пар пула, которые пользователь действительно прочитал в target -> y
Кандидаты и признаки строятся одним и тем же кодом для обоих окон (нет рассинхрона обучения и оценки).
"""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ..candidates.merge import pool_recall
from ..data.catalog import Catalog
from ..data.matrices import build_interaction_matrix
from ..data.schema import USER_IDX_COL
from ..features.builder import FeatureBuilder
from .first_layer import Candidates, make_labels, users_with_target
from .ranker import Ranker, grouped_ndcg

log = logging.getLogger(__name__)


@dataclass
class Window:
    name: str
    candidates: Candidates       # обученный первый уровень 
    pool: pd.DataFrame           # пул: user_idx, item_idx, {gen}_score, {gen}_rank, n_sources, rrf
    X: pd.DataFrame              # признаки второго уровня, строки как в pool
    y: np.ndarray                # 1, если пользователь прочитал книгу в target
    users: np.ndarray            # пользователи с непустой целью
    target: csr_matrix           # бинарная цель (n_users, n_items)
    pool_recall: float           # доля релевантных пар, попавших в пул, — потолок для ранкера


def build_window(name: str, history: pd.DataFrame, target_df: pd.DataFrame, catalog: Catalog,
                 candidates_kwargs: dict | None = None, pool_size: int | None = None) -> Window:
    """
    history — положительные взаимодействия, которые модель вправе знать;
    target_df — что пользователи прочитали после (источник label);
    candidates_kwargs — параметры Candidates (generators, k_per_generator, rrf_k);
    pool_size — сколько лучших по RRF кандидатов оставить на пользователя (None — весь пул);
    max_users — взять случайную подвыборку пользователей (ускорение).
    """
    candidates = Candidates(**(candidates_kwargs or {})).fit(history, catalog)

    target = build_interaction_matrix(target_df, catalog, weighting="binary")
    users = users_with_target(target)
    if len(users) == 0:
        raise ValueError(f"В цели окна {name!r} нет ни одного пользователя из каталога")
    # Получаем пул кандидатов 
    pool = candidates.get_candidates(users, max_per_user=pool_size)
    # Преобразуем матрицу взаимодействий в target переменную 
    y = make_labels(pool, target)
    # Получаем признаки для каждой пары пользователь - рекоммендация 
    features = FeatureBuilder(list(candidates.generators)).fit(history, catalog)
    X = features.transform(pool, candidates.matrices["binary"])

    recall = pool_recall(pool, target, users)
    log.info("окно %s: %d пользователей, %d строк пула, %d позитивов, потолок Recall пула %.3f",
             name, len(users), len(pool), int(y.sum()), recall)
    return Window(name, candidates, pool, X, y, users, target, recall)


def fit_ranker(window: Window, holdout_frac: float = 0.2, seed: int = 42, **ranker_kwargs) -> tuple[Ranker, dict]:
    """
    Обучает ранкер на окне. Часть пользователей откладывается и нужна только для выбора числа деревьев.
    Пользователи, у которых в пуле нет ни одного позитива, для обучения бесполезны и пропускаются.
    Возвращает (ранкер, {best_iteration, holdout_ndcg@10, holdout_rrf_ndcg@10, ...}).
    """
    groups = window.pool[USER_IDX_COL].to_numpy(dtype=np.int64)
    has_positive = pd.Series(window.y).groupby(groups).transform("sum").to_numpy() > 0
    users = np.unique(groups[has_positive])
    if len(users) < 2:
        raise ValueError("Для обучения ранкера нужно хотя бы 2 пользователя с позитивами в пуле")

    rng = np.random.default_rng(seed)
    holdout = rng.choice(users, size=max(1, int(len(users) * holdout_frac)), replace=False)
    in_hold = np.isin(groups, holdout) & has_positive
    in_train = has_positive & ~in_hold

    X, y = window.X, window.y
    ranker = Ranker(**ranker_kwargs).fit(X[in_train], y[in_train], groups[in_train],
                                         eval_set=(X[in_hold], y[in_hold], groups[in_hold]))
    info = {
        "backend": ranker.backend_, "best_iteration": ranker.best_iteration_,
        "train_users": int(len(users) - len(holdout)), "holdout_users": int(len(holdout)),
        "holdout_ndcg@10": grouped_ndcg(y[in_hold], ranker.predict(X[in_hold]), groups[in_hold], 10),
        "holdout_rrf_ndcg@10": grouped_ndcg(y[in_hold], -X.loc[in_hold, "rrf_rank"].to_numpy(),
                                            groups[in_hold], 10),
    }
    log.info("ранкер: %d деревьев; NDCG@10 на отложенных пользователях val: ранкер %.4f vs RRF %.4f",
             info["best_iteration"], info["holdout_ndcg@10"], info["holdout_rrf_ndcg@10"])
    return ranker, info