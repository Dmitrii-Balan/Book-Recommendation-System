"""
Ранкер второго уровня: градиентный бустинг, который по признакам пары (пользователь, книга)
упорядочивает кандидатов первого уровня.

Модели:
  "catboost" — CatBoostRanker с функцией потерь YetiRank (учится упорядочивать книги внутри пользователя);
  "hgb"      — HistGradientBoostingClassifier из sklearn (оценка «прочитает / нет»);

Качество измеряется внутри пользователя, а не по всем строкам сразу.

    ranker = Ranker().fit(X, y, groups, eval_set=(X_val, y_val, groups_val))
    scores = ranker.predict(X_new)
"""
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def grouped_ndcg(y, score, groups, k: int = 10) -> float:
    """
    Среднее NDCG@k по группам (пользователям); группы без релевантных книг пропускаются.
    y - истинная релевантность
    score - предсказанная релевантность 
    groups - разделяет предсказания на группы 
    """
    # Создаём датафрейм по исходным данным 
    frame = pd.DataFrame({"g": np.asarray(groups), "y": np.asarray(y, dtype=np.float64),
                          "s": np.asarray(score, dtype=np.float64)})
    # Сортируем предсказания внутри каждой группы (пользователя)
    frame = frame.sort_values(["g", "s"], ascending=[True, False], kind="stable")
    # Позиция рекоммендованной книги в группе 
    frame["pos"] = frame.groupby("g", sort=False).cumcount()
    # Считаем коэффициенты убывания и IDCG 
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = np.cumsum(discount)
    # Оставляем не более k рекомендаций в каждой группе 
    top = frame[frame["pos"] < k]
    dcg = (top["y"] * discount[top["pos"].to_numpy()]).groupby(top["g"]).sum()
    # Считаем количество релевантных книг по пользователям 
    n_relevant = frame.groupby("g")["y"].sum()
    n_relevant = n_relevant[n_relevant > 0]
    ideal_dcg = pd.Series(ideal[np.minimum(n_relevant.to_numpy().astype(int), k) - 1], index=n_relevant.index)
    return float((dcg.reindex(n_relevant.index).fillna(0.0) / ideal_dcg).mean())


class Ranker:
    def __init__(self, backend: str = "auto", iterations: int = 500, learning_rate: float = 0.05,
                 depth: int = 6, l2_leaf_reg: float = 5.0, early_stopping_rounds: int = 50,
                 max_neg_per_user: int | None = 200, random_state: int = 42):
        """
        iterations — максимум деревьев; реальное число выбирается по eval_set (early stopping);
        max_neg_per_user — оставлять в обучении не больше стольких случайных «негативов» на пользователя
        (все позитивы сохраняются): сильно ускоряет обучение; None — не урезать.
        """
        self.backend = backend
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.early_stopping_rounds = early_stopping_rounds
        self.max_neg_per_user = max_neg_per_user
        self.random_state = random_state

    def _resolve_backend(self) -> str:
        if self.backend in ("catboost", "hgb"):
            return self.backend
        if self.backend != "auto":
            raise ValueError(f"backend должен быть auto / catboost / hgb, а не {self.backend!r}")
        return "catboost"
        
    def _subsample(self, y: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Маска строк: все позитивы + не больше max_neg_per_user случайных негативов на пользователя."""
        if not self.max_neg_per_user:
            return np.ones(len(y), dtype=bool)
        keep = y == 1
        # Получаем индексы негативных примеров 
        negatives = np.flatnonzero(~keep)
        # Генерируем случайные числа для негативов, для случайно сортировки и удаления
        noise = pd.Series(np.random.default_rng(self.random_state).random(len(negatives)))
        order = noise.groupby(codes[negatives]).rank(method="first").to_numpy()
        keep[negatives[order <= self.max_neg_per_user]] = True
        return keep

    @staticmethod
    def _sort_by_group(X: pd.DataFrame, y: np.ndarray, groups):
        """catboost требует, чтобы строки одной группы шли подряд."""
        order = np.argsort(np.asarray(groups), kind="stable")
        return X.iloc[order].reset_index(drop=True), y[order], np.asarray(groups)[order]

    def fit(self, X: pd.DataFrame, y, groups, eval_set: tuple | None = None) -> "Ranker":
        """
        Передаём кандидатов на рекомендации (X) и есть ли эта книга в target окне (y)
        X, y, groups — обучающая выборка (groups — id пользователя у каждой строки);
        eval_set — (X_val, y_val, groups_val): на нём выбирается число деревьев (early stopping).
        """
        # Приведение данных
        y = np.asarray(y).astype(np.int8)
        self.feature_names_ = list(X.columns)
        self.backend_ = self._resolve_backend()
        # Кодируем пользователей 
        codes, _ = pd.factorize(np.asarray(groups))
        # Удаляем лишние негативы   
        keep = self._subsample(y, codes)
        X, y, codes = X[keep], y[keep], codes[keep]
        log.info("ранкер (%s): %d строк, %d позитивов, %d пользователей",
                 self.backend_, len(y), int(y.sum()), len(np.unique(codes)))

        if self.backend_ == "catboost":
            self._fit_catboost(X, y, codes, eval_set)
        else:
            self._fit_hgb(X, y, eval_set)
        return self

    def _fit_catboost(self, X, y, codes, eval_set) -> None:
        from catboost import CatBoostRanker, Pool

        self.model_ = CatBoostRanker(
            loss_function="YetiRank", eval_metric="NDCG:top=10", iterations=self.iterations,
            learning_rate=self.learning_rate, depth=self.depth, l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_state, verbose=False, allow_writing_files=False)
        
        Xs, ys, gs = self._sort_by_group(X, y, codes)
        fit_kwargs = {}
        if eval_set is not None:
            Xv, yv, gv = eval_set
            Xv, yv, gv = self._sort_by_group(Xv[self.feature_names_], np.asarray(yv).astype(np.int8), gv)
            fit_kwargs = dict(eval_set=Pool(Xv, label=yv, group_id=gv), use_best_model=True,
                              early_stopping_rounds=self.early_stopping_rounds)
        self.model_.fit(Pool(Xs, label=ys, group_id=gs), **fit_kwargs)
        # Сохраняем оптимальное количество деревьев
        self.best_iteration_ = int(self.model_.get_best_iteration() + 1) if eval_set is not None \
            else self.iterations

    def _fit_hgb(self, X, y, eval_set) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier

        def make(n_iter: int) -> HistGradientBoostingClassifier:
            return HistGradientBoostingClassifier(
                max_iter=n_iter, learning_rate=max(self.learning_rate, 0.05), max_depth=self.depth,
                l2_regularization=self.l2_leaf_reg, min_samples_leaf=100, early_stopping=False,
                random_state=self.random_state)

        model = make(self.iterations).fit(X, y)
        best = self.iterations
        if eval_set is not None:
            # Число деревьев выбираем по NDCG на отложенных пользователях (раз в 10 деревьев)
            Xv, yv, gv = eval_set
            best_ndcg, since = -1.0, 0
            for stage, raw in enumerate(model.staged_decision_function(Xv[self.feature_names_]), start=1):
                if stage % 10 and stage != self.iterations:
                    continue
                ndcg = grouped_ndcg(yv, raw, gv, 10)
                if ndcg > best_ndcg + 1e-6:
                    best, best_ndcg, since = stage, ndcg, 0
                else:
                    since += 10
                    if since >= self.early_stopping_rounds:
                        break
            if best < self.iterations:          # HistGradientBoosting нельзя «обрезать» — переобучаем
                model = make(best).fit(X, y)
        self.model_, self.best_iteration_ = model, best

    # ------------------------------------------------------------------ применение

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Оценка каждой пары: чем выше, тем раньше книга в выдаче (сравнивать только внутри пользователя)."""
        X = X[self.feature_names_]
        if self.backend_ == "catboost":
            return np.asarray(self.model_.predict(X), dtype=np.float64)
        return np.asarray(self.model_.decision_function(X), dtype=np.float64)

    def feature_importance(self) -> pd.Series | None:
        """
        Важность признаков (только catboost), None — если её получить нельзя.
        Тип указан явно: для ранжирующих функций потерь catboost по умолчанию берёт LossFunctionChange,
        а ему нужна обучающая выборка. PredictionValuesChange считается по самой модели.
        """
        if self.backend_ != "catboost":
            return None
        try:
            values = self.model_.get_feature_importance(type="PredictionValuesChange")
        except Exception as error:      # важность — только для вывода, она не должна ронять прогон
            log.warning("не удалось получить важность признаков: %s", error)
            return None
        return pd.Series(values, index=self.feature_names_, name="importance").sort_values(ascending=False)