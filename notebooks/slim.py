import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import ElasticNet
from tqdm.auto import tqdm


class SLIMSklearn:
    """
    SLIM (Sparse Linear Methods for Top-N Recommendation).

    Обучает Elastic Net регрессию для каждой книги:
        R[:, j] ~= R[:, other_items] @ w_j

    Коэффициенты неотрицательны, самопредсказание запрещено.
    """

    def __init__(
        self,
        alpha=1e-4,
        l1_ratio=0.5,
        max_iter=500,
        tol=1e-4,
        max_items=300,
        user_col="user_id",
        item_col="item_id",
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_items = max_items
        self.user_col = user_col
        self.item_col = item_col

    def fit(self, X: pd.DataFrame, y=None):
        """Обучает item-item матрицу весов только на train."""

        pairs = (
            X[[self.user_col, self.item_col]]
            .dropna()
            .drop_duplicates()
            .copy()
        )

        # Для первого эксперимента ограничиваем каталог.
        # Частоты считаются исключительно на train.
        if self.max_items is not None:
            selected_items = (
                pairs[self.item_col]
                .value_counts()
                .head(self.max_items)
                .index
            )
            pairs = pairs[
                pairs[self.item_col].isin(selected_items)
            ].copy()

        if pairs[self.item_col].nunique() < 2:
            raise ValueError(
                "Для SLIM необходимо хотя бы две книги."
            )

        # Отображения исходных ID в индексы матрицы
        self.user_ids_ = (
            pairs[self.user_col].drop_duplicates().to_numpy()
        )
        self.item_ids_ = (
            pairs[self.item_col].drop_duplicates().to_numpy()
        )

        self.user2idx = {
            user_id: idx
            for idx, user_id in enumerate(self.user_ids_)
        }
        self.item2idx = {
            item_id: idx
            for idx, item_id in enumerate(self.item_ids_)
        }

        rows = pairs[self.user_col].map(
            self.user2idx
        ).to_numpy(dtype=np.int32)

        cols = pairs[self.item_col].map(
            self.item2idx
        ).to_numpy(dtype=np.int32)

        # Бинарная матрица взаимодействий пользователей с книгами
        self.R_ = csr_matrix(
            (
                np.ones(len(pairs), dtype=np.float32),
                (rows, cols),
            ),
            shape=(len(self.user_ids_), len(self.item_ids_)),
        )

        R_csc = self.R_.tocsc()
        n_users, n_items = R_csc.shape

        # W[i, j] — вклад книги i в предсказание книги j
        w_rows = []
        w_cols = []
        w_data = []

        for j in tqdm(range(n_items), desc="Training SLIM"):
            # Исключаем целевую книгу j из признаков.
            X_j = hstack(
                (R_csc[:, :j], R_csc[:, j + 1:]),
                format="csc",
                dtype=np.float32,
            )

            # Цель — взаимодействия с книгой j
            y_j = R_csc[:, j].toarray().ravel()

            model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                positive=True,
                fit_intercept=False,
                max_iter=self.max_iter,
                tol=self.tol,
                selection="cyclic",
            )

            model.fit(X_j, y_j)

            # Храним только ненулевые коэффициенты
            nonzero = np.flatnonzero(
                model.coef_ > 1e-8
            )

            # Возвращаем индексы к исходной нумерации:
            # после j все признаки сдвинуты на один индекс
            source_items = np.where(
                nonzero < j,
                nonzero,
                nonzero + 1,
            )

            w_rows.extend(source_items.tolist())
            w_cols.extend([j] * len(nonzero))
            w_data.extend(
                model.coef_[nonzero].astype(
                    np.float32
                ).tolist()
            )

        self.W_ = csr_matrix(
            (
                np.asarray(w_data, dtype=np.float32),
                (
                    np.asarray(w_rows, dtype=np.int32),
                    np.asarray(w_cols, dtype=np.int32),
                ),
            ),
            shape=(n_items, n_items),
        )

        return self

    def recommend(
        self,
        user_id,
        n_recommendations=10,
    ) -> pd.DataFrame:
        """Возвращает Top-N непрочитанных книг пользователя."""

        if user_id not in self.user2idx:
            raise ValueError(
                f"Неизвестный пользователь: {user_id}"
            )

        user_idx = self.user2idx[user_id]
        profile = self.R_.getrow(user_idx)

        # Оценки всех книг: r_u @ W
        scores = np.asarray(
            (profile @ self.W_).toarray()
        ).ravel()

        # Исключаем книги, с которыми было взаимодействие
        scores[profile.indices] = -np.inf

        # Кандидаты с положительной оценкой
        candidates = np.flatnonzero(
            np.isfinite(scores) & (scores > 0)
        )

        top_idx = candidates[
            np.argsort(scores[candidates])[::-1][
                :n_recommendations
            ]
        ]

        return pd.DataFrame({
            self.item_col: self.item_ids_[top_idx],
            "score": scores[top_idx],
        })