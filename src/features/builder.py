"""
Признаки второго уровня (для ранкера) для пар (пользователь, книга) из пула кандидатов.

Главное правило — только прошлое: статистика книг считается по той же истории, на которой
учился первый уровень (для окна val это train, для окна test — train+val). Будущее в признаки
не попадает.

Группы признаков:
  источники    {gen}_score, {gen}_rank (NaN, если генератор книгу не предлагал), n_sources, rrf, rrf_rank
               — приходят из пула первого уровня (см. src/candidates/merge.py);
  книга        i_log_pop (популярность), i_pop_pct (её перцентиль), i_recent_share (доля читателей
               за последние 90 дней), i_mean_rating, i_n_ratings, i_year, i_n_genres;
  пользователь u_n_items (размер истории), u_mean_log_pop (насколько «мейнстримно» он читает);
  пара         genre_aff_sum / genre_aff_max (близость жанров книги к жанрам из истории пользователя),
               author_cnt (сколько книг этого автора он уже читал), pop_gap и year_gap
               (отличие книги от «средней» книги пользователя).

Использование:
    features = FeatureBuilder(generator_names).fit(history, catalog)
    X = features.transform(pool, binary_matrix)     # строки X — как строки pool
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.feature_extraction.text import CountVectorizer

from ..candidates.merge import N_SOURCES_COL, RRF_COL
from ..data.catalog import Catalog
from ..data.schema import DATE_COL, ITEM_IDX_COL, RATING_COL, USER_IDX_COL

RECENT_DAYS = 90
CHUNK = 200_000        # строк пула за раз в разреженных операциях (ограничивает память)


def _split_commas(text: str) -> list[str]:
    """«Фэнтези, Приключения» -> ["фэнтези", "приключения"]."""
    return [part.strip().lower() for part in str(text).split(",") if part.strip()]


def _bag_of_words(column: pd.Series | None, n_items: int) -> csr_matrix:
    """
    Матрица «книга × жанр (или автор)»: 1, если значение есть у книги.
    column - это одна колонка таблицы catalog.meta 
    """
    if column is None:
        return csr_matrix((n_items, 0), dtype=np.float32)
    vectorizer = CountVectorizer(tokenizer=_split_commas, lowercase=False, binary=True)
    try:
        return vectorizer.fit_transform(column.fillna("").astype(str)).astype(np.float32).tocsr()
    except ValueError:                      # пустой словарь: во всех книгах поле пустое
        return csr_matrix((n_items, 0), dtype=np.float32)


class FeatureBuilder:

    def __init__(self, generator_names: list[str]):
        self.generator_names = list(generator_names)

    def fit(self, history: pd.DataFrame, catalog: Catalog) -> "FeatureBuilder":
        """Считает статистику книг по истории (положительные взаимодействия до отсечки окна)."""
        n_items = catalog.items_len
        meta = catalog.meta
        # Добавляет к истории индексы user_idx и item_idx
        enc = catalog.encode(history)
        # Получаем «сегодня» для окна — последний день истории
        cutoff = enc[DATE_COL].max()

        # Считаем количество читателей каждой книги
        readers = np.bincount(enc[ITEM_IDX_COL], minlength=n_items).astype(np.float64)
        # Количество читателей за последние RECENT_DAYS (популярность)
        recent = enc[enc[DATE_COL] > cutoff - pd.Timedelta(days=RECENT_DAYS)]
        recent_readers = np.bincount(recent[ITEM_IDX_COL], minlength=n_items)
        # Группируем рейтинги по книгам
        ratings = enc.groupby(ITEM_IDX_COL)[RATING_COL]

        # Считаем необходимые статистики 
        self.log_pop_ = np.log1p(readers)
        self.year_ = pd.to_numeric(meta["year"], errors="coerce").to_numpy(dtype=np.float64) \
            if "year" in meta else np.full(n_items, np.nan)
        # Составляем разреженные матрицы по признакам 
        self.genres_ = _bag_of_words(meta.get("genres"), n_items)
        self.authors_ = _bag_of_words(meta.get("authors"), n_items)

        self.item_stats_ = pd.DataFrame({
            "i_log_pop": self.log_pop_,
            "i_pop_pct": pd.Series(readers).rank(pct=True).to_numpy(), # Доля книг, менее популярная чем эта 
            "i_recent_share": recent_readers / (readers + 1.0),
            "i_mean_rating": ratings.mean().reindex(range(n_items)).to_numpy(),
            "i_n_ratings": np.log1p(ratings.count().reindex(range(n_items)).fillna(0).to_numpy()),
            "i_year": self.year_,
            "i_n_genres": self.genres_.getnnz(axis=1),
        })
        # Фиксируем порядок колонок 
        self.feature_names_ = self._feature_names()
        return self

    def _feature_names(self) -> list[str]:
        names = [f"{gen}_{part}" for gen in self.generator_names for part in ("score", "rank")]
        names += [N_SOURCES_COL, RRF_COL, "rrf_rank"]
        names += list(self.item_stats_.columns)
        names += ["u_n_items", "u_mean_log_pop"]
        names += ["genre_aff_sum", "genre_aff_max", "author_cnt", "pop_gap", "year_gap"]
        return names

    def transform(self, pool: pd.DataFrame, user_matrix: csr_matrix) -> pd.DataFrame:
        """ 
        Для каждой строки пользователь - кандидат собирает вектор признаков  
        pool — пул первого уровня (user_idx, item_idx, {gen}_score, {gen}_rank, n_sources, rrf);
        user_matrix — матрица истории (n_users, n_items): по ней считается профиль пользователя.
        Возвращает float32-таблицу с колонками feature_names_ в порядке строк pool.
        """
        pool = pool.reset_index(drop=True)
        users = pool[USER_IDX_COL].to_numpy(dtype=np.int64)
        items = pool[ITEM_IDX_COL].to_numpy(dtype=np.int64)
        uniq, inv = np.unique(users, return_inverse=True)       # inv: номер пользователя среди uniq у каждой строки

        X = pd.DataFrame(index=pool.index)
        # Переносим признаки из пула
        for gen in self.generator_names:
            X[f"{gen}_score"] = pool[f"{gen}_score"].to_numpy()
            X[f"{gen}_rank"] = pool[f"{gen}_rank"].to_numpy()
        X[N_SOURCES_COL] = pool[N_SOURCES_COL].to_numpy()
        X[RRF_COL] = pool[RRF_COL].to_numpy()
        X["rrf_rank"] = pool.groupby(USER_IDX_COL)[RRF_COL].rank(method="min", ascending=False).to_numpy()

        # Метаданные и статистика для каждой книги 
        for col in self.item_stats_.columns:
            X[col] = self.item_stats_[col].to_numpy()[items]

        # Пользователь: по его строке в матрице истории
        hist = (user_matrix[uniq] > 0).astype(np.float32).tocsr()
        # Количество прочитанных книг каждым пользователем 
        n_hist = np.asarray(hist.sum(axis=1)).ravel()
        safe = np.maximum(n_hist, 1.0) # Чтобы избегать деления на 0
        # Средний логарифм популярности его книг
        mean_pop = np.asarray(hist @ self.log_pop_).ravel() / safe
        # Средний год издания прочитанных книг
        has_year = ~np.isnan(self.year_)
        n_year = np.asarray(hist @ has_year.astype(np.float64)).ravel() # Кол-во прочитанных книг с указанным годом 
        mean_year = np.where(n_year > 0, np.asarray(hist @ np.where(has_year, self.year_, 0.0)).ravel()
                             / np.maximum(n_year, 1.0), np.nan)
        X["u_n_items"] = np.log1p(n_hist)[inv]
        X["u_mean_log_pop"] = mean_pop[inv]

        # Количство книг прочитанных каждого жанра, нормализованное по общему количество прочитанных книг 
        profile = csr_matrix(diags(1.0 / safe) @ (hist @ self.genres_)) if self.genres_.shape[1] else None
        # Количество книг прочитанных у каждого автора 
        user_authors = (hist @ self.authors_).tocsr() if self.authors_.shape[1] else None
        
        genre_sum = np.zeros(len(pool), dtype=np.float32)
        genre_max = np.zeros(len(pool), dtype=np.float32)
        author_cnt = np.zeros(len(pool), dtype=np.float32)
        for start in range(0, len(pool), CHUNK):
            part = slice(start, start + CHUNK)
            if profile is not None:
                both = profile[inv[part]].multiply(self.genres_[items[part]]).tocsr()
                genre_sum[part] = np.asarray(both.sum(axis=1)).ravel()
                genre_max[part] = both.max(axis=1).toarray().ravel()
            if user_authors is not None:
                shared = user_authors[inv[part]].multiply(self.authors_[items[part]]).tocsr()
                author_cnt[part] = np.asarray(shared.sum(axis=1)).ravel()
        X["genre_aff_sum"] = genre_sum
        X["genre_aff_max"] = genre_max
        X["author_cnt"] = author_cnt
        X["pop_gap"] = self.log_pop_[items] - mean_pop[inv]
        X["year_gap"] = np.abs(self.year_[items] - mean_year[inv])

        return X[self.feature_names_].astype(np.float32)