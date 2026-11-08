import numpy as np
import pandas as pd

import logging

from src.candidates.registry import build_generators
from src.candidates.merge import merge_candidates
from src.data.catalog import Catalog
from src.data.matrices import build_interaction_matrix

from src.data.schema import USER_IDX_COL, ITEM_IDX_COL

log = logging.getLogger(__name__)


class Candidates:
    generators : dict
    pool : pd.DataFrame

    def __init__(self, generators: list[str] | None = None, k_per_generator: int = 200, rrf_k: int = 100):
        '''
        generators - генераторы кандидатов 
        k_per_generator - количество книг, рекоммендуемых одним генератором 
        rrf_k - параметр для ранжирования кандидатов в пуле
        '''
        self.generator_names = generators
        self.k_per_generator = k_per_generator
        self.rrf_k = rrf_k

    def fit(self, history: pd.DataFrame, catalog: Catalog):
        # Создаём генераторы кандидатов
        self.generators = build_generators(self.generator_names)
        self.catalog = catalog

        # Строим для них матрицы взаимодействий + бинарную матрицу (понадобится для кандидатов)
        weightings = {weighting for _, weighting in self.generators.values()} | {"binary"}
        self.matrices = {w: build_interaction_matrix(history, catalog, weighting=w) for w in weightings}
        
        # Обучем генераторы
        for i, (name, (gen, weighting)) in enumerate(self.generators.items(), start=1):
            log.info("[%d/%d] %s (матрица: %s) — обучение", i, len(self.generators), name, weighting)
            gen.fit(self.matrices[weighting], self.catalog)
            
        return self
    
    def get_candidates(self, users, max_per_user: int | None = None):
        users = np.asarray(users)
        # Получаем рекомендации от каждого генератора
        recs = {}
        for name, (gen, _) in self.generators.items():
            recs[name] = gen.recommend(users, k=self.k_per_generator)
        # Объеденяем кандидатов в общий пул
        return merge_candidates(users, recs, max_per_user=max_per_user, rrf_k=self.rrf_k)
        

def users_with_target(target_matrix) -> np.ndarray:
    """Индексы пользователей, у которых в цели есть хотя бы одна книга."""
    return np.flatnonzero(np.diff(target_matrix.indptr) > 0)


def make_labels(pool: pd.DataFrame, target_matrix) -> np.ndarray:
    """1, если пара (пользователь, книга) из пула есть в цели, иначе 0."""
    hits = target_matrix[pool[USER_IDX_COL].to_numpy(), pool[ITEM_IDX_COL].to_numpy()]
    return (np.asarray(hits).ravel() > 0).astype(np.int8)