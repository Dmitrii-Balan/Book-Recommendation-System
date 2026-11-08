import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import logging
from dataclasses import dataclass

from src.candidates.registry import build_generators
from src.candidates.merge import merge_candidates
from src.data.prepare import PreparedData
from src.data.matrices import build_interaction_matrix

log = logging.getLogger(__name__)


@dataclass
class Candidates:
    generators : dict
    pul : pd.DataFrame

    @classmethod
    def generate_candidates(cls, data: PreparedData, target: str, n_recomendations: int = 200, generators: dict = None,
                            rrf_k: int = 100) -> 'Candidates':

        if target not in ['val', 'train']:
            raise ValueError(f"Значение target должно быть 'val' или 'train', а не {target}")
        
        # Создаём генераторы кандидатов
        if generators is None:
            generators = build_generators()

        # Узнаём обучаемую и предсказываемую части 
        history = data.splits.history(target) 
        target = build_interaction_matrix(data.splits.target(target), data.catalog, weighting="binary")

        # Оставляем только тех пользователей, для которых есть информация в target
        users = np.flatnonzero(np.diff(target.indptr) > 0)
        
        # Строим для них матрицы взаимодействий 
        matrices: dict[str, object] = {}
        for _, weighting in generators.values():
            if weighting not in matrices:
                matrices[weighting] = build_interaction_matrix(history, data.catalog, weighting=weighting)

        # Обучем генераторы и получаем рекоммендации 
        recs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        
        for i, (name, (gen, weighting)) in enumerate(generators.items(), start=1):
            
            log.info("[%d/%d] %s (матрица: %s) — обучение", i, len(generators), name, weighting)
            gen.fit(matrices[weighting], data.catalog)

            # Получаем рекомендуемые книги и их оценки
            items, scores = gen.recommend(users, k=n_recomendations)
            recs[name] = (items, scores)
            
        pool = merge_candidates(users, recs, rrf_k=rrf_k)

        return Candidates(generators, pool)
