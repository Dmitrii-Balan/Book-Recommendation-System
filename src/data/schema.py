"""
Единая схема данных: имена колонок, одинаковые во всём проекте.

В сыром items.csv книга называется `id`, в interactions.csv — `item_id`.
При загрузке (loading.py) всё приводится к `item_id`, дальше везде только он.
"""

USER_COL = "user_id"
ITEM_COL = "item_id"
PROGRESS_COL = "progress"
RATING_COL = "rating"
DATE_COL = "start_date"

INTERACTION_COLUMNS = [USER_COL, ITEM_COL, PROGRESS_COL, RATING_COL, DATE_COL]

TEXT_COLUMNS = ["title", "genres", "authors"]
YEAR_COL = "year"
ITEM_COLUMNS = [ITEM_COL, *TEXT_COLUMNS, YEAR_COL]

# Индексы после кодирования через Catalog (0..N-1)
USER_IDX_COL = "user_idx"
ITEM_IDX_COL = "item_idx"
