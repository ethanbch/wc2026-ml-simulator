"""Time-series cross-validation helpers."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def build_time_series_splits(
    df: pd.DataFrame,
    date_col: str = "date",
    n_splits: int = 5,
    gap: int = 0,
) -> list[tuple[list[int], list[int]]]:
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    return [
        (train_idx.tolist(), val_idx.tolist())
        for train_idx, val_idx in tscv.split(df_sorted)
    ]


def split_by_date(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    train_df = df[df[date_col] < pd.to_datetime(train_end)]
    val_df = df[
        (df[date_col] >= pd.to_datetime(train_end))
        & (df[date_col] < pd.to_datetime(val_end))
    ]
    test_df = df[df[date_col] >= pd.to_datetime(val_end)]
    return train_df, val_df, test_df
