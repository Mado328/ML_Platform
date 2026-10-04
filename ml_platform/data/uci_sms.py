import csv
from pathlib import Path

import pandas as pd


VALID_LABELS = {"ham", "spam"}


def load_uci_sms(path: str | Path) -> pd.DataFrame:
    """Load and validate the UCI SMS Spam Collection."""
    try:
        data = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["label", "text"],
            dtype={"label": "string", "text": "string"},
            encoding="utf-8-sig",
            keep_default_na=False,
            quoting=csv.QUOTE_NONE,
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The dataset is empty") from exc

    if data.empty:
        raise ValueError("The dataset is empty")

    labels = set(data["label"].unique())
    unexpected_labels = labels - VALID_LABELS
    missing_labels = VALID_LABELS - labels

    if unexpected_labels or missing_labels:
        raise ValueError(
            f"Invalid labels: unexpected={unexpected_labels}, missing={missing_labels}"
        )

    empty_text_rows = data.index[data["text"].str.strip().eq("")].tolist()
    if empty_text_rows:
        raise ValueError(f"Empty message text at rows: {empty_text_rows[:10]}")

    return data
