"""Loading and screening-evaluation helpers for the UCI Autism Screening Adult (AQ-10) data."""

from __future__ import annotations

import csv
import hashlib
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ITEMS = [f"A{i}_Score" for i in range(1, 11)]
ARFF_NAME = "Autism-Adult-Data.arff"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parse_arff(text: str) -> pd.DataFrame:
    names = [m.strip("'") for m in re.findall(r"^@attribute\s+(\S+)", text, flags=re.I | re.M)]
    body = text.split("@data", 1)[1] if "@data" in text else text.split("@DATA", 1)[1]
    rows = [r for r in csv.reader(line for line in body.strip().splitlines() if line.strip())]
    return pd.DataFrame(rows, columns=names)


def load_aq10(path: str | Path) -> pd.DataFrame:
    """Load the UCI archive (.zip) or the .arff file and return a cleaned frame.

    Cleaning: '?' -> missing; ages above 100 (one record of 383) -> missing;
    yes/no fields -> 0/1; label `screen_positive` from `Class/ASD`.
    """
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            text = z.read(ARFF_NAME).decode("latin-1")
    else:
        text = path.read_text(encoding="latin-1")
    df = _parse_arff(text).replace("?", np.nan)
    df = df.rename(columns={"jundice": "jaundice", "austim": "family_autism", "contry_of_res": "country"})
    df[ITEMS] = df[ITEMS].astype(int)
    df["result"] = pd.to_numeric(df["result"])
    df["age"] = pd.to_numeric(df["age"])
    df.loc[df["age"] > 100, "age"] = np.nan
    for col in ["jaundice", "family_autism", "used_app_before"]:
        df[col] = (df[col] == "yes").astype(int)
    df["male"] = (df["gender"] == "m").astype(int)
    df["ethnicity"] = (df["ethnicity"].str.strip("' ").replace({"others": "Others"})
                       .fillna("Not reported"))
    df["screen_positive"] = (df["Class/ASD"] == "YES").astype(int)
    return df


def rank_items(items: pd.DataFrame, y) -> list[str]:
    """Order items by how much more often screen-positive people endorse them."""
    y = np.asarray(y)
    gap = items[y == 1].mean() - items[y == 0].mean()
    return list(gap.sort_values(ascending=False).index)


def choose_cutoff(scores, y, target_sensitivity: float = 0.95) -> int:
    """Highest integer cutoff t such that (score >= t) reaches the target sensitivity."""
    scores = np.asarray(scores)
    y = np.asarray(y)
    for t in range(int(scores.max()), int(scores.min()) - 1, -1):
        if ((scores >= t) & (y == 1)).sum() / (y == 1).sum() >= target_sensitivity:
            return t
    return int(scores.min())


def screening_metrics(y, flagged) -> dict[str, float]:
    y = np.asarray(y).astype(bool)
    f = np.asarray(flagged).astype(bool)
    tp, fn = (f & y).sum(), (~f & y).sum()
    tn, fp = (~f & ~y).sum(), (f & ~y).sum()
    div = lambda a, b: a / b if b else float("nan")
    return {"sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "PPV": div(tp, tp + fp), "NPV": div(tn, tn + fn)}


def short_form_cv(df: pd.DataFrame, k: int, n_splits=5, n_repeats=10, target_sensitivity=0.95, seed=0):
    """Cross-validated k-item short form.

    In every training fold: rank items, keep the top k, pick the cutoff that
    reaches the target sensitivity. Then apply both, unchanged, to the test fold.
    Returns one row of metrics per fold plus the items and cutoff chosen.
    """
    y = df["screen_positive"].values
    rows = []
    for r in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for fold, (train, test) in enumerate(cv.split(df, y)):
            tr, te = df.iloc[train], df.iloc[test]
            chosen = rank_items(tr[ITEMS], y[train])[:k]
            cutoff = choose_cutoff(tr[chosen].sum(axis=1), y[train], target_sensitivity)
            flagged = te[chosen].sum(axis=1) >= cutoff
            rows.append({"repeat": r, "fold": fold, "k": k, "cutoff": cutoff,
                         "items": ",".join(sorted(chosen, key=lambda s: int(s[1:-6]))),
                         **screening_metrics(y[test], flagged)})
    return pd.DataFrame(rows)
