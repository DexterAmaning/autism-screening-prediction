from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aq10 import ITEMS, choose_cutoff, load_aq10, rank_items, screening_metrics, sha256, short_form_cv

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "autism_screening_adult_uci426.zip"
EXPECTED_SHA256 = "c57b29866c9715ff49ff4cce52228e6c460bdce3f615000c33b4d53df99d6d09"


def _toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    items = pd.DataFrame(rng.integers(0, 2, (n, 10)), columns=ITEMS)
    df = items.copy()
    df["screen_positive"] = (items.sum(axis=1) >= 6).astype(int)
    return df


def test_screening_metrics():
    m = screening_metrics([1, 1, 0, 0], [1, 0, 0, 1])
    assert m == {"sensitivity": 0.5, "specificity": 0.5, "PPV": 0.5, "NPV": 0.5}


def test_choose_cutoff_meets_target_on_training_data():
    df = _toy()
    scores = df[ITEMS].sum(axis=1)
    t = choose_cutoff(scores, df["screen_positive"], 0.95)
    sens = screening_metrics(df["screen_positive"], scores >= t)["sensitivity"]
    assert sens >= 0.95
    assert t == 6  # the toy label is exactly score >= 6


def test_rank_items_puts_informative_item_first():
    df = _toy()
    df["A3_Score"] = df["screen_positive"]  # make one item perfectly informative
    assert rank_items(df[ITEMS], df["screen_positive"])[0] == "A3_Score"


def test_short_form_full_length_recovers_rule():
    res = short_form_cv(_toy(), k=10, n_repeats=2)
    assert (res["sensitivity"] == 1).all() and (res["specificity"] == 1).all()
    assert set(res["cutoff"]) == {6}


@pytest.mark.skipif(not RAW.exists(), reason="UCI archive not present")
class TestRealData:
    def test_checksum(self):
        assert sha256(RAW) == EXPECTED_SHA256

    def test_label_is_the_aq10_rule(self):
        df = load_aq10(RAW)
        assert len(df) == 704
        assert (df["result"] == df[ITEMS].sum(axis=1)).all()
        assert (df["screen_positive"] == (df["result"] >= 7)).all()
