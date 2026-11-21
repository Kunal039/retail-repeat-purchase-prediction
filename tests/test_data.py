import pandas as pd

from repeat_purchase.config import FEATURES, TARGET
from repeat_purchase.data import CustomerDataGenerator, clean, engineer


def test_generator_is_reproducible():
    a = CustomerDataGenerator(500, seed=7).generate()
    b = CustomerDataGenerator(500, seed=7).generate()
    pd.testing.assert_frame_equal(a, b)


def test_generator_shape_and_target():
    df = CustomerDataGenerator(500).generate()
    assert len(df) == 560  # 500 customers + 60 injected duplicates
    assert set(df[TARGET].unique()) <= {0, 1}


def test_clean_removes_duplicates_and_caps_outliers():
    # full-size dataset: the 15 injected outliers are < 1 % of rows, so 99th-percentile capping removes them
    raw = CustomerDataGenerator(5000).generate()
    df, log = clean(raw)
    assert df["customer_id"].is_unique
    assert log["duplicates_removed"] == 60
    assert df["avg_basket_value"].max() < 250000
    assert log["clean_rows"] == 5000


def test_engineer_adds_expected_features():
    df = engineer(clean(CustomerDataGenerator(300).generate())[0])
    for col in FEATURES:
        assert col in df.columns
    assert set(df["recency_bucket"].unique()) <= {"Recent", "Lapsing", "Dormant"}
    assert (df["engagement_score"].dropna() >= 0).all()
