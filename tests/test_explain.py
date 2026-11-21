import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from repeat_purchase.config import DATA_DIR, FEATURES, INPUT_COLUMNS, TARGET
from repeat_purchase.data import CustomerDataGenerator, clean, engineer
from repeat_purchase.explain import contributions
from repeat_purchase.models import make_pipeline


@pytest.fixture(scope="module")
def lr_model():
    df = engineer(clean(CustomerDataGenerator(800).generate())[0])
    return make_pipeline(LogisticRegression(max_iter=1000)).fit(df[FEATURES], df[TARGET])


@pytest.fixture
def customer():
    return pd.read_csv(DATA_DIR / "sample_customers.csv").iloc[[0]]


def test_contributions_reconstruct_probability(lr_model, customer):
    contrib = contributions(lr_model, customer)
    logit = contrib["contribution"].sum() + lr_model.named_steps["model"].intercept_[0]
    expected = lr_model.predict_proba(engineer(customer[INPUT_COLUMNS]))[0, 1]
    assert 1 / (1 + np.exp(-logit)) == pytest.approx(expected)


def test_one_hot_columns_are_grouped(lr_model, customer):
    contrib = contributions(lr_model, customer)
    assert contrib["feature"].is_unique
    assert {"membership", "channel", "recency_bucket"} <= set(contrib["feature"])
    assert contrib["contribution"].abs().is_monotonic_decreasing


def test_contributions_require_single_row(lr_model):
    two = pd.read_csv(DATA_DIR / "sample_customers.csv").iloc[:2]
    with pytest.raises(ValueError):
        contributions(lr_model, two)
