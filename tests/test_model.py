import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from repeat_purchase.config import DATA_DIR, FEATURES, TARGET
from repeat_purchase.data import CustomerDataGenerator, clean, engineer
from repeat_purchase.models import candidate_models, make_pipeline
from repeat_purchase.predict import score


@pytest.fixture(scope="module")
def small_model():
    df = engineer(clean(CustomerDataGenerator(800).generate())[0])
    model = make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))
    return model.fit(df[FEATURES], df[TARGET])


def test_seven_candidate_models():
    assert len(candidate_models()) == 7


def test_pipeline_handles_missing_values(small_model):
    df = engineer(clean(CustomerDataGenerator(200, seed=1).generate())[0])
    assert df[FEATURES].isna().any().any()  # data really has gaps
    prob = small_model.predict_proba(df[FEATURES])[:, 1]
    assert ((prob >= 0) & (prob <= 1)).all()


def test_score_sample_customers(small_model):
    customers = pd.read_csv(DATA_DIR / "sample_customers.csv")
    out = score(customers, model=small_model)
    assert len(out) == len(customers)
    assert out["repeat_probability"].between(0, 1).all()
    assert set(out["recommended_action"]) <= {"Loyalty reward / upsell", "Win-back discount campaign"}
    # an engaged, recent Gold/Silver shopper should outrank a dormant, unhappy one
    p = out.set_index("customer_id")["repeat_probability"]
    assert p["N001"] > p["N002"]


def test_score_rejects_missing_columns(small_model):
    with pytest.raises(ValueError, match="missing required columns"):
        score(pd.DataFrame({"age": [30]}), model=small_model)
