from pathlib import Path

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from repeat_purchase import config
from repeat_purchase.data import CustomerDataGenerator, clean, engineer
from repeat_purchase.models import make_pipeline

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(scope="module")
def temp_model(tmp_path_factory):
    """Train a small model in a temp dir so the test never touches models/ on disk."""
    df = engineer(clean(CustomerDataGenerator(800).generate())[0])
    model = make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(df[config.FEATURES], df[config.TARGET])
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(model, path)
    return path


@pytest.fixture
def app(temp_model, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", temp_model)
    return AppTest.from_file(APP, default_timeout=60).run()


def test_app_renders_without_errors(app):
    assert not app.exception
    assert not app.error
    assert [t.label for t in app.tabs] == ["Score a customer", "Score a CSV", "Model performance"]


def test_single_customer_prediction_is_a_percentage(app):
    metric = next(m for m in app.metric if m.label.startswith("Repeat chance"))
    assert metric.value.endswith("%")
    assert 0 <= float(metric.value.rstrip("%")) <= 100


def test_batch_tab_scores_sample_customers(app):
    scored = next(m for m in app.metric if m.label == "Customers scored")
    assert scored.value == "5"


def test_threshold_changes_recommendation(app):
    app.sidebar.slider[0].set_value(0.95).run()
    assert not app.exception
    likely = next(m for m in app.metric if m.label == "Likely to return")
    assert likely.value == "0%"
