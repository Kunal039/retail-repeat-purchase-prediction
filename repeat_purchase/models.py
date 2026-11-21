"""Pre-processing pipeline and the candidate models."""
import warnings

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEED

# scikit-learn 1.9 deprecates SVC(probability=True) in favour of CalibratedClassifierCV.
# It still works until 1.11; we keep it so the SVM baseline matches the published results.
warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated",
                        category=FutureWarning)


def preprocessor() -> ColumnTransformer:
    """Median-impute + scale numeric columns; mode-impute + one-hot encode categorical ones."""
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUMERIC_FEATURES), ("cat", cat, CATEGORICAL_FEATURES)])


def make_pipeline(model) -> Pipeline:
    return Pipeline([("prep", preprocessor()), ("model", model)])


def candidate_models() -> dict:
    """Fresh, unfitted instances of the seven models compared in the project."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=15),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "Support Vector Machine": SVC(probability=True, random_state=SEED),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=600,
                                              early_stopping=True, random_state=SEED),
    }


RF_GRID = {"model__n_estimators": [200, 400], "model__max_depth": [6, 10, None],
           "model__min_samples_leaf": [1, 5, 10]}
LR_GRID = {"model__C": [0.01, 0.1, 1, 10], "model__class_weight": [None, "balanced"]}
