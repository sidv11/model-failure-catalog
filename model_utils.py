"""
model_utils.py — rebuilds the exact Day 8 order-cancellation model (same
features, same LightGBM + SMOTE pipeline) so this project can stress-test
it without duplicating that logic by hand in the notebook.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

NUM_FEATURES = ["num_unique_products", "total_quantity", "total_value", "avg_unit_price", "Hour"]
CAT_FEATURES = ["Country", "DayOfWeek"]


def prepare_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Buckets rare countries into 'Other', same as Day 8, so unseen/rare
    countries don't each get their own sparse one-hot column."""
    data = orders.copy()
    top_countries = data["Country"].value_counts().head(10).index
    data["Country"] = data["Country"].where(data["Country"].isin(top_countries), "Other")
    return data


def train_model(orders: pd.DataFrame, random_state: int = 42):
    """
    Trains the exact same model as Day 8 and returns everything needed to
    stress-test it: the fitted preprocessor, the fitted model, and the
    train/test split (with the ORIGINAL readable feature values, not the
    encoded ones, so failures can be inspected in plain terms).
    """
    orders = prepare_orders(orders)
    X = orders[NUM_FEATURES + CAT_FEATURES]
    y = orders["IsCancelled"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])
    X_train_prep = preprocess.fit_transform(X_train)
    X_test_prep = preprocess.transform(X_test)

    smote = SMOTE(random_state=random_state)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_prep, y_train)

    model = lgb.LGBMClassifier(random_state=random_state, verbose=-1)
    model.fit(X_train_sm, y_train_sm)

    return {
        "preprocess": preprocess,
        "model": model,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
    }


def predict(bundle: dict, X_new: pd.DataFrame):
    """Runs new/synthetic rows through the fitted preprocessor + model."""
    X_prep = bundle["preprocess"].transform(X_new[NUM_FEATURES + CAT_FEATURES])
    proba = bundle["model"].predict_proba(X_prep)[:, 1]
    pred = bundle["model"].predict(X_prep)
    return pred, proba
