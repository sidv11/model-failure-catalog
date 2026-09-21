"""
features.py — turns raw line-item transactions into one row per ORDER,
with features a shop could actually know at the moment an order is placed
(before knowing whether it'll be cancelled).

This is the reusable "brain" of the project — the notebook just calls
build_order_features() once and gets a model-ready table back.
"""
import pandas as pd


def build_order_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse line-item-level transactions (one row per product in an order)
    into order-level features (one row per order/Invoice).

    Why collapse at all: a model that predicts "will THIS ORDER be
    cancelled" needs one row per order, not one row per product-in-an-order —
    otherwise a 5-item order would count 5 times toward the answer.

    IMPORTANT — a leak this project deliberately hit and fixed:
    In this dataset, a cancelled order's Quantity is recorded as NEGATIVE
    (that's literally how a cancellation is logged — 100% of cancelled line
    items have Quantity < 0). If we summed Quantity/Price as-is, the model
    would just learn "negative total = cancelled", which is the answer
    hiding inside the input, not a genuine risk signal. We use abs() so the
    features describe the SIZE of the order (how many items, how much it's
    worth), not its sign — the sign is the label in disguise.
    """
    data = df.copy()
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], format="%m/%d/%y %H:%M")
    data["IsCancelled"] = data["Invoice"].astype(str).str.startswith("C")
    data["LineValue"] = data["Quantity"].abs() * data["Price"]

    order_features = data.groupby("Invoice").agg(
        IsCancelled=("IsCancelled", "first"),
        Country=("Country", "first"),
        InvoiceDate=("InvoiceDate", "first"),
        num_unique_products=("StockCode", "nunique"),
        total_quantity=("Quantity", lambda s: s.abs().sum()),
        total_value=("LineValue", "sum"),
        avg_unit_price=("Price", "mean"),
    ).reset_index()

    order_features["DayOfWeek"] = order_features["InvoiceDate"].dt.day_name()
    order_features["Hour"] = order_features["InvoiceDate"].dt.hour
    order_features["IsWeekend"] = order_features["DayOfWeek"].isin(["Saturday", "Sunday"])

    return order_features
