import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/retail_data.csv")

df["Order_Date"] = pd.to_datetime(df["Order_Date"])

print("=" * 60)
print("RETAILPULSE AI - PROPER CHURN PREDICTION")
print("=" * 60)


# ============================================================
# DEFINE OBSERVATION AND PREDICTION WINDOWS
# ============================================================

observation_end = pd.Timestamp("2026-03-31")
prediction_start = pd.Timestamp("2026-04-01")
prediction_end = pd.Timestamp("2026-06-30")

print("\nOBSERVATION PERIOD")
print("2024-01-01 to 2026-03-31")

print("\nPREDICTION PERIOD")
print("2026-04-01 to 2026-06-30")


# ============================================================
# HISTORICAL DATA
# ============================================================

historical = df[
    df["Order_Date"] <= observation_end
].copy()

future = df[
    (df["Order_Date"] >= prediction_start) &
    (df["Order_Date"] <= prediction_end)
].copy()


# ============================================================
# CREATE CUSTOMER FEATURES
# ============================================================

analysis_date = observation_end + pd.Timedelta(days=1)

customer_features = (
    historical
    .groupby("Customer_ID")
    .agg(
        Recency=(
            "Order_Date",
            lambda x: (analysis_date - x.max()).days
        ),
        Frequency=(
            "Order_ID",
            "nunique"
        ),
        Monetary=(
            "Revenue",
            "sum"
        ),
        Total_Quantity=(
            "Quantity",
            "sum"
        ),
        Average_Order_Value=(
            "Revenue",
            "mean"
        ),
        Average_Rating=(
            "Customer_Rating",
            "mean"
        ),
        Average_Delivery_Days=(
            "Delivery_Days",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# FUTURE PURCHASE BEHAVIOR
# ============================================================

future_customers = set(
    future["Customer_ID"].unique()
)

customer_features["Churn"] = (
    ~customer_features["Customer_ID"]
    .isin(future_customers)
).astype(int)


# ============================================================
# REMOVE CUSTOMERS WITH INSUFFICIENT HISTORY
# ============================================================

customer_features = customer_features[
    customer_features["Frequency"] >= 2
].copy()


# ============================================================
# CHECK TARGET DISTRIBUTION
# ============================================================

print("\nCUSTOMER DATASET")
print(
    f"Customers available: {len(customer_features):,}"
)

print("\nCHURN DISTRIBUTION")

print(
    customer_features["Churn"]
    .value_counts()
    .rename(
        {
            0: "Active",
            1: "Churned"
        }
    )
)


# ============================================================
# FEATURES
# ============================================================

features = [
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Quantity",
    "Average_Order_Value",
    "Average_Rating",
    "Average_Delivery_Days"
]

X = customer_features[features]

y = customer_features["Churn"]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAINING DATA")
print(f"Training samples: {len(X_train):,}")

print("\nTESTING DATA")
print(f"Testing samples : {len(X_test):,}")


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\nMODEL PERFORMANCE")

print(f"Accuracy : {accuracy:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred
    )
)


print("\nCONFUSION MATRIX")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame(
    {
        "Feature": features,
        "Importance": model.feature_importances_
    }
)

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")

print(importance)


# ============================================================
# PREDICT CUSTOMER CHURN PROBABILITY
# ============================================================

customer_features["Churn_Probability"] = (
    model.predict_proba(
        customer_features[features]
    )[:, 1]
)


# ============================================================
# RISK LEVEL
# ============================================================

def risk_level(probability):

    if probability >= 0.70:
        return "High Risk"

    elif probability >= 0.40:
        return "Medium Risk"

    else:
        return "Low Risk"


customer_features["Risk_Level"] = (
    customer_features["Churn_Probability"]
    .apply(risk_level)
)


# ============================================================
# REVENUE AT RISK
# ============================================================

customer_features["Revenue_At_Risk"] = (
    customer_features["Monetary"] *
    customer_features["Churn_Probability"]
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

customer_features.to_csv(
    "data/customer_churn_predictions.csv",
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/churn_model.pkl"
)


# ============================================================
# BUSINESS SUMMARY
# ============================================================

print("\nCUSTOMER RISK SUMMARY")

risk_summary = (
    customer_features["Risk_Level"]
    .value_counts()
)

print(risk_summary)


# ============================================================
# REVENUE AT RISK
# ============================================================

high_risk = customer_features[
    customer_features["Risk_Level"] == "High Risk"
]

medium_risk = customer_features[
    customer_features["Risk_Level"] == "Medium Risk"
]

print("\nHIGH RISK CUSTOMERS")

print(
    f"Customers      : {len(high_risk):,}"
)

print(
    f"Revenue at Risk: "
    f"₹{high_risk['Revenue_At_Risk'].sum():,.2f}"
)


print("\nMEDIUM RISK CUSTOMERS")

print(
    f"Customers      : {len(medium_risk):,}"
)

print(
    f"Revenue at Risk: "
    f"₹{medium_risk['Revenue_At_Risk'].sum():,.2f}"
)


# ============================================================
# TOP HIGH-RISK CUSTOMERS
# ============================================================

print("\nTOP 10 HIGH-RISK CUSTOMERS")

top_risk = (
    high_risk
    .sort_values(
        "Churn_Probability",
        ascending=False
    )
    .head(10)
)

print(
    top_risk[
        [
            "Customer_ID",
            "Recency",
            "Frequency",
            "Monetary",
            "Churn_Probability",
            "Risk_Level",
            "Revenue_At_Risk"
        ]
    ]
)


# ============================================================
# FILES
# ============================================================

print("\nFILES CREATED")

print("✓ data/customer_churn_predictions.csv")
print("✓ models/churn_model.pkl")

print("\n" + "=" * 60)
print("PROPER CHURN MODEL COMPLETED")
print("=" * 60)