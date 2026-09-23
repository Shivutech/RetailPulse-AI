import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import joblib

print("=" * 60)
print("RETAILPULSE AI V2 - CHURN PREDICTION")
print("=" * 60)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("data/retail_data_v2.csv")
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

print("\nDataset loaded successfully")
print("Rows:", len(df))

# --------------------------------------------------
# 2. TEMPORAL SPLIT
# --------------------------------------------------

observation_end = pd.Timestamp("2026-03-31")
prediction_start = pd.Timestamp("2026-04-01")
prediction_end = pd.Timestamp("2026-06-30")

observation = df[df["Order_Date"] <= observation_end].copy()

future = df[
    (df["Order_Date"] >= prediction_start) &
    (df["Order_Date"] <= prediction_end)
].copy()

print("\nObservation Period:")
print("2024-01-01 to 2026-03-31")

print("\nPrediction Period:")
print("2026-04-01 to 2026-06-30")

# --------------------------------------------------
# 3. CUSTOMER FEATURES
# --------------------------------------------------

customer_features = observation.groupby("Customer_ID").agg(
    Last_Purchase_Date=("Order_Date", "max"),
    Frequency=("Order_ID", "count"),
    Monetary=("Revenue", "sum"),
    Total_Quantity=("Quantity", "sum"),
    Average_Order_Value=("Revenue", "mean"),
    Average_Rating=("Customer_Rating", "mean"),
    Average_Delivery_Days=("Delivery_Days", "mean")
).reset_index()

# Recency
customer_features["Recency"] = (
    observation_end - customer_features["Last_Purchase_Date"]
).dt.days

# --------------------------------------------------
# 4. CREATE CHURN TARGET
# --------------------------------------------------

future_customers = set(future["Customer_ID"].unique())

customer_features["Churn"] = (
    ~customer_features["Customer_ID"].isin(future_customers)
).astype(int)

print("\nCUSTOMER TARGET DISTRIBUTION")
print(customer_features["Churn"].value_counts())

# --------------------------------------------------
# 5. SELECT FEATURES
# --------------------------------------------------

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

# --------------------------------------------------
# 6. TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# --------------------------------------------------
# 7. TRAIN MODEL
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 8. PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# 9. MODEL PERFORMANCE
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# 10. FEATURE IMPORTANCE
# --------------------------------------------------

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(
    "Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance.to_string(index=False))

# --------------------------------------------------
# 11. PREDICT ALL CUSTOMERS
# --------------------------------------------------

customer_features["Churn_Probability"] = model.predict_proba(
    customer_features[features]
)[:, 1]

customer_features["Risk_Level"] = pd.cut(
    customer_features["Churn_Probability"],
    bins=[-0.01, 0.30, 0.60, 1.01],
    labels=["Low", "Medium", "High"]
)

# --------------------------------------------------
# 12. REVENUE AT RISK
# --------------------------------------------------

customer_features["Revenue_at_Risk"] = (
    customer_features["Monetary"] *
    customer_features["Churn_Probability"]
)

print("\nRISK DISTRIBUTION")
print(customer_features["Risk_Level"].value_counts())

print("\nREVENUE AT RISK")
print(
    customer_features.groupby("Risk_Level", observed=False)[
        "Revenue_at_Risk"
    ].sum()
)

# --------------------------------------------------
# 13. SAVE PREDICTIONS
# --------------------------------------------------

output_file = "data/customer_churn_predictions_v2.csv"

customer_features.to_csv(
    output_file,
    index=False
)

# --------------------------------------------------
# 14. SAVE MODEL
# --------------------------------------------------

model_file = "models/churn_model_v2.pkl"

joblib.dump(model, model_file)

print("\n" + "=" * 60)
print("FILES CREATED")
print("=" * 60)

print(f"\nPrediction file: {output_file}")
print(f"Model file     : {model_file}")

print("\nCHURN MODEL V2 COMPLETED")
print("=" * 60)