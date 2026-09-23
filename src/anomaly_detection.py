import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("RETAILPULSE AI V2 - ANOMALY DETECTION")
print("=" * 60)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("data/retail_data_v2.csv")
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

print("\nDataset loaded successfully")
print("Rows:", len(df))

# --------------------------------------------------
# 2. MONTHLY BUSINESS DATA
# --------------------------------------------------

monthly = (
    df.set_index("Order_Date")
      .resample("MS")
      .agg(
          Revenue=("Revenue", "sum"),
          Profit=("Profit", "sum"),
          Quantity=("Quantity", "sum"),
          Orders=("Order_ID", "count"),
          Customers=("Customer_ID", "nunique"),
          Average_Order_Value=("Revenue", "mean"),
          Average_Rating=("Customer_Rating", "mean"),
          Average_Delivery_Days=("Delivery_Days", "mean")
      )
      .reset_index()
)

print("\nMonthly business data created")
print("Months:", len(monthly))

# --------------------------------------------------
# 3. FEATURES FOR ANOMALY DETECTION
# --------------------------------------------------

features = [
    "Revenue",
    "Profit",
    "Quantity",
    "Orders",
    "Customers",
    "Average_Order_Value",
    "Average_Rating",
    "Average_Delivery_Days"
]

X = monthly[features].copy()

# --------------------------------------------------
# 4. STANDARDIZE FEATURES
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# 5. ISOLATION FOREST
# --------------------------------------------------

model = IsolationForest(
    n_estimators=300,
    contamination=0.15,
    random_state=42
)

model.fit(X_scaled)

# --------------------------------------------------
# 6. DETECT ANOMALIES
# --------------------------------------------------

monthly["Anomaly"] = model.predict(X_scaled)

monthly["Anomaly_Score"] = model.decision_function(X_scaled)

monthly["Status"] = np.where(
    monthly["Anomaly"] == -1,
    "Anomaly",
    "Normal"
)

# --------------------------------------------------
# 7. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("ANOMALY DETECTION RESULTS")
print("=" * 60)

print("\nStatus Distribution:")
print(monthly["Status"].value_counts())

print("\nDetected Anomalies:")

anomalies = monthly[
    monthly["Status"] == "Anomaly"
].copy()

if len(anomalies) > 0:

    print(
        anomalies[
            [
                "Order_Date",
                "Revenue",
                "Profit",
                "Quantity",
                "Orders",
                "Customers",
                "Anomaly_Score"
            ]
        ].to_string(index=False)
    )

else:
    print("No anomalies detected.")

# --------------------------------------------------
# 8. ANOMALY SUMMARY
# --------------------------------------------------

if len(anomalies) > 0:

    highest_revenue_anomaly = anomalies.loc[
        anomalies["Revenue"].idxmax()
    ]

    lowest_revenue_anomaly = anomalies.loc[
        anomalies["Revenue"].idxmin()
    ]

    print("\n" + "=" * 60)
    print("ANOMALY SUMMARY")
    print("=" * 60)

    print(
        "\nHighest Revenue Anomaly:"
    )

    print(
        highest_revenue_anomaly[
            [
                "Order_Date",
                "Revenue",
                "Profit",
                "Quantity"
            ]
        ].to_string()
    )

    print(
        "\nLowest Revenue Anomaly:"
    )

    print(
        lowest_revenue_anomaly[
            [
                "Order_Date",
                "Revenue",
                "Profit",
                "Quantity"
            ]
        ].to_string()
    )

# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

output_file = "data/anomaly_detection_results.csv"

monthly.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 60)
print("FILES CREATED")
print("=" * 60)

print("\nAnomaly results:")
print(output_file)

print("\nANOMALY DETECTION COMPLETED")
print("=" * 60)