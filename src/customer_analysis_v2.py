import pandas as pd
import numpy as np

print("=" * 60)
print("RETAILPULSE AI V2 - CUSTOMER RFM ANALYSIS")
print("=" * 60)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("data/retail_data_v2.csv")
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

print("\nDataset loaded successfully")
print("Rows:", len(df))
print("Customers:", df["Customer_ID"].nunique())

# --------------------------------------------------
# 2. REFERENCE DATE
# --------------------------------------------------

reference_date = df["Order_Date"].max() + pd.Timedelta(days=1)

print("\nReference Date:", reference_date.date())

# --------------------------------------------------
# 3. RFM CALCULATION
# --------------------------------------------------

rfm = df.groupby("Customer_ID").agg(
    Last_Purchase_Date=("Order_Date", "max"),
    Frequency=("Order_ID", "nunique"),
    Monetary=("Revenue", "sum")
).reset_index()

# Recency
rfm["Recency"] = (
    reference_date - rfm["Last_Purchase_Date"]
).dt.days

# Average Order Value
rfm["Average_Order_Value"] = (
    rfm["Monetary"] / rfm["Frequency"]
)

# --------------------------------------------------
# 4. RFM SCORES
# --------------------------------------------------

# Recency:
# Lower recency is better
rfm["R_Score"] = pd.qcut(
    rfm["Recency"],
    5,
    labels=[5, 4, 3, 2, 1],
    duplicates="drop"
).astype(int)

# Frequency:
# Higher frequency is better
rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

# Monetary:
# Higher spending is better
rfm["M_Score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

# Combined score
rfm["RFM_Score"] = (
    rfm["R_Score"].astype(str)
    + rfm["F_Score"].astype(str)
    + rfm["M_Score"].astype(str)
)

# Numerical RFM score
rfm["RFM_Total"] = (
    rfm["R_Score"]
    + rfm["F_Score"]
    + rfm["M_Score"]
)

# --------------------------------------------------
# 5. CUSTOMER SEGMENTS
# --------------------------------------------------

def segment_customer(row):

    r = row["R_Score"]
    f = row["F_Score"]
    m = row["M_Score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"

    elif r >= 3 and f >= 4:
        return "Loyal Customers"

    elif r >= 4 and f >= 2:
        return "Potential Loyalists"

    elif r <= 2 and f >= 3:
        return "At Risk"

    elif r <= 2 and f <= 2 and m <= 2:
        return "Lost Customers"

    else:
        return "Others"


rfm["Segment"] = rfm.apply(
    segment_customer,
    axis=1
)

# --------------------------------------------------
# 6. DISPLAY SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("CUSTOMER SEGMENT DISTRIBUTION")
print("=" * 60)

segment_summary = (
    rfm.groupby("Segment")
    .agg(
        Customers=("Customer_ID", "count"),
        Revenue=("Monetary", "sum"),
        Average_Order_Value=("Average_Order_Value", "mean"),
        Average_Frequency=("Frequency", "mean"),
        Average_Recency=("Recency", "mean")
    )
    .sort_values(
        "Revenue",
        ascending=False
    )
)

print(
    segment_summary.to_string()
)

# --------------------------------------------------
# 7. TOP CUSTOMERS
# --------------------------------------------------

print("\n" + "=" * 60)
print("TOP 10 CUSTOMERS BY REVENUE")
print("=" * 60)

top_customers = (
    rfm.sort_values(
        "Monetary",
        ascending=False
    )
    .head(10)
)

print(
    top_customers[
        [
            "Customer_ID",
            "Segment",
            "Recency",
            "Frequency",
            "Monetary",
            "Average_Order_Value"
        ]
    ].to_string(index=False)
)

# --------------------------------------------------
# 8. SEGMENT PERCENTAGE
# --------------------------------------------------

segment_percentage = (
    rfm["Segment"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n" + "=" * 60)
print("SEGMENT PERCENTAGE")
print("=" * 60)

print(segment_percentage)

# --------------------------------------------------
# 9. SAVE RFM DATA
# --------------------------------------------------

output_file = "data/customer_rfm_v2.csv"

rfm.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 60)
print("FILE CREATED")
print("=" * 60)

print("\nRFM file:", output_file)

print("\nRFM ANALYSIS V2 COMPLETED")
print("=" * 60)