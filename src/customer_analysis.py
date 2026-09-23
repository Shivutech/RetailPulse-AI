import pandas as pd
import numpy as np

# ---------------------------------------
# LOAD DATA
# ---------------------------------------

df = pd.read_csv("data/retail_data.csv")

df["Order_Date"] = pd.to_datetime(df["Order_Date"])


# ---------------------------------------
# ANALYSIS DATE
# ---------------------------------------

analysis_date = df["Order_Date"].max() + pd.Timedelta(days=1)


# ---------------------------------------
# RFM CALCULATION
# ---------------------------------------

rfm = (
    df.groupby("Customer_ID")
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
        )
    )
    .reset_index()
)


# ---------------------------------------
# RFM SCORES
# ---------------------------------------

rfm["R_Score"] = pd.qcut(
    rfm["Recency"],
    5,
    labels=[5, 4, 3, 2, 1],
    duplicates="drop"
).astype(int)

rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["M_Score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)


# ---------------------------------------
# RFM TOTAL SCORE
# ---------------------------------------

rfm["RFM_Score"] = (
    rfm["R_Score"] +
    rfm["F_Score"] +
    rfm["M_Score"]
)


# ---------------------------------------
# CUSTOMER SEGMENTATION
# ---------------------------------------

def assign_segment(row):

    r = row["R_Score"]
    f = row["F_Score"]
    m = row["M_Score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"

    elif r >= 3 and f >= 4:
        return "Loyal Customers"

    elif r >= 4 and f >= 2:
        return "Potential Loyalists"

    elif r <= 2 and f >= 3 and m >= 3:
        return "At Risk"

    elif r <= 2 and f <= 2:
        return "Lost Customers"

    else:
        return "Others"


rfm["Segment"] = rfm.apply(
    assign_segment,
    axis=1
)


# ---------------------------------------
# CUSTOMER VALUE
# ---------------------------------------

rfm["Average_Order_Value"] = (
    rfm["Monetary"] /
    rfm["Frequency"]
)


# ---------------------------------------
# SAVE RFM DATA
# ---------------------------------------

rfm.to_csv(
    "data/customer_rfm.csv",
    index=False
)


# ---------------------------------------
# SUMMARY
# ---------------------------------------

print("=" * 60)
print("RETAILPULSE AI - CUSTOMER RFM ANALYSIS")
print("=" * 60)

print("\nTOTAL CUSTOMERS")
print(f"{len(rfm):,}")

print("\nRFM DATA PREVIEW")
print(
    rfm[
        [
            "Customer_ID",
            "Recency",
            "Frequency",
            "Monetary",
            "RFM_Score",
            "Segment"
        ]
    ].head(10)
)

print("\nCUSTOMER SEGMENTS")

segment_summary = (
    rfm.groupby("Segment")
    .agg(
        Customers=("Customer_ID", "count"),
        Revenue=("Monetary", "sum"),
        Average_Recency=("Recency", "mean"),
        Average_Frequency=("Frequency", "mean")
    )
    .sort_values(
        "Revenue",
        ascending=False
    )
)

print(segment_summary)

print("\nTOP 10 CUSTOMERS BY MONETARY VALUE")

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
            "Recency",
            "Frequency",
            "Monetary",
            "RFM_Score",
            "Segment"
        ]
    ]
)

print("\nRFM DATA SAVED TO:")
print("data/customer_rfm.csv")

print("\n" + "=" * 60)
print("CUSTOMER ANALYSIS COMPLETED")
print("=" * 60)