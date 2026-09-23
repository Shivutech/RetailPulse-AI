import pandas as pd

# ============================================================
# LOAD V2 DATA
# ============================================================

df = pd.read_csv("data/retail_data_v2.csv")

df["Order_Date"] = pd.to_datetime(df["Order_Date"])

print("=" * 60)
print("RETAILPULSE AI V2 - DATA QUALITY REPORT")
print("=" * 60)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n1. DATASET SHAPE")

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# COLUMNS
# ============================================================

print("\n2. COLUMNS")

for column in df.columns:
    print(f"✓ {column}")


# ============================================================
# MISSING VALUES
# ============================================================

print("\n3. MISSING VALUES")

missing = df.isnull().sum()

print(missing)


# ============================================================
# DUPLICATES
# ============================================================

print("\n4. DUPLICATE ROWS")

print(
    f"Duplicates: {df.duplicated().sum()}"
)


# ============================================================
# UNIQUE CUSTOMERS
# ============================================================

print("\n5. UNIQUE CUSTOMERS")

print(
    f"Customers: {df['Customer_ID'].nunique():,}"
)


# ============================================================
# DATE RANGE
# ============================================================

print("\n6. DATE RANGE")

print(
    f"From: {df['Order_Date'].min().date()}"
)

print(
    f"To  : {df['Order_Date'].max().date()}"
)


# ============================================================
# BUSINESS KPIs
# ============================================================

revenue = df["Revenue"].sum()
profit = df["Profit"].sum()
orders = df["Order_ID"].nunique()
customers = df["Customer_ID"].nunique()
units = df["Quantity"].sum()

aov = revenue / orders

profit_margin = (
    profit / revenue
) * 100

print("\n7. BUSINESS KPIs")

print(
    f"Revenue        : ₹{revenue:,.2f}"
)

print(
    f"Profit         : ₹{profit:,.2f}"
)

print(
    f"Orders         : {orders:,}"
)

print(
    f"Customers      : {customers:,}"
)

print(
    f"Units Sold     : {units:,}"
)

print(
    f"Average Order  : ₹{aov:,.2f}"
)

print(
    f"Profit Margin  : {profit_margin:.2f}%"
)


# ============================================================
# CUSTOMER PROFILE
# ============================================================

print("\n8. CUSTOMER PROFILE")

profile_summary = (
    df.groupby("Customer_Profile")
    .agg(
        Customers=("Customer_ID", "nunique"),
        Orders=("Order_ID", "nunique"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum")
    )
)

profile_summary["Profit_Margin"] = (
    profile_summary["Profit"] /
    profile_summary["Revenue"] * 100
)

print(profile_summary)


# ============================================================
# CATEGORY
# ============================================================

print("\n9. CATEGORY PERFORMANCE")

category = (
    df.groupby("Category")
    .agg(
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum"),
        Orders=("Order_ID", "nunique")
    )
    .sort_values(
        "Revenue",
        ascending=False
    )
)

category["Profit_Margin"] = (
    category["Profit"] /
    category["Revenue"] * 100
)

print(category)


# ============================================================
# REGION
# ============================================================

print("\n10. REGION PERFORMANCE")

region = (
    df.groupby("Region")
    .agg(
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum"),
        Orders=("Order_ID", "nunique")
    )
    .sort_values(
        "Revenue",
        ascending=False
    )
)

print(region)


# ============================================================
# CUSTOMER TYPE
# ============================================================

print("\n11. CUSTOMER TYPE")

customer_type = (
    df.groupby("Customer_Type")
    .agg(
        Customers=("Customer_ID", "nunique"),
        Orders=("Order_ID", "nunique"),
        Revenue=("Revenue", "sum")
    )
)

print(customer_type)


# ============================================================
# TOP PRODUCTS
# ============================================================

print("\n12. TOP PRODUCTS")

top_products = (
    df.groupby("Product")["Revenue"]
    .sum()
    .sort_values(
        ascending=False
    )
    .head(10)
)

print(top_products)


# ============================================================
# CUSTOMER PROFILE CHECK
# ============================================================

print("\n13. PROFILE DISTRIBUTION")

profile_counts = (
    df.groupby("Customer_Profile")
    ["Customer_ID"]
    .nunique()
)

print(profile_counts)


print("\n" + "=" * 60)
print("V2 DATA QUALITY ANALYSIS COMPLETED")
print("=" * 60)