import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/retail_data.csv")

# Convert date
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

print("=" * 60)
print("RETAILPULSE AI - DATA QUALITY REPORT")
print("=" * 60)

# 1. Dataset shape
print("\n1. DATASET SHAPE")
print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

# 2. Column names
print("\n2. COLUMNS")
print(df.columns.tolist())

# 3. Data types
print("\n3. DATA TYPES")
print(df.dtypes)

# 4. Missing values
print("\n4. MISSING VALUES")
missing = df.isnull().sum()
print(missing)

# 5. Duplicate rows
print("\n5. DUPLICATE ROWS")
print(f"Duplicates: {df.duplicated().sum()}")

# 6. Unique customers
print("\n6. UNIQUE CUSTOMERS")
print(f"Customers: {df['Customer_ID'].nunique():,}")

# 7. Date range
print("\n7. DATE RANGE")
print(f"From: {df['Order_Date'].min().date()}")
print(f"To  : {df['Order_Date'].max().date()}")

# 8. Business metrics
print("\n8. BUSINESS METRICS")

total_revenue = df["Revenue"].sum()
total_profit = df["Profit"].sum()
total_orders = df["Order_ID"].nunique()
total_customers = df["Customer_ID"].nunique()
total_quantity = df["Quantity"].sum()
average_order_value = total_revenue / total_orders

print(f"Total Revenue       : ₹{total_revenue:,.2f}")
print(f"Total Profit        : ₹{total_profit:,.2f}")
print(f"Total Orders        : {total_orders:,}")
print(f"Total Customers     : {total_customers:,}")
print(f"Total Units Sold    : {total_quantity:,}")
print(f"Average Order Value : ₹{average_order_value:,.2f}")

# 9. Profit margin
profit_margin = (total_profit / total_revenue) * 100

print(f"Profit Margin       : {profit_margin:.2f}%")

# 10. Category analysis
print("\n9. CATEGORY PERFORMANCE")

category_analysis = (
    df.groupby("Category")
    .agg(
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum"),
        Orders=("Order_ID", "nunique"),
        Quantity=("Quantity", "sum")
    )
    .sort_values("Revenue", ascending=False)
)

category_analysis["Profit_Margin_%"] = (
    category_analysis["Profit"] /
    category_analysis["Revenue"] * 100
)

print(category_analysis)

# 11. Region analysis
print("\n10. REGION PERFORMANCE")

region_analysis = (
    df.groupby("Region")
    .agg(
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum"),
        Orders=("Order_ID", "nunique")
    )
    .sort_values("Revenue", ascending=False)
)

print(region_analysis)

# 12. Customer type
print("\n11. CUSTOMER TYPE")

customer_type = (
    df.groupby("Customer_Type")
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("Order_ID", "nunique"),
        Customers=("Customer_ID", "nunique")
    )
)

print(customer_type)

# 13. Payment method
print("\n12. PAYMENT METHOD")

payment_analysis = (
    df.groupby("Payment_Mode")
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("Order_ID", "nunique")
    )
    .sort_values("Revenue", ascending=False)
)

print(payment_analysis)

# 14. Top products
print("\n13. TOP 10 PRODUCTS BY REVENUE")

top_products = (
    df.groupby("Product")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print(top_products)

# 15. Statistical summary
print("\n14. NUMERICAL SUMMARY")

print(
    df[
        [
            "Quantity",
            "Unit_Price",
            "Discount",
            "Revenue",
            "Cost",
            "Profit",
            "Customer_Rating",
            "Delivery_Days"
        ]
    ].describe()
)

print("\n" + "=" * 60)
print("DATA QUALITY ANALYSIS COMPLETED")
print("=" * 60)