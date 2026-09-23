import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv("data/retail_data.csv")

df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Create output directory
output_dir = Path("screenshots/eda")
output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# STYLE
# -----------------------------

sns.set_theme(style="whitegrid")

# -----------------------------
# 1. MONTHLY REVENUE
# -----------------------------

monthly_revenue = (
    df.groupby(df["Order_Date"].dt.to_period("M"))["Revenue"]
    .sum()
)

plt.figure(figsize=(12, 6))

monthly_revenue.plot()

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue (₹)")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    output_dir / "monthly_revenue.png",
    dpi=150
)

plt.close()

# -----------------------------
# 2. MONTHLY PROFIT
# -----------------------------

monthly_profit = (
    df.groupby(df["Order_Date"].dt.to_period("M"))["Profit"]
    .sum()
)

plt.figure(figsize=(12, 6))

monthly_profit.plot()

plt.title("Monthly Profit Trend")
plt.xlabel("Month")
plt.ylabel("Profit (₹)")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    output_dir / "monthly_profit.png",
    dpi=150
)

plt.close()

# -----------------------------
# 3. CATEGORY REVENUE
# -----------------------------

category_revenue = (
    df.groupby("Category")["Revenue"]
    .sum()
    .sort_values()
)

plt.figure(figsize=(10, 6))

category_revenue.plot(kind="barh")

plt.title("Revenue by Category")
plt.xlabel("Revenue (₹)")
plt.ylabel("Category")

plt.tight_layout()

plt.savefig(
    output_dir / "category_revenue.png",
    dpi=150
)

plt.close()

# -----------------------------
# 4. CATEGORY PROFIT
# -----------------------------

category_profit = (
    df.groupby("Category")["Profit"]
    .sum()
    .sort_values()
)

plt.figure(figsize=(10, 6))

category_profit.plot(kind="barh")

plt.title("Profit by Category")
plt.xlabel("Profit (₹)")
plt.ylabel("Category")

plt.tight_layout()

plt.savefig(
    output_dir / "category_profit.png",
    dpi=150
)

plt.close()

# -----------------------------
# 5. REGIONAL REVENUE
# -----------------------------

region_revenue = (
    df.groupby("Region")["Revenue"]
    .sum()
    .sort_values()
)

plt.figure(figsize=(9, 6))

region_revenue.plot(kind="bar")

plt.title("Revenue by Region")
plt.xlabel("Region")
plt.ylabel("Revenue (₹)")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    output_dir / "region_revenue.png",
    dpi=150
)

plt.close()

# -----------------------------
# 6. CUSTOMER TYPE
# -----------------------------

customer_revenue = (
    df.groupby("Customer_Type")["Revenue"]
    .sum()
)

plt.figure(figsize=(8, 6))

customer_revenue.plot(kind="bar")

plt.title("Revenue by Customer Type")
plt.xlabel("Customer Type")
plt.ylabel("Revenue (₹)")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    output_dir / "customer_type_revenue.png",
    dpi=150
)

plt.close()

# -----------------------------
# 7. PAYMENT METHOD
# -----------------------------

payment_orders = (
    df["Payment_Mode"]
    .value_counts()
)

plt.figure(figsize=(10, 6))

payment_orders.plot(kind="bar")

plt.title("Orders by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Orders")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    output_dir / "payment_methods.png",
    dpi=150
)

plt.close()

# -----------------------------
# 8. TOP 10 PRODUCTS
# -----------------------------

top_products = (
    df.groupby("Product")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .sort_values()
)

plt.figure(figsize=(10, 7))

top_products.plot(kind="barh")

plt.title("Top 10 Products by Revenue")
plt.xlabel("Revenue (₹)")
plt.ylabel("Product")

plt.tight_layout()

plt.savefig(
    output_dir / "top_products.png",
    dpi=150
)

plt.close()

# -----------------------------
# 9. DISCOUNT VS PROFIT
# -----------------------------

discount_profit = (
    df.groupby("Discount")["Profit"]
    .mean()
)

plt.figure(figsize=(9, 6))

discount_profit.plot(
    marker="o"
)

plt.title("Average Profit vs Discount")
plt.xlabel("Discount (%)")
plt.ylabel("Average Profit (₹)")

plt.tight_layout()

plt.savefig(
    output_dir / "discount_vs_profit.png",
    dpi=150
)

plt.close()

# -----------------------------
# 10. CUSTOMER RATING VS REVENUE
# -----------------------------

rating_revenue = (
    df.groupby("Customer_Rating")["Revenue"]
    .sum()
)

plt.figure(figsize=(9, 6))

rating_revenue.plot(kind="bar")

plt.title("Revenue by Customer Rating")
plt.xlabel("Customer Rating")
plt.ylabel("Revenue (₹)")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    output_dir / "rating_revenue.png",
    dpi=150
)

plt.close()

# -----------------------------
# COMPLETE
# -----------------------------

print("=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nCharts saved in: {output_dir}")

print("\nGenerated files:")

for file in sorted(output_dir.iterdir()):
    print(f"✓ {file.name}")