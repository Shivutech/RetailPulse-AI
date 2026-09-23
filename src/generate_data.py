import pandas as pd
import numpy as np
from pathlib import Path

# Reproducibility
np.random.seed(42)

# -----------------------------
# PROJECT SETTINGS
# -----------------------------

NUM_ORDERS = 50000
NUM_CUSTOMERS = 10000

# -----------------------------
# MASTER DATA
# -----------------------------

products = {
    "Electronics": [
        "Laptop",
        "Smartphone",
        "Tablet",
        "Monitor",
        "Headphones"
    ],
    "Home Appliances": [
        "Refrigerator",
        "Washing Machine",
        "Microwave",
        "Air Conditioner",
        "Vacuum Cleaner"
    ],
    "Furniture": [
        "Office Chair",
        "Study Table",
        "Sofa",
        "Bed",
        "Bookshelf"
    ],
    "Fashion": [
        "T-Shirt",
        "Jeans",
        "Jacket",
        "Sneakers",
        "Backpack"
    ],
    "Beauty": [
        "Face Cream",
        "Perfume",
        "Shampoo",
        "Makeup Kit",
        "Skincare Set"
    ]
}

regions = {
    "North": ["Delhi", "Kanpur", "Lucknow", "Chandigarh"],
    "South": ["Bangalore", "Chennai", "Hyderabad", "Kochi"],
    "West": ["Mumbai", "Pune", "Ahmedabad", "Surat"],
    "East": ["Kolkata", "Bhubaneswar", "Patna", "Guwahati"]
}

payment_modes = [
    "UPI",
    "Credit Card",
    "Debit Card",
    "Cash on Delivery",
    "Net Banking"
]

# -----------------------------
# PRODUCT PRICE RANGES
# -----------------------------

price_ranges = {
    "Laptop": (45000, 120000),
    "Smartphone": (12000, 90000),
    "Tablet": (15000, 60000),
    "Monitor": (8000, 45000),
    "Headphones": (1000, 25000),

    "Refrigerator": (20000, 90000),
    "Washing Machine": (18000, 70000),
    "Microwave": (7000, 30000),
    "Air Conditioner": (30000, 90000),
    "Vacuum Cleaner": (5000, 30000),

    "Office Chair": (3000, 25000),
    "Study Table": (4000, 30000),
    "Sofa": (15000, 100000),
    "Bed": (12000, 80000),
    "Bookshelf": (3000, 20000),

    "T-Shirt": (500, 3000),
    "Jeans": (1000, 5000),
    "Jacket": (1500, 8000),
    "Sneakers": (1500, 12000),
    "Backpack": (700, 5000),

    "Face Cream": (300, 2500),
    "Perfume": (500, 6000),
    "Shampoo": (200, 1500),
    "Makeup Kit": (700, 5000),
    "Skincare Set": (800, 6000)
}

# -----------------------------
# CREATE CUSTOMER IDs
# -----------------------------

customers = [
    f"CUST{str(i).zfill(5)}"
    for i in range(1, NUM_CUSTOMERS + 1)
]

# -----------------------------
# CREATE ORDERS
# -----------------------------

rows = []

start_date = pd.Timestamp("2024-01-01")
end_date = pd.Timestamp("2026-06-30")

date_range_days = (end_date - start_date).days

category_names = list(products.keys())

for i in range(NUM_ORDERS):

    order_id = f"ORD{str(i + 1).zfill(6)}"

    customer_id = np.random.choice(customers)

    # Random order date
    order_date = start_date + pd.Timedelta(
        days=np.random.randint(0, date_range_days + 1)
    )

    # Region
    region = np.random.choice(
        list(regions.keys()),
        p=[0.30, 0.25, 0.25, 0.20]
    )

    city = np.random.choice(regions[region])

    # Category
    category = np.random.choice(
        category_names,
        p=[0.30, 0.20, 0.15, 0.20, 0.15]
    )

    # Product
    product = np.random.choice(products[category])

    # Price
    min_price, max_price = price_ranges[product]

    unit_price = np.random.randint(
        min_price,
        max_price + 1
    )

    # Quantity
    quantity = np.random.choice(
        [1, 2, 3, 4, 5],
        p=[0.55, 0.25, 0.12, 0.06, 0.02]
    )

    # Discount
    discount = np.random.choice(
        [0, 5, 10, 15, 20, 25],
        p=[0.20, 0.20, 0.25, 0.18, 0.12, 0.05]
    )

    gross_amount = quantity * unit_price

    discount_amount = gross_amount * discount / 100

    revenue = gross_amount - discount_amount

    # Cost is usually 60-85% of revenue
    cost_ratio = np.random.uniform(0.60, 0.85)

    cost = revenue * cost_ratio

    profit = revenue - cost

    # Payment
    payment_mode = np.random.choice(
        payment_modes,
        p=[0.40, 0.20, 0.15, 0.15, 0.10]
    )

    # Customer type
    customer_type = np.random.choice(
        ["New", "Returning"],
        p=[0.35, 0.65]
    )

    # Rating
    customer_rating = np.random.choice(
        [1, 2, 3, 4, 5],
        p=[0.03, 0.07, 0.15, 0.40, 0.35]
    )

    # Delivery days
    delivery_days = np.random.randint(1, 11)

    rows.append([
        order_id,
        order_date,
        customer_id,
        product,
        category,
        region,
        city,
        quantity,
        unit_price,
        discount,
        round(revenue, 2),
        round(cost, 2),
        round(profit, 2),
        payment_mode,
        customer_type,
        customer_rating,
        delivery_days
    ])

# -----------------------------
# CREATE DATAFRAME
# -----------------------------

columns = [
    "Order_ID",
    "Order_Date",
    "Customer_ID",
    "Product",
    "Category",
    "Region",
    "City",
    "Quantity",
    "Unit_Price",
    "Discount",
    "Revenue",
    "Cost",
    "Profit",
    "Payment_Mode",
    "Customer_Type",
    "Customer_Rating",
    "Delivery_Days"
]

df = pd.DataFrame(rows, columns=columns)

# Convert date
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Sort by date
df = df.sort_values("Order_Date")

# -----------------------------
# SAVE DATASET
# -----------------------------

output_path = Path("data/retail_data.csv")

df.to_csv(output_path, index=False)

print("=" * 60)
print("RetailPulse AI Dataset Generated Successfully")
print("=" * 60)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Customers: {df['Customer_ID'].nunique():,}")
print(f"Date Range: {df['Order_Date'].min().date()} to "
      f"{df['Order_Date'].max().date()}")

print("\nDataset Preview:")
print(df.head())

print("\nCategory Distribution:")
print(df["Category"].value_counts())

print("\nRegion Distribution:")
print(df["Region"].value_counts())

print(f"\nDataset saved at: {output_path}")