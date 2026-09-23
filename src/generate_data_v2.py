import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

# ============================================================
# SETTINGS
# ============================================================

NUM_CUSTOMERS = 10000
START_DATE = pd.Timestamp("2024-01-01")
END_DATE = pd.Timestamp("2026-06-30")

# ============================================================
# MASTER DATA
# ============================================================

products = {
    "Electronics": {
        "Laptop": (45000, 120000),
        "Smartphone": (12000, 90000),
        "Tablet": (15000, 60000),
        "Monitor": (8000, 45000),
        "Headphones": (1000, 25000)
    },
    "Home Appliances": {
        "Refrigerator": (20000, 90000),
        "Washing Machine": (18000, 70000),
        "Microwave": (7000, 30000),
        "Air Conditioner": (30000, 90000),
        "Vacuum Cleaner": (5000, 30000)
    },
    "Furniture": {
        "Office Chair": (3000, 25000),
        "Study Table": (4000, 30000),
        "Sofa": (15000, 100000),
        "Bed": (12000, 80000),
        "Bookshelf": (3000, 20000)
    },
    "Fashion": {
        "T-Shirt": (500, 3000),
        "Jeans": (1000, 5000),
        "Jacket": (1500, 8000),
        "Sneakers": (1500, 12000),
        "Backpack": (700, 5000)
    },
    "Beauty": {
        "Face Cream": (300, 2500),
        "Perfume": (500, 6000),
        "Shampoo": (200, 1500),
        "Makeup Kit": (700, 5000),
        "Skincare Set": (800, 6000)
    }
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

# ============================================================
# CUSTOMER PROFILES
# ============================================================

customer_ids = [
    f"CUST{str(i).zfill(5)}"
    for i in range(1, NUM_CUSTOMERS + 1)
]

profiles = []

for customer_id in customer_ids:

    segment = np.random.choice(
        [
            "Loyal",
            "Regular",
            "Occasional",
            "At_Risk"
        ],
        p=[0.20, 0.40, 0.25, 0.15]
    )

    region = np.random.choice(
        list(regions.keys()),
        p=[0.30, 0.25, 0.25, 0.20]
    )

    # Different customers have different purchase behavior
    if segment == "Loyal":
        purchase_rate = np.random.uniform(0.75, 1.0)
        avg_orders = np.random.randint(12, 25)
        spending_factor = np.random.uniform(1.2, 1.8)

    elif segment == "Regular":
        purchase_rate = np.random.uniform(0.45, 0.75)
        avg_orders = np.random.randint(6, 14)
        spending_factor = np.random.uniform(0.8, 1.3)

    elif segment == "Occasional":
        purchase_rate = np.random.uniform(0.20, 0.45)
        avg_orders = np.random.randint(2, 7)
        spending_factor = np.random.uniform(0.5, 1.0)

    else:
        purchase_rate = np.random.uniform(0.08, 0.30)
        avg_orders = np.random.randint(2, 6)
        spending_factor = np.random.uniform(0.5, 1.0)

    profiles.append(
        {
            "Customer_ID": customer_id,
            "Segment_Profile": segment,
            "Region_Profile": region,
            "Purchase_Rate": purchase_rate,
            "Expected_Orders": avg_orders,
            "Spending_Factor": spending_factor
        }
    )

profiles_df = pd.DataFrame(profiles)

# ============================================================
# GENERATE TRANSACTIONS
# ============================================================

rows = []

order_number = 1

for _, customer in profiles_df.iterrows():

    customer_id = customer["Customer_ID"]

    segment = customer["Segment_Profile"]

    region = customer["Region_Profile"]

    expected_orders = int(customer["Expected_Orders"])

    spending_factor = customer["Spending_Factor"]

    # Number of orders
    num_orders = max(
        1,
        int(
            np.random.normal(
                expected_orders,
                max(1, expected_orders * 0.20)
            )
        )
    )

    # At-risk customers gradually become inactive
    if segment == "At_Risk":

        active_end = pd.Timestamp(
            np.random.choice(
                pd.date_range(
                    "2025-06-01",
                    "2026-03-31"
                )
            )
        )

    else:

        active_end = END_DATE

    available_days = (
        active_end - START_DATE
    ).days

    if available_days < 30:
        available_days = 30

    for _ in range(num_orders):

        order_date = (
            START_DATE +
            pd.Timedelta(
                days=np.random.randint(
                    0,
                    available_days + 1
                )
            )
        )

        # Category preference
        category = np.random.choice(
            list(products.keys()),
            p=[
                0.30,
                0.20,
                0.15,
                0.20,
                0.15
            ]
        )

        product = np.random.choice(
            list(products[category].keys())
        )

        min_price, max_price = products[
            category
        ][product]

        base_price = np.random.randint(
            min_price,
            max_price + 1
        )

        unit_price = int(
            base_price * spending_factor
        )

        unit_price = max(
            min_price,
            min(unit_price, max_price)
        )

        quantity = np.random.choice(
            [1, 2, 3, 4],
            p=[0.58, 0.25, 0.12, 0.05]
        )

        discount = np.random.choice(
            [0, 5, 10, 15, 20, 25],
            p=[
                0.20,
                0.20,
                0.25,
                0.18,
                0.12,
                0.05
            ]
        )

        gross = quantity * unit_price

        discount_amount = (
            gross * discount / 100
        )

        revenue = gross - discount_amount

        # Higher discount slightly reduces margin
        cost_ratio = np.random.uniform(
            0.58,
            0.82
        )

        cost = revenue * cost_ratio

        profit = revenue - cost

        payment = np.random.choice(
            payment_modes,
            p=[
                0.40,
                0.20,
                0.15,
                0.15,
                0.10
            ]
        )

        if segment == "Loyal":
            rating = np.random.choice(
                [3, 4, 5],
                p=[0.05, 0.40, 0.55]
            )

        elif segment == "At_Risk":
            rating = np.random.choice(
                [1, 2, 3, 4, 5],
                p=[0.10, 0.20, 0.30, 0.30, 0.10]
            )

        else:
            rating = np.random.choice(
                [1, 2, 3, 4, 5],
                p=[0.03, 0.07, 0.15, 0.40, 0.35]
            )

        delivery_days = np.random.randint(
            1,
            11
        )

        customer_type = (
            "Returning"
            if num_orders > 1
            else "New"
        )

        rows.append(
            [
                f"ORD{str(order_number).zfill(7)}",
                order_date,
                customer_id,
                product,
                category,
                region,
                np.random.choice(regions[region]),
                quantity,
                unit_price,
                discount,
                round(revenue, 2),
                round(cost, 2),
                round(profit, 2),
                payment,
                customer_type,
                rating,
                delivery_days,
                segment
            ]
        )

        order_number += 1


# ============================================================
# CREATE DATAFRAME
# ============================================================

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
    "Delivery_Days",
    "Customer_Profile"
]

df = pd.DataFrame(
    rows,
    columns=columns
)

df["Order_Date"] = pd.to_datetime(
    df["Order_Date"]
)

df = df.sort_values(
    "Order_Date"
)

# ============================================================
# SAVE
# ============================================================

output = Path(
    "data/retail_data_v2.csv"
)

df.to_csv(
    output,
    index=False
)

print("=" * 60)
print("RETAILPULSE AI V2 DATASET")
print("=" * 60)

print(
    f"Rows      : {len(df):,}"
)

print(
    f"Columns   : {len(df.columns)}"
)

print(
    f"Customers : {df['Customer_ID'].nunique():,}"
)

print(
    f"Date      : {df['Order_Date'].min().date()} "
    f"to {df['Order_Date'].max().date()}"
)

print("\nCUSTOMER PROFILE")

print(
    df.groupby("Customer_Profile")
    ["Customer_ID"]
    .nunique()
)

print("\nDATASET SAVED:")
print(output)

print("=" * 60)