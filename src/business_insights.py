import pandas as pd

# ---------------------------------------
# LOAD DATA
# ---------------------------------------

df = pd.read_csv("data/retail_data.csv")

df["Order_Date"] = pd.to_datetime(df["Order_Date"])


# ---------------------------------------
# KPI CALCULATIONS
# ---------------------------------------

def calculate_kpis(df):

    revenue = df["Revenue"].sum()
    profit = df["Profit"].sum()
    orders = df["Order_ID"].nunique()
    customers = df["Customer_ID"].nunique()
    units = df["Quantity"].sum()

    aov = revenue / orders
    profit_margin = (profit / revenue) * 100

    return {
        "revenue": revenue,
        "profit": profit,
        "orders": orders,
        "customers": customers,
        "units": units,
        "aov": aov,
        "profit_margin": profit_margin
    }


# ---------------------------------------
# CATEGORY INSIGHTS
# ---------------------------------------

def category_insights(df):

    result = (
        df.groupby("Category")
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order_ID", "nunique")
        )
    )

    result["Profit_Margin"] = (
        result["Profit"] / result["Revenue"] * 100
    )

    result = result.sort_values(
        "Revenue",
        ascending=False
    )

    return result


# ---------------------------------------
# REGION INSIGHTS
# ---------------------------------------

def region_insights(df):

    result = (
        df.groupby("Region")
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order_ID", "nunique")
        )
    )

    result["Profit_Margin"] = (
        result["Profit"] / result["Revenue"] * 100
    )

    return result.sort_values(
        "Revenue",
        ascending=False
    )


# ---------------------------------------
# PRODUCT INSIGHTS
# ---------------------------------------

def product_insights(df):

    result = (
        df.groupby("Product")
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            Quantity=("Quantity", "sum"),
            Orders=("Order_ID", "nunique")
        )
    )

    result["Profit_Margin"] = (
        result["Profit"] / result["Revenue"] * 100
    )

    return result.sort_values(
        "Revenue",
        ascending=False
    )


# ---------------------------------------
# CUSTOMER INSIGHTS
# ---------------------------------------

def customer_type_insights(df):

    result = (
        df.groupby("Customer_Type")
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order_ID", "nunique"),
            Customers=("Customer_ID", "nunique")
        )
    )

    result["Revenue_Per_Customer"] = (
        result["Revenue"] /
        result["Customers"]
    )

    return result


# ---------------------------------------
# MONTHLY TREND
# ---------------------------------------

def monthly_trend(df):

    result = (
        df.groupby(
            df["Order_Date"].dt.to_period("M")
        )
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order_ID", "nunique")
        )
        .reset_index()
    )

    result["Order_Date"] = (
        result["Order_Date"]
        .dt.to_timestamp()
    )

    return result


# ---------------------------------------
# TOP PRODUCTS
# ---------------------------------------

def top_products(df, n=10):

    return (
        df.groupby("Product")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
    )


# ---------------------------------------
# BUSINESS INSIGHTS
# ---------------------------------------

def generate_insights(df):

    kpis = calculate_kpis(df)

    categories = category_insights(df)

    regions = region_insights(df)

    products = product_insights(df)

    customers = customer_type_insights(df)

    insights = []

    # Revenue leader
    top_category = categories.index[0]

    insights.append(
        f"{top_category} is the highest revenue-generating category."
    )

    # Region leader
    top_region = regions.index[0]

    insights.append(
        f"{top_region} is the highest revenue-generating region."
    )

    # Product leader
    top_product = products.index[0]

    insights.append(
        f"{top_product} is the highest revenue-generating product."
    )

    # Returning customers
    returning_revenue = customers.loc[
        "Returning",
        "Revenue"
    ]

    new_revenue = customers.loc[
        "New",
        "Revenue"
    ]

    if returning_revenue > new_revenue:

        insights.append(
            "Returning customers generate more revenue than new customers."
        )

    # Profit margin
    if kpis["profit_margin"] >= 25:

        insights.append(
            f"Overall profit margin is healthy at "
            f"{kpis['profit_margin']:.2f}%."
        )

    return insights


# ---------------------------------------
# RUN ANALYSIS
# ---------------------------------------

if __name__ == "__main__":

    kpis = calculate_kpis(df)

    print("=" * 60)
    print("RETAILPULSE AI - BUSINESS INSIGHTS")
    print("=" * 60)

    print("\nKEY PERFORMANCE INDICATORS")

    print(
        f"Revenue        : ₹{kpis['revenue']:,.2f}"
    )

    print(
        f"Profit         : ₹{kpis['profit']:,.2f}"
    )

    print(
        f"Orders         : {kpis['orders']:,}"
    )

    print(
        f"Customers      : {kpis['customers']:,}"
    )

    print(
        f"Units Sold     : {kpis['units']:,}"
    )

    print(
        f"Average Order  : ₹{kpis['aov']:,.2f}"
    )

    print(
        f"Profit Margin  : {kpis['profit_margin']:.2f}%"
    )

    print("\nBUSINESS INSIGHTS")

    insights = generate_insights(df)

    for i, insight in enumerate(insights, 1):

        print(f"{i}. {insight}")

    print("\n" + "=" * 60)