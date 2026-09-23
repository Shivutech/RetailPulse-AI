import os
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai

# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY not found in .env file."
    )

client = genai.Client(
    api_key=api_key
)

# Primary model + fallback models.
# If one model is temporarily unavailable, the next one is tried.
MODEL_NAMES = [
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]


# ============================================================
# LOAD RETAIL DATA
# ============================================================

def load_business_data():

    df = pd.read_csv(
        "data/retail_data_v2.csv"
    )

    rfm = pd.read_csv(
        "data/customer_rfm_v2.csv"
    )

    churn = pd.read_csv(
        "data/customer_churn_predictions_v2.csv"
    )

    forecast = pd.read_csv(
        "data/demand_forecast.csv"
    )

    anomalies = pd.read_csv(
        "data/anomaly_detection_results.csv"
    )

    return (
        df,
        rfm,
        churn,
        forecast,
        anomalies
    )


# ============================================================
# CREATE BUSINESS CONTEXT
# ============================================================

def create_business_context():

    (
        df,
        rfm,
        churn,
        forecast,
        anomalies
    ) = load_business_data()

    # --------------------------------------------------------
    # BUSINESS KPIs
    # --------------------------------------------------------

    revenue = df["Revenue"].sum()
    profit = df["Profit"].sum()
    orders = len(df)
    customers = df["Customer_ID"].nunique()
    units = df["Quantity"].sum()

    aov = revenue / orders

    margin = (
        profit / revenue
    ) * 100

    # --------------------------------------------------------
    # TOP PERFORMERS
    # --------------------------------------------------------

    category_revenue = (
        df.groupby("Category")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    region_revenue = (
        df.groupby("Region")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    product_revenue = (
        df.groupby("Product")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    # --------------------------------------------------------
    # RFM
    # --------------------------------------------------------

    segment_counts = (
        rfm["Segment"]
        .value_counts()
        .to_dict()
    )

    segment_revenue = (
        rfm.groupby("Segment")["Monetary"]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )

    # --------------------------------------------------------
    # CHURN
    # --------------------------------------------------------

    risk_counts = (
        churn["Risk_Level"]
        .value_counts()
        .to_dict()
    )

    revenue_at_risk = (
        churn["Revenue_at_Risk"].sum()
    )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    forecast_data = forecast[
        [
            "Order_Date",
            "Forecast_Quantity"
        ]
    ].to_dict(
        orient="records"
    )

    # --------------------------------------------------------
    # ANOMALIES
    # --------------------------------------------------------

    detected_anomalies = anomalies[
        anomalies["Status"] == "Anomaly"
    ][
        [
            "Order_Date",
            "Revenue",
            "Profit",
            "Quantity"
        ]
    ].to_dict(
        orient="records"
    )

    # --------------------------------------------------------
    # BUSINESS CONTEXT
    # --------------------------------------------------------

    context = f"""
You are the AI Business Analyst for RetailPulse AI,
an AI-powered retail business intelligence system.

Analyze the provided business metrics and answer the
user's question using ONLY the available business data.

BUSINESS KPIs
-------------
Total Revenue: ₹{revenue:,.2f}
Total Profit: ₹{profit:,.2f}
Orders: {orders:,}
Customers: {customers:,}
Units Sold: {units:,}
Average Order Value: ₹{aov:,.2f}
Profit Margin: {margin:.2f}%

TOP CATEGORIES BY REVENUE
-------------------------
{category_revenue.to_dict()}

TOP REGIONS BY REVENUE
----------------------
{region_revenue.to_dict()}

TOP PRODUCTS BY REVENUE
-----------------------
{product_revenue.head(10).to_dict()}

CUSTOMER SEGMENTS
-----------------
Customer Counts:
{segment_counts}

Segment Revenue:
{segment_revenue}

CHURN RISK
----------
Risk Distribution:
{risk_counts}

Estimated Revenue at Risk:
₹{revenue_at_risk:,.2f}

DEMAND FORECAST
---------------
{forecast_data}

DETECTED ANOMALIES
------------------
{detected_anomalies}

INSTRUCTIONS
------------
1. Answer clearly and concisely.
2. Use actual numbers from the provided data.
3. Do not invent facts.
4. If the data does not contain enough information,
   explicitly say so.
5. Explain the business implication when useful.
6. Give practical recommendations when appropriate.
"""

    return context


# ============================================================
# ASK GEMINI WITH RETRY + FALLBACK
# ============================================================

def ask_business_analyst(question):

    context = create_business_context()

    prompt = f"""
{context}

USER QUESTION
-------------
{question}

Provide a professional business-analysis response.
Use short sections or bullet points when helpful.
"""

    last_error = None

    for model_name in MODEL_NAMES:

        # Try each model up to 2 times.
        for attempt in range(2):

            try:

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                if response and response.text:
                    return response.text

                last_error = (
                    f"{model_name} returned an empty response."
                )

            except Exception as e:

                last_error = e

                error_text = str(e).lower()

                # Retry temporary server/rate-limit errors.
                temporary_error = (
                    "503" in error_text
                    or "unavailable" in error_text
                    or "429" in error_text
                    or "resource_exhausted" in error_text
                    or "500" in error_text
                )

                if temporary_error and attempt == 0:
                    time.sleep(2)
                    continue

                # Move to the next fallback model.
                break

    return (
        "AI analysis is temporarily unavailable. "
        "Please try again in a moment."
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RETAILPULSE AI - BUSINESS ANALYST TEST")
    print("=" * 60)

    question = (
        "Which category generates the highest revenue "
        "and what does this mean for the business?"
    )

    answer = ask_business_analyst(
        question
    )

    print("\nAI BUSINESS ANALYST RESPONSE")
    print("-" * 60)
    print(answer)

    print("\n" + "=" * 60)
