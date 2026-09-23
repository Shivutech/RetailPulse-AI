import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.ai_business_analyst import ask_business_analyst

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RetailPulse AI",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.dashboard-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.dashboard-subtitle {
    color: #64748b;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

.metric-card {
    padding: 20px;
    border-radius: 12px;
    background-color: white;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.section-title {
    font-size: 1.35rem;
    font-weight: 650;
    margin-top: 1.5rem;
    margin-bottom: 0.8rem;
}

.insight-box {
    padding: 15px 18px;
    border-radius: 10px;
    background-color: #f1f5f9;
    border-left: 4px solid #334155;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/retail_data_v2.csv"
    )

    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"]
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

    forecast["Order_Date"] = pd.to_datetime(
        forecast["Order_Date"]
    )

    anomalies = pd.read_csv(
        "data/anomaly_detection_results.csv"
    )

    anomalies["Order_Date"] = pd.to_datetime(
        anomalies["Order_Date"]
    )

    return (
        df,
        rfm,
        churn,
        forecast,
        anomalies
    )


try:

    (
        df,
        rfm,
        churn,
        forecast,
        anomalies
    ) = load_data()

except Exception as e:

    st.error(
        f"Data loading error: {e}"
    )

    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 RetailPulse AI")

st.sidebar.caption(
    "AI-Powered Business Intelligence"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Overview",
        "📈 Sales Analytics",
        "👥 Customer Intelligence",
        "🤖 AI Predictions",
        "🧠 AI Business Analyst",
        "🚨 Anomaly Detection",
        "🔮 Demand Forecast"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "RetailPulse AI converts business data "
    "into actionable insights using analytics "
    "and machine learning."
)

# ============================================================
# COMMON KPIs
# ============================================================

total_revenue = df["Revenue"].sum()
total_profit = df["Profit"].sum()
total_orders = len(df)
total_customers = df["Customer_ID"].nunique()
total_units = df["Quantity"].sum()

aov = total_revenue / total_orders

profit_margin = (
    total_profit / total_revenue
) * 100

# ============================================================
# OVERVIEW
# ============================================================

if page == "📊 Overview":

    st.markdown(
        '<div class="dashboard-title">RetailPulse AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        'AI-Powered Retail Business Intelligence & Decision Support System'
        '</div>',
        unsafe_allow_html=True
    )

    # KPI ROW

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Revenue",
        f"₹{total_revenue / 1e7:.2f} Cr"
    )

    col2.metric(
        "Total Profit",
        f"₹{total_profit / 1e7:.2f} Cr"
    )

    col3.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    col4.metric(
        "Customers",
        f"{total_customers:,}"
    )

    col5, col6, col7 = st.columns(3)

    col5.metric(
        "Units Sold",
        f"{total_units:,}"
    )

    col6.metric(
        "Average Order Value",
        f"₹{aov:,.0f}"
    )

    col7.metric(
        "Profit Margin",
        f"{profit_margin:.2f}%"
    )

    # --------------------------------------------------------
    # MONTHLY TREND
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Monthly Business Performance</div>',
        unsafe_allow_html=True
    )

    monthly = (
        df.set_index("Order_Date")
        .resample("MS")
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum")
        )
        .reset_index()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=monthly["Order_Date"],
            y=monthly["Revenue"],
            mode="lines+markers",
            name="Revenue"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=monthly["Order_Date"],
            y=monthly["Profit"],
            mode="lines+markers",
            name="Profit"
        )
    )

    fig.update_layout(
        height=420,
        hovermode="x unified",
        yaxis_title="Amount (₹)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # CATEGORY + REGION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        category = (
            df.groupby("Category")["Revenue"]
            .sum()
            .sort_values(
                ascending=False
            )
            .reset_index()
        )

        fig = px.bar(
            category,
            x="Category",
            y="Revenue",
            title="Revenue by Category"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        region = (
            df.groupby("Region")["Revenue"]
            .sum()
            .sort_values(
                ascending=False
            )
            .reset_index()
        )

        fig = px.bar(
            region,
            x="Region",
            y="Revenue",
            title="Revenue by Region"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # KEY INSIGHTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Key Business Insights</div>',
        unsafe_allow_html=True
    )

    top_category = (
        df.groupby("Category")["Revenue"]
        .sum()
        .idxmax()
    )

    top_region = (
        df.groupby("Region")["Revenue"]
        .sum()
        .idxmax()
    )

    top_product = (
        df.groupby("Product")["Revenue"]
        .sum()
        .idxmax()
    )

    insights = [
        f"📌 {top_category} is the highest revenue-generating category.",
        f"📌 {top_region} is the highest revenue-generating region.",
        f"📌 {top_product} is the highest revenue-generating product.",
        f"📌 Overall business profit margin is {profit_margin:.2f}%.",
        f"📌 Returning customers contribute the majority of total revenue."
    ]

    for insight in insights:

        st.markdown(
            f'<div class="insight-box">{insight}</div>',
            unsafe_allow_html=True
        )

# ============================================================
# SALES ANALYTICS
# ============================================================

elif page == "📈 Sales Analytics":

    st.title("📈 Sales Analytics")

    # Filters

    col1, col2, col3 = st.columns(3)

    with col1:

        categories = st.multiselect(
            "Category",
            sorted(df["Category"].unique()),
            default=sorted(df["Category"].unique())
        )

    with col2:

        regions = st.multiselect(
            "Region",
            sorted(df["Region"].unique()),
            default=sorted(df["Region"].unique())
        )

    with col3:

        customer_types = st.multiselect(
            "Customer Type",
            sorted(df["Customer_Type"].unique()),
            default=sorted(df["Customer_Type"].unique())
        )

    filtered = df[
        df["Category"].isin(categories)
        & df["Region"].isin(regions)
        & df["Customer_Type"].isin(customer_types)
    ]

    # KPIs

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Revenue",
        f"₹{filtered['Revenue'].sum() / 1e7:.2f} Cr"
    )

    c2.metric(
        "Profit",
        f"₹{filtered['Profit'].sum() / 1e7:.2f} Cr"
    )

    c3.metric(
        "Orders",
        f"{len(filtered):,}"
    )

    c4.metric(
        "Units",
        f"{filtered['Quantity'].sum():,}"
    )

    # Monthly Revenue

    monthly_sales = (
        filtered.set_index("Order_Date")
        .resample("MS")
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum")
        )
        .reset_index()
    )

    fig = px.line(
        monthly_sales,
        x="Order_Date",
        y=["Revenue", "Profit"],
        markers=True,
        title="Monthly Revenue & Profit"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

        product_sales = (
            filtered.groupby("Product")["Revenue"]
            .sum()
            .sort_values(
                ascending=False
            )
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            product_sales,
            x="Revenue",
            y="Product",
            orientation="h",
            title="Top 10 Products"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        category_sales = (
            filtered.groupby("Category")["Revenue"]
            .sum()
            .reset_index()
        )

        fig = px.pie(
            category_sales,
            names="Category",
            values="Revenue",
            title="Revenue Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ============================================================
# CUSTOMER INTELLIGENCE
# ============================================================

elif page == "👥 Customer Intelligence":

    st.title("👥 Customer Intelligence")

    # Segment distribution

    segment_count = (
        rfm["Segment"]
        .value_counts()
        .reset_index()
    )

    segment_count.columns = [
        "Segment",
        "Customers"
    ]

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            segment_count,
            names="Segment",
            values="Customers",
            title="Customer Segment Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        segment_revenue = (
            rfm.groupby("Segment")["Monetary"]
            .sum()
            .sort_values(
                ascending=False
            )
            .reset_index()
        )

        fig = px.bar(
            segment_revenue,
            x="Segment",
            y="Monetary",
            title="Revenue by Customer Segment"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # RFM table

    st.subheader("Customer Segment Summary")

    summary = (
        rfm.groupby("Segment")
        .agg(
            Customers=("Customer_ID", "count"),
            Revenue=("Monetary", "sum"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Recency=("Recency", "mean"),
            Avg_Order_Value=("Average_Order_Value", "mean")
        )
        .sort_values(
            "Revenue",
            ascending=False
        )
    )

    st.dataframe(
        summary,
        use_container_width=True
    )

# ============================================================
# AI PREDICTIONS
# ============================================================

elif page == "🤖 AI Predictions":

    st.title("🤖 AI Predictions")

    st.subheader("Customer Churn Risk")

    # Risk distribution

    risk_count = (
        churn["Risk_Level"]
        .value_counts()
        .reset_index()
    )

    risk_count.columns = [
        "Risk_Level",
        "Customers"
    ]

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            risk_count,
            x="Risk_Level",
            y="Customers",
            title="Churn Risk Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        revenue_risk = (
            churn.groupby("Risk_Level")[
                "Revenue_at_Risk"
            ]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            revenue_risk,
            x="Risk_Level",
            y="Revenue_at_Risk",
            title="Revenue at Risk"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # High-risk customers

    st.subheader("High-Risk Customers")

    high_risk = (
        churn[
            churn["Risk_Level"] == "High"
        ]
        .sort_values(
            "Churn_Probability",
            ascending=False
        )
        .head(20)
    )

    st.dataframe(
        high_risk[
            [
                "Customer_ID",
                "Churn_Probability",
                "Risk_Level",
                "Monetary",
                "Recency",
                "Frequency",
                "Revenue_at_Risk"
            ]
        ],
        use_container_width=True
    )

# ============================================================
# AI BUSINESS ANALYST
# ============================================================

elif page == "🧠 AI Business Analyst":

    st.title("🧠 AI Business Analyst")

    st.caption(
        "Ask questions about your retail business data and get "
        "AI-powered business analysis."
    )

    # --------------------------------------------------------
    # SAMPLE QUESTIONS
    # --------------------------------------------------------

    st.subheader("💡 Try a question")

    sample_questions = [
        "Which category generates the highest revenue?",
        "Which customers should we target for retention?",
        "How much revenue is currently at risk?",
        "What is the expected demand for the next 3 months?",
        "Which region generates the highest revenue?",
        "What business anomalies were detected?"
    ]

    selected_question = st.selectbox(
        "Choose a sample question",
        ["Select a question"] + sample_questions
    )

    # --------------------------------------------------------
    # USER QUESTION
    # --------------------------------------------------------

    question = st.text_input(
        "Ask your own business question",
        placeholder="Example: Which category should we focus on?"
    )

    # Use selected question if user hasn't typed one
    if (
        selected_question != "Select a question"
        and not question.strip()
    ):
        question = selected_question

    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Analyze",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter or select a business question."
            )

        else:

            with st.spinner(
                "AI Business Analyst is analyzing your data..."
            ):

                answer = ask_business_analyst(
                    question
                )

            st.subheader("📊 AI Analysis")

            st.markdown(
                answer
            )

    # --------------------------------------------------------
    # BUSINESS DATA SUMMARY
    # --------------------------------------------------------

    st.divider()

    st.subheader("📌 Available Business Intelligence")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Revenue",
        f"₹{total_revenue / 1e7:.2f} Cr"
    )

    col2.metric(
        "Profit",
        f"₹{total_profit / 1e7:.2f} Cr"
    )

    col3.metric(
        "Customers",
        f"{total_customers:,}"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Orders",
        f"{total_orders:,}"
    )

    col5.metric(
        "Profit Margin",
        f"{profit_margin:.2f}%"
    )

    col6.metric(
        "Average Order Value",
        f"₹{aov:,.0f}"
    )

# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "🚨 Anomaly Detection":

    st.title("🚨 Anomaly Detection")

    anomaly_count = (
        anomalies["Status"] == "Anomaly"
    ).sum()

    normal_count = (
        anomalies["Status"] == "Normal"
    ).sum()

    c1, c2 = st.columns(2)

    c1.metric(
        "Normal Months",
        normal_count
    )

    c2.metric(
        "Anomalous Months",
        anomaly_count
    )

    # Revenue trend

    fig = go.Figure()

    normal = anomalies[
        anomalies["Status"] == "Normal"
    ]

    anomaly = anomalies[
        anomalies["Status"] == "Anomaly"
    ]

    fig.add_trace(
        go.Scatter(
            x=normal["Order_Date"],
            y=normal["Revenue"],
            mode="markers",
            name="Normal"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=anomaly["Order_Date"],
            y=anomaly["Revenue"],
            mode="markers",
            marker=dict(size=12),
            name="Anomaly"
        )
    )

    fig.update_layout(
        title="Revenue Anomaly Timeline",
        xaxis_title="Month",
        yaxis_title="Revenue"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Detected Anomalies")

    st.dataframe(
        anomaly[
            [
                "Order_Date",
                "Revenue",
                "Profit",
                "Quantity",
                "Orders",
                "Customers",
                "Anomaly_Score"
            ]
        ],
        use_container_width=True
    )

# ============================================================
# DEMAND FORECAST
# ============================================================

elif page == "🔮 Demand Forecast":

    st.title("🔮 Demand Forecast")

    st.info(
        "AI-powered short-term demand forecast "
        "for the next three months."
    )

    # Forecast table

    st.subheader("Next 3 Months Forecast")

    st.dataframe(
        forecast,
        use_container_width=True
    )

    # Forecast chart

    historical = (
        df.set_index("Order_Date")
        .resample("MS")["Quantity"]
        .sum()
        .reset_index()
    )

    historical["Type"] = "Historical"

    future = forecast.copy()

    future["Type"] = "Forecast"

    future = future.rename(
        columns={
            "Forecast_Quantity": "Quantity"
        }
    )

    combined = pd.concat(
        [
            historical[
                ["Order_Date", "Quantity", "Type"]
            ],
            future[
                ["Order_Date", "Quantity", "Type"]
            ]
        ],
        ignore_index=True
    )

    fig = px.line(
        combined,
        x="Order_Date",
        y="Quantity",
        color="Type",
        markers=True,
        title="Historical Demand vs Forecast"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Forecast insight

    average_forecast = forecast[
        "Forecast_Quantity"
    ].mean()

    st.success(
        f"Expected average monthly demand "
        f"for the next 3 months: "
        f"approximately {average_forecast:,.0f} units."
    )

# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "RetailPulse AI | Data Analytics with AI"
)