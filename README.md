# RetailPulse AI

## AI-Powered Business Intelligence & Decision Support System

RetailPulse AI is an end-to-end retail analytics platform that transforms raw transaction data into **business insights, predictive intelligence, and actionable recommendations**.

The project follows the pipeline:

**Data → Information → Insight → Decision → Action**

---

## 📌 Project Overview

Retail businesses generate large amounts of transactional data, but raw data alone does not provide clear answers to important business questions.

RetailPulse AI combines:

- Exploratory Data Analysis
- Customer RFM Segmentation
- Customer Churn Prediction
- Demand Forecasting
- Anomaly Detection
- Interactive Business Dashboard
- Generative AI-powered Business Analyst

The goal is to help businesses understand sales performance, customer behavior, churn risk, future demand, and unusual business activity from a single dashboard.

---

## 🎯 Objectives

- Analyze retail sales and profitability.
- Identify high-performing categories, products, and regions.
- Segment customers according to purchasing behavior.
- Predict customers who are at risk of churn.
- Estimate future product demand.
- Detect unusual monthly business activity.
- Provide AI-generated business analysis using actual project data.
- Present results through an interactive Streamlit dashboard.

---

## 🚀 Key Features

### 1. 📊 Business Overview

Provides high-level KPIs including:

- Total Revenue
- Total Profit
- Total Orders
- Total Customers
- Units Sold
- Average Order Value
- Profit Margin

---

### 2. 📈 Sales Analytics

Interactive analysis of:

- Revenue trends
- Profit trends
- Category performance
- Regional performance
- Product performance
- Payment modes
- Customer types

Users can apply filters to explore the business data.

---

### 3. 👥 Customer Intelligence

Customer behavior is analyzed using **RFM Analysis**:

- **Recency**: How recently a customer purchased
- **Frequency**: How frequently a customer purchased
- **Monetary**: How much a customer spent

Customers are grouped into segments such as:

- Champions
- Loyal Customers
- Potential Loyalists
- At Risk
- Lost Customers
- Others

#### RFM Results

| Segment | Customers | Share |
|---|---:|---:|
| Champions | 2,246 | 22.46% |
| Loyal Customers | 1,298 | 12.98% |
| At Risk | 1,130 | 11.30% |
| Others | 1,675 | 16.75% |
| Potential Loyalists | 1,215 | 12.15% |
| Lost Customers | 2,436 | 24.36% |

---

### 4. 🤖 Customer Churn Prediction

A machine learning model predicts customer churn risk using a temporal train/prediction setup.

Features include:

- Recency
- Frequency
- Monetary
- Total Quantity
- Average Order Value
- Average Rating
- Average Delivery Days

#### Model Performance

- Accuracy: **69.47%**
- ROC-AUC: **0.7701**

#### Churn Risk Distribution

- High Risk: 3,557
- Medium Risk: 4,328
- Low Risk: 2,071

The system also estimates revenue associated with customers at risk.

---

### 5. 🔮 Demand Forecasting

Monthly sales quantity is forecasted using a machine learning model.

Dataset period:

**January 2024 – June 2026**

Model evaluation:

- MAE: **361.90**
- RMSE: **413.23**

Next three-month baseline forecast:

| Month | Forecast Quantity |
|---|---:|
| July 2026 | 4,643 |
| August 2026 | 4,674 |
| September 2026 | 4,616 |

The forecast provides a short-term demand baseline that can support inventory and planning decisions.

---

### 6. 🚨 Anomaly Detection

Isolation Forest is used to identify unusual monthly business activity.

The analysis identified:

- 25 normal months
- 5 anomalous months

Detected anomaly months include:

- January 2025
- February 2025
- February 2026
- May 2026
- June 2026

The dashboard allows users to inspect unusual revenue, profit, and quantity patterns.

---

### 7. 🧠 AI Business Analyst

RetailPulse AI integrates Google's Gemini API to provide natural-language business analysis.

Users can ask questions such as:

> Which category generates the highest revenue?

> Which customer segment contributes the most revenue?

> What is the current churn risk?

> What does the demand forecast indicate?

The AI Business Analyst receives actual project metrics as business context and is instructed not to invent information outside the available data.

Example:

**Question:** Which category generates the highest revenue?

**Result:** Electronics generates **₹1,726,519,342.50** in revenue.

This converts dashboard data into understandable business explanations and recommendations.

---

## 📊 Dataset

The project uses a synthetic retail transaction dataset generated specifically for analytics and machine learning experimentation.

### Dataset Statistics

- **83,573** transactions
- **10,000** customers
- **18** columns
- Date range: **January 2024 – June 2026**
- No missing values
- No duplicate records

### Main Columns

```text
Order_ID
Order_Date
Customer_ID
Product
Category
Region
City
Quantity
Unit_Price
Discount
Revenue
Cost
Profit
Payment_Mode
Customer_Type
Customer_Rating
Delivery_Days
Customer_Profile
```

---

## 💡 Key Business Insights

### Revenue

- Total Revenue: **₹3,543,550,355.05**
- Total Profit: **₹1,062,426,893.06**
- Profit Margin: **29.98%**
- Average Order Value: **₹42,400.66**

### Category

**Electronics** is the highest-revenue category with approximately **₹1.73 billion** in revenue.

### Region

**North** is the highest-revenue region with approximately **₹1.07 billion** in revenue.

### Product

**Laptop** is the highest-revenue product with approximately **₹679.79 million** in revenue.

### Customers

- Returning Customers: **9,357**
- New Customers: **643**

---

## 🧠 Machine Learning Components

| Component | Purpose | Output |
|---|---|---|
| RFM Analysis | Customer segmentation | Customer segments |
| Churn Prediction | Identify churn risk | Risk levels |
| Demand Forecasting | Estimate future demand | Forecast quantity |
| Isolation Forest | Detect unusual activity | Anomaly status |

---

## 🛠️ Technology Stack

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- XGBoost
- Joblib

### Visualization

- Matplotlib
- Seaborn
- Plotly

### Dashboard

- Streamlit

### Generative AI

- Google Gemini API
- `google-genai`

### Environment

- Python Virtual Environment
- python-dotenv

### Development Tools

- VS Code
- Jupyter Notebook
- Git
- GitHub

---

## 📁 Project Structure

```text
RetailPulse-AI/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── retail_data_v2.csv
│   ├── customer_rfm_v2.csv
│   ├── customer_churn_predictions_v2.csv
│   ├── monthly_sales_data.csv
│   ├── demand_forecast.csv
│   └── anomaly_detection_results.csv
│
├── models/
│   ├── churn_model_v2.pkl
│   └── demand_forecasting_model.pkl
│
├── notebooks/
│
├── screenshots/
│
├── src/
│   ├── generate_data_v2.py
│   ├── data_analysis_v2.py
│   ├── customer_analysis_v2.py
│   ├── churn_model_v2.py
│   ├── demand_forecasting.py
│   ├── anomaly_detection.py
│   └── ai_business_analyst.py
│
└── venv/
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd RetailPulse-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

#### Windows

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Gemini API Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

Never upload `.env` or your API key to GitHub.

The project already includes `.env` in `.gitignore`.

---

## ▶️ Run the Application

From the project root:

```bash
streamlit run app.py
```

The Streamlit dashboard will open in your browser.

---

## 🖥️ Dashboard Sections

RetailPulse AI contains the following dashboard pages:

1. **📊 Overview**
2. **📈 Sales Analytics**
3. **👥 Customer Intelligence**
4. **🤖 AI Predictions**
5. **🧠 AI Business Analyst**
6. **🚨 Anomaly Detection**
7. **🔮 Demand Forecast**

---

## 🔄 System Workflow

```text
                    Retail Transaction Data
                              │
                              ▼
                     Data Preparation
                              │
                              ▼
                  Exploratory Data Analysis
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
        RFM Analysis    Churn Prediction   Demand Forecast
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                    Anomaly Detection
                              │
                              ▼
                  Streamlit BI Dashboard
                              │
                              ▼
                   AI Business Analyst
                              │
                              ▼
                Business Insights & Actions
```

---

## 📈 Project Metrics Summary

| Metric | Result |
|---|---:|
| Transactions | 83,573 |
| Customers | 10,000 |
| Revenue | ₹3.54B |
| Profit | ₹1.06B |
| Profit Margin | 29.98% |
| Average Order Value | ₹42,400.66 |
| Churn Accuracy | 69.47% |
| Churn ROC-AUC | 0.7701 |
| Demand Forecast MAE | 361.90 |
| Demand Forecast RMSE | 413.23 |
| Detected Anomalies | 5 |

---

## 🔒 Data & API Security

The project follows basic security practices:

- API keys are stored in `.env`.
- `.env` is excluded through `.gitignore`.
- No API key should be committed to GitHub.
- Generated business data is stored locally in the `data/` directory.

---

## 🔮 Future Enhancements

Potential future improvements include:

- Real-time database integration
- Live sales data ingestion
- Automated email alerts
- Advanced time-series forecasting
- Product-level demand forecasting
- Recommendation engine
- Customer lifetime value prediction
- Automated business reports
- Cloud deployment
- Role-based dashboard access
- More advanced conversational analytics

---

## 👨‍💻 Author

**Vivek Pal**

B.Tech in Computer Science & Engineering  
Specialization: Artificial Intelligence & Machine Learning

---

## ⭐ Project Vision

RetailPulse AI aims to bridge the gap between **data analytics, machine learning, and business decision-making**.

Instead of simply displaying numbers, the system transforms business data into:

**Insights → Predictions → Recommendations → Decisions**

