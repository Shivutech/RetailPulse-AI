import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

print("=" * 60)
print("RETAILPULSE AI V2 - DEMAND FORECASTING")
print("=" * 60)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("data/retail_data_v2.csv")
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

print("\nDataset loaded successfully")
print("Rows:", len(df))

# --------------------------------------------------
# 2. MONTHLY DEMAND
# --------------------------------------------------

monthly = (
    df.set_index("Order_Date")
      .resample("MS")
      .agg(
          Revenue=("Revenue", "sum"),
          Quantity=("Quantity", "sum"),
          Orders=("Order_ID", "count"),
          Customers=("Customer_ID", "nunique"),
          Profit=("Profit", "sum")
      )
      .reset_index()
)

print("\nMonthly data:")
print(monthly.tail())

# --------------------------------------------------
# 3. CREATE TIME FEATURES
# --------------------------------------------------

monthly["Month"] = monthly["Order_Date"].dt.month
monthly["Year"] = monthly["Order_Date"].dt.year

monthly["Month_Index"] = np.arange(len(monthly))

# Lag features
monthly["Quantity_Lag_1"] = monthly["Quantity"].shift(1)
monthly["Quantity_Lag_2"] = monthly["Quantity"].shift(2)
monthly["Quantity_Lag_3"] = monthly["Quantity"].shift(3)

monthly["Quantity_Rolling_3"] = (
    monthly["Quantity"]
    .shift(1)
    .rolling(3)
    .mean()
)

# Remove rows with missing lag values
model_data = monthly.dropna().copy()

# --------------------------------------------------
# 4. FEATURES
# --------------------------------------------------

features = [
    "Month",
    "Year",
    "Month_Index",
    "Quantity_Lag_1",
    "Quantity_Lag_2",
    "Quantity_Lag_3",
    "Quantity_Rolling_3"
]

X = model_data[features]
y = model_data["Quantity"]

# --------------------------------------------------
# 5. TEMPORAL TRAIN/TEST SPLIT
# --------------------------------------------------

split_index = int(len(model_data) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining months:", len(X_train))
print("Testing months :", len(X_test))

# --------------------------------------------------
# 6. TRAIN MODEL
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 7. EVALUATION
# --------------------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nMAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")

# --------------------------------------------------
# 8. TEST PREDICTIONS
# --------------------------------------------------

test_results = model_data.iloc[split_index:].copy()

test_results["Actual_Quantity"] = y_test.values
test_results["Predicted_Quantity"] = predictions

test_results["Prediction_Error"] = (
    test_results["Actual_Quantity"]
    - test_results["Predicted_Quantity"]
)

print("\nTEST PREDICTIONS")
print(
    test_results[
        [
            "Order_Date",
            "Actual_Quantity",
            "Predicted_Quantity"
        ]
    ].to_string(index=False)
)

# --------------------------------------------------
# 9. FUTURE FORECAST - NEXT 3 MONTHS
# --------------------------------------------------

forecast_data = monthly.copy()

future_forecasts = []

for i in range(3):

    next_date = (
        forecast_data["Order_Date"].max()
        + pd.DateOffset(months=1)
    )

    month = next_date.month
    year = next_date.year
    month_index = len(forecast_data)

    lag_1 = forecast_data["Quantity"].iloc[-1]
    lag_2 = forecast_data["Quantity"].iloc[-2]
    lag_3 = forecast_data["Quantity"].iloc[-3]

    rolling_3 = np.mean([lag_1, lag_2, lag_3])

    future_X = pd.DataFrame([{
        "Month": month,
        "Year": year,
        "Month_Index": month_index,
        "Quantity_Lag_1": lag_1,
        "Quantity_Lag_2": lag_2,
        "Quantity_Lag_3": lag_3,
        "Quantity_Rolling_3": rolling_3
    }])

    forecast_quantity = model.predict(future_X)[0]

    future_forecasts.append({
        "Order_Date": next_date,
        "Forecast_Quantity": round(forecast_quantity)
    })

    # Add forecast for recursive prediction
    new_row = pd.DataFrame([{
        "Order_Date": next_date,
        "Revenue": np.nan,
        "Quantity": forecast_quantity,
        "Orders": np.nan,
        "Customers": np.nan,
        "Profit": np.nan
    }])

    forecast_data = pd.concat(
        [forecast_data, new_row],
        ignore_index=True
    )

forecast_df = pd.DataFrame(future_forecasts)

print("\n" + "=" * 60)
print("NEXT 3 MONTH DEMAND FORECAST")
print("=" * 60)

print(
    forecast_df.to_string(index=False)
)

# --------------------------------------------------
# 10. SAVE FORECAST
# --------------------------------------------------

forecast_file = "data/demand_forecast.csv"

forecast_df.to_csv(
    forecast_file,
    index=False
)

# --------------------------------------------------
# 11. SAVE MODEL
# --------------------------------------------------

model_file = "models/demand_forecasting_model.pkl"

joblib.dump(model, model_file)

# --------------------------------------------------
# 12. SAVE MONTHLY DATA
# --------------------------------------------------

monthly.to_csv(
    "data/monthly_sales_data.csv",
    index=False
)

print("\n" + "=" * 60)
print("FILES CREATED")
print("=" * 60)

print("\nMonthly data       : data/monthly_sales_data.csv")
print("Forecast file      : data/demand_forecast.csv")
print("Forecasting model  : models/demand_forecasting_model.pkl")

print("\nDEMAND FORECASTING COMPLETED")
print("=" * 60)