import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# === Load dataset ===
df = pd.read_csv("Final_Corrected_Tesla_Data_with_Logical_Targets_updated.csv")
df['Date'] = pd.to_datetime(df['Date'])

# === Add previous days' closing prices as features ===
df['Prev_Close_1'] = df['Close'].shift(1)
df['Prev_Close_2'] = df['Close'].shift(2)
df['Prev_Close_3'] = df['Close'].shift(3)
df.bfill(inplace=True)  # Fill NaNs caused by shifting

# === Feature and target selection ===
feature_columns = [
    'Open', 'High', 'Low', 'Volume',
    'Williams_%R', 'ATR', 'MACD_Histogram', 'OBV_EMA', 'SMA_10', 'EMA_26',
    'Momentum', 'MACD', 'SMA_50', 'RSI', 'Prev_Close_1', 'Prev_Close_2', 'Prev_Close_3',
    'Stochastic_%K', 'Stochastic_%D', 'BB_Upper', 'BB_Lower', 'BB_Width'
]
X = df[feature_columns]
y = df['Close']

# === Normalize features ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === Add year column for filtering ===
df['Year'] = df['Date'].dt.year

# === Variable to control test year (TA can change this) ===
year_to_test = 2019  # <-- TA can change this to any year in the dataset

# === Split into train/test based on year ===
X_train = X_scaled[df['Year'] != year_to_test]
y_train = y[df['Year'] != year_to_test]
X_test = X_scaled[df['Year'] == year_to_test]
y_test = y[df['Year'] == year_to_test]
dates_test = df['Date'][df['Year'] == year_to_test].reset_index(drop=True)

# === Train the model ===
xgb_regressor = XGBRegressor(
    n_estimators=300,
    max_depth=20,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.8,
    random_state=42
)
xgb_regressor.fit(X_train, y_train)

# === Predict ===
y_pred_train = xgb_regressor.predict(X_train)
y_pred_test = xgb_regressor.predict(X_test)

# === Evaluation Metrics ===
print("\n==== Evaluation Metrics ====")
print(f"Test Year: {year_to_test}")
print("Train Set:")
print(f"  MAE:  {mean_absolute_error(y_train, y_pred_train):.2f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f}")
print("Test Set:")
print(f"  MAE:  {mean_absolute_error(y_test, y_pred_test):.2f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")

# === Show 7 sample predictions from the test year ===
print(f"\n==== Sample Predictions for Year {year_to_test} ====")
sample_df = pd.DataFrame({
    'Date': dates_test,
    'Actual_Close': y_test.values,
    'Predicted_Close': y_pred_test
})
print(sample_df.head(7).to_string(index=False))

# # === Trading Simulation ===
# def run_simulation_on_year(year, starting_balance=10000, fee=0.01):
#     print(f"\n==== Trading Simulation for Year {year} ====")
#     df_sim = df[df['Year'] == year].copy().reset_index(drop=True)
    
#     if df_sim.shape[0] < 2:
#         print("Not enough data to simulate.")
#         return

#     # Add predictions to df_sim
#     X_sim = scaler.transform(df_sim[feature_columns])
#     df_sim['Predicted_Close'] = xgb_regressor.predict(X_sim)

#     balance = starting_balance
#     shares = 0
#     history = []

#     for i in range(len(df_sim) - 1):  # up to second-last day
#         today = df_sim.iloc[i]
#         tomorrow = df_sim.iloc[i + 1]

#         predicted_today = today['Predicted_Close']
#         predicted_tomorrow = tomorrow['Predicted_Close']
#         actual_price = today['Close']

#         action = "HOLD"
#         shares_traded = 0
#         amount_used = 0

#         if predicted_tomorrow > predicted_today and balance > actual_price * 5:
#             amount_used = balance * 0.8 * (1 - fee)
#             shares_traded = amount_used / actual_price
#             balance -= amount_used
#             shares += shares_traded
#             action = "BUY"

#         elif predicted_tomorrow < predicted_today and shares > 0:
#             proceeds = shares * actual_price * (1 - fee)
#             balance += proceeds
#             shares_traded = shares
#             shares = 0
#             action = "SELL"

#         history.append({
#             "Date": today['Date'],
#             "Actual Price": round(actual_price, 2),
#             "Predicted Today": round(predicted_today, 2),
#             "Predicted Tomorrow": round(predicted_tomorrow, 2),
#             "Action": action,
#             "Shares Traded": round(shares_traded, 2),
#             "Shares Held": round(shares, 2),
#             "Cash Balance": round(balance, 2),
#             "Portfolio Value": round(balance + shares * actual_price, 2)
#         })

#     # Final day (no tomorrow)
#     final_day = df_sim.iloc[-1]
#     final_price = final_day['Close']
#     history.append({
#         "Date": final_day['Date'],
#         "Actual Price": round(final_price, 2),
#         "Predicted Today": round(final_day['Predicted_Close'], 2),
#         "Predicted Tomorrow": None,
#         "Action": "HOLD",
#         "Shares Traded": 0,
#         "Shares Held": round(shares, 2),
#         "Cash Balance": round(balance, 2),
#         "Portfolio Value": round(balance + shares * final_price, 2)
#     })

#     # Display results
#     df_result = pd.DataFrame(history)
#     print(df_result.head(10).to_string(index=False))
#     print(f"\n✅ Final Portfolio Value: ${df_result.iloc[-1]['Portfolio Value']:.2f}")

# # === Run the simulation on the selected year ===
# run_simulation_on_year(year_to_test)
