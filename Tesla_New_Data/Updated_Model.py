import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load dataset
df = pd.read_csv("HistoricalData_With_Indicators_Full.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Add lag features
df['Prev_Close_1'] = df['Close'].shift(1)
df['Prev_Close_2'] = df['Close'].shift(2)
df['Prev_Close_3'] = df['Close'].shift(3)
df.bfill(inplace=True)

# Feature set
feature_columns = [
    'Open', 'High', 'Low', 'Volume',
    'RSI', 'MACD', 'MACD_Histogram', 'MACD_Signal',
    'OBV_EMA', 'SMA_10', 'SMA_50', 'EMA_12', 'EMA_26',
    'ATR', 'Momentum', 'Williams_%R',
    'Stochastic_%K', 'Stochastic_%D',
    'BB_Upper', 'BB_Lower', 'BB_Width',
    'Prev_Close_1', 'Prev_Close_2', 'Prev_Close_3'
]
X = df[feature_columns]
y = df['Close']

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Add Year column for filtering
df['Year'] = df['Date'].dt.year
test_year = 2024  # <-- Change this to test other years

# Train-test split by year
X_train = X_scaled[df['Year'] != test_year]
y_train = y[df['Year'] != test_year]
X_test = X_scaled[df['Year'] == test_year]
y_test = y[df['Year'] == test_year]
dates_test = df['Date'][df['Year'] == test_year].reset_index(drop=True)

# Train XGBoost Regressor
xgb_regressor = XGBRegressor(
    n_estimators=300,
    max_depth=20,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.8,
    random_state=42
)
xgb_regressor.fit(X_train, y_train)

# Predictions
y_pred_train = xgb_regressor.predict(X_train)
y_pred_test = xgb_regressor.predict(X_test)

# Evaluation metrics
print("\n==== Evaluation Metrics ====")
print(f"Test Year: {test_year}")
print("Train Set:")
print(f"  MAE:  {mean_absolute_error(y_train, y_pred_train):.2f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f}")
print("Test Set:")
print(f"  MAE:  {mean_absolute_error(y_test, y_pred_test):.2f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")

# Sample predictions 
print(f"\n==== Sample Predictions for Year {test_year} ====")
sample_df = pd.DataFrame({
    'Date': dates_test,
    'Actual_Close': y_test.values,
    'Predicted_Close': y_pred_test
})
print(sample_df.head(5).to_string(index=False))

# # # Save model and scaler (if needed)
# xgb_regressor.save_model("xgboost_regressor_model.json")
# joblib.dump(scaler, "scaler_regressor.pkl")
