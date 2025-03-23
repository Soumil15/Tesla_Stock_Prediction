## Note: This simulation only works for the future dates required for the assignment, i.e., March 24, 25, 26, 27, and 28.


## Please run the "Updated_Model.py" to check the model's performance 

import pandas as pd
import numpy as np

# === Load predicted dataset ===
df = pd.read_csv("Future_Market_Predictions.csv") 
df['Date'] = pd.to_datetime(df['Date'])

# === Filter only simulation window: March 24 to March 28 ===
start_date = pd.to_datetime("2025-03-24")
end_date = pd.to_datetime("2025-03-28")
df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy().reset_index(drop=True)

# === Initialize simulation ===
balance = 10000  # Starting capital
shares = 0
fee = 0.01
history = []

# === Trading Simulation (1-day lookahead using actual close prices) ===
for i in range(len(df) - 1):
    today = df.iloc[i]
    tomorrow = df.iloc[i + 1]

    actual_today = today['Close']
    actual_tomorrow = tomorrow['Close']

    action = "HOLD"
    shares_traded = 0
    amount_used = 0

    # Buy if tomorrow's actual price is expected to be higher
    if actual_tomorrow > actual_today and balance > actual_today * 5:
        amount_used = balance * 0.8 * (1 - fee)
        shares_traded = amount_used / actual_today
        balance -= amount_used
        shares += shares_traded
        action = "BUY"

    # Sell if tomorrow's actual price is expected to be lower
    elif actual_tomorrow < actual_today and shares > 0:
        proceeds = shares * actual_today * (1 - fee)
        balance += proceeds
        shares_traded = shares
        shares = 0
        action = "SELL"

    history.append({
        "Date": today['Date'],
        "Actual Price Today": round(actual_today, 2),
        "Actual Price Tomorrow": round(actual_tomorrow, 2),
        "Action": action,
        "Shares Traded": round(shares_traded, 2),
        "Shares Held": round(shares, 2),
        "Cash Balance": round(balance, 2),
        "Portfolio Value": round(balance + shares * actual_today, 2)
    })

# Log the final day
final_day = df.iloc[-1]
final_price = final_day['Close']
history.append({
    "Date": final_day['Date'],
    "Actual Price Today": round(final_price, 2),
    "Actual Price Tomorrow": None,
    "Action": "HOLD",
    "Shares Traded": 0,
    "Shares Held": round(shares, 2),
    "Cash Balance": round(balance, 2),
    "Portfolio Value": round(balance + shares * final_price, 2)
})


# === Output ===
df_result = pd.DataFrame(history)
print("\n===== Final Trading Simulation (March 24–28) with 1-Day Lookahead =====")
print(df_result)

# Final portfolio value
final_value = balance + (shares * final_price)
print(f"\n✅ Final Portfolio Value: ${final_value:.2f}")
