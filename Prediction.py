##Baseline model : A logistic regression model based on three features : 
# 1. 50-day moving average : it is an average of the closing prices of the last 50 days.

# 2. 200-day moving average : average of the closing prices of the last 200 days.

# 3. RSI (Relative Strength Index) : helps in identifying whether a stock is overbought or oversold. If RSI is above 70, the stock is overbought (meaning it might be due for a price correction i.e. sell signal)

# - If RSI is below 30, stock is oversold (meaning it might be due for a price rebound i.e. buy signal)



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the data
data = pd.read_csv('TSLA.csv')

# Preview the data
print(data.head())

# Calculate 50-day and 200-day moving averages
data['50_MA'] = data['Close'].rolling(window=50).mean()
data['200_MA'] = data['Close'].rolling(window=200).mean()

# Calculate the RSI (Relative Strength Index)
# Calculate the difference between each element and the one before it
data['Price_Change'] = data['Close'].diff()

# Separate gains and losses
data['Gain'] = data['Price_Change'].where(data['Price_Change'] > 0, 0)
data['Loss'] = -data['Price_Change'].where(data['Price_Change'] < 0, 0)

# Calculate the average gain and loss over a 14-day period
data['Avg_Gain'] = data['Gain'].rolling(window=14).mean()
data['Avg_Loss'] = data['Loss'].rolling(window=14).mean()

# Calculate Relative Strength (RS)
data['RS'] = data['Avg_Gain'] / data['Avg_Loss']

# Calculate the RSI
data['RSI'] = 100 - (100 / (1 + data['RS']))

# Fill any NaN values in the RSI column with 50 (or any other default value)
data['RSI'].fillna(50, inplace=True)

# Drop intermediate columns if only the final RSI is needed
data.drop(['Price_Change', 'Gain', 'Loss', 'Avg_Gain', 'Avg_Loss', 'RS'], axis=1, inplace=True)

# Preview the updated data
print(data.head())

# Create the target variable (Hold by default)
data['Target'] = 0  # Hold (0) by default

# Set 'Buy' (1) if the next day's closing price is higher
data['Target'] = data['Target'].where(data['Close'].shift(-1) <= data['Close'], 1)

# Set 'Sell' (-1) if the next day's closing price is lower
data['Target'] = data['Target'].where(data['Close'].shift(-1) >= data['Close'], -1)

# Remove rows with missing values in the moving averages (these come from rolling calculations)
data.dropna(subset=['50_MA', '200_MA'], inplace=True)

# Split the data into features and target
X = data[['50_MA', '200_MA', 'RSI']]  # Features
y = data['Target']  # Target

# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now print the lengths of X_train and y_train
print(f"X_train length: {len(X_train)}, y_train length: {len(y_train)}")

# Initialize Logistic Regression and OneVsRestClassifier
log_reg = LogisticRegression(max_iter=10000)
ovr = OneVsRestClassifier(log_reg)

# Fit the model on the training data
ovr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ovr.predict(X_test)

# Print the first few predictions and actual values
print("Predicted labels:", y_pred[:10])
print("Actual labels:", y_test[:10].values)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
