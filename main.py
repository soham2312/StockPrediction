import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import seaborn as sns

# 1. Data understanding and EDA

# Load the data
df = pd.read_csv('stock_price.csv')

# Convert date column to datetime
df['日付け'] = pd.to_datetime(df['日付け'])

# Convert 'Volume' column to numeric by handling 'M' for million and 'B' for billion
df['出来高'] = df['出来高'].replace({'M': '*1e6', 'B': '*1e9'}, regex=True).map(pd.eval).astype(float)
# Remove the '%' sign from the 'Change Rate %' column and convert to float
df['変化率 %'] = df['変化率 %'].str.replace('%', '').astype(float) / 100

# Set date as index
df.set_index('日付け', inplace=True)

# Display basic information about the dataset
print(df.info())
print("\nBasic statistics:")
print(df.describe())

# Plot closing prices
plt.figure(figsize=(16, 8))
plt.title('NTT Stock Closing Price History')
plt.plot(df['終値'])
plt.xlabel('Date', fontsize=12)
plt.ylabel('Closing Price (JPY)', fontsize=12)
plt.show()

# 2. Data Preprocessing and Feature Engineering

# Select features for prediction
data = df[['終値', '始値', '高値', '安値', '出来高', '変化率 %']].copy()
data.columns = ['close', 'open', 'high', 'low', 'volume', 'change_rate']

# Create additional features
data['MA7'] = data['close'].rolling(window=7).mean()
data['MA30'] = data['close'].rolling(window=30).mean()
data['Daily_Return'] = data['close'].pct_change()
data['Volatility'] = data['Daily_Return'].rolling(window=30).std() * np.sqrt(252)

# Remove rows with NaN values
data.dropna(inplace=True)

# Split the data into training and testing sets based on date
split_date = '2023-01-01'
train_data = data[data.index < split_date]
test_data = data[data.index >= split_date]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

# Prepare data for LSTM model
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, 0])  # Predicting the closing price
    return np.array(X), np.array(y)

seq_length = 60
X_train, y_train = create_sequences(scaled_train_data, seq_length)
X_test, y_test = create_sequences(scaled_test_data, seq_length)

# 3. Model Selection and Training

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.1, verbose=1)

# 4. Model Evaluation and Results Analysis

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform predictions
train_predictions = scaler.inverse_transform(np.concatenate((train_predictions, np.zeros((len(train_predictions), scaled_train_data.shape[1]-1))), axis=1))[:, 0]
test_predictions = scaler.inverse_transform(np.concatenate((test_predictions, np.zeros((len(test_predictions), scaled_test_data.shape[1]-1))), axis=1))[:, 0]

# Get actual prices for comparison
train_actual = scaler.inverse_transform(scaled_train_data)[seq_length:, 0]
test_actual = scaler.inverse_transform(scaled_test_data)[seq_length:, 0]

# Calculate RMSE for train and test predictions
train_rmse = np.sqrt(np.mean((train_predictions - train_actual)**2))
test_rmse = np.sqrt(np.mean((test_predictions - test_actual)**2))

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")

# Plot results
plt.figure(figsize=(16, 8))
plt.title('Model Performance')
plt.plot(train_data.index[seq_length:], train_predictions, label='Train Predictions')
plt.plot(test_data.index[seq_length:], test_predictions, label='Test Predictions')
plt.plot(data.index[seq_length:], data['close'][seq_length:], label='Actual Prices')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Stock Price (JPY)', fontsize=12)
plt.legend()
plt.show()

# Plot training history
plt.figure(figsize=(16, 8))
plt.title('Model Training History')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.show()

# Compare predicted vs actual prices for test set
plt.figure(figsize=(16, 8))
plt.title('Predicted vs Actual Prices (Test Set)')
plt.plot(test_data.index[seq_length:], test_predictions, label='Predicted Prices')
plt.plot(test_data.index[seq_length:], test_actual, label='Actual Prices')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Stock Price (JPY)', fontsize=12)
plt.legend()
plt.show()

# Calculate and print the accuracy of price movement direction
def calculate_direction_accuracy(actual, predicted):
    actual_direction = np.sign(np.diff(actual))
    predicted_direction = np.sign(np.diff(predicted))
    correct_direction = np.sum(actual_direction == predicted_direction)
    return correct_direction / len(actual_direction) * 100

direction_accuracy = calculate_direction_accuracy(test_actual, test_predictions)
print(f"Price Movement Direction Accuracy: {direction_accuracy:.2f}%")