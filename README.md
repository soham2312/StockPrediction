# Stock Price Prediction using LSTM

This project demonstrates how to predict stock prices using Long Short-Term Memory (LSTM) models. The dataset used contains stock prices of NTT and includes fields such as open, close, high, low, volume, and daily percentage change. The model is designed to predict future stock prices based on past data, with additional features like moving averages and volatility to enhance performance.

## Project Overview

### Key Features:
- Exploratory Data Analysis (EDA): Analyzing trends in closing prices, daily returns, and volatility.
- Feature Engineering: Adding moving averages (MA7, MA30), daily returns, and volatility to better capture stock price movements.
- LSTM Model: Predict future closing prices using past data and engineered features.
- Performance Evaluation: The model is evaluated using Root Mean Squared Error (RMSE) and accuracy in predicting the price movement direction.

## Dataset:
- Fields:
  - Open
  - Close
  - High
  - Low
  - Volume (converted from M for millions and B for billions)
  - Change Rate (%)
- Time Range: The dataset contains stock price data up to the year 2023.

## Evaluation Metrics:
- RMSE (Root Mean Squared Error): Measures the difference between predicted and actual values.
- Price Movement Accuracy: Measures the accuracy in predicting whether the stock price will go up or down.

## Model Architecture

The model uses LSTM layers to capture the temporal dependencies of stock prices. The architecture is as follows:
- LSTM Layer 1: 50 units, returns sequences.
- LSTM Layer 2: 50 units, does not return sequences.
- Dense Layer 1: 25 units.
- Dense Layer 2: 1 unit (predicts the closing price).

## Installation and Requirements

To run this project locally, you'll need to install the following dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn keras seaborn


# Running the Project

To successfully run the project and replicate the results, follow these steps:

## 1. Clone the Repository
First, clone this repository to your local machine using:

```
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
```

2. Prepare the Dataset
Ensure that your stock price data file is in the root directory of the project. The dataset should be a .csv file named stock_price.csv, containing the following columns:

Date (as 日付け)
Open Price (as 始値)
Close Price (as 終値)
High Price (as 高値)
Low Price (as 安値)
Volume (as 出来高)
Change Rate % (as 変化率 %)

3. Run the Python Script
Once the data is prepared, run the main script to process the data, build the LSTM model, and make predictions. Use this command to execute the script:

```
python stock_price_prediction.py
```

4. View the Output
Model Training:
The script will train an LSTM model on the historical stock data, with progress displayed during the epochs.
Evaluation Metrics:
After training, you will see the RMSE values for both training and testing datasets.
The script will also calculate the accuracy of the model in predicting stock price movement (up or down).
Plots:
A comparison of predicted vs actual stock prices.
The training loss and validation loss over time.
A detailed plot comparing predicted prices with actual prices in the test dataset.

5. Analyze Results
Once the model is trained and tested, the following metrics will be printed in the console:

```
Train RMSE: X (replace with actual value)
Test RMSE: Y (replace with actual value)
Direction Accuracy: Z% (replace with actual value)
```