import os
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import sys
import tensorflow as tf

tf.compat.v1.experimental.output_all_intermediates(True)
tf.config.run_functions_eagerly(True)

# Base directory where models are stored
BASE_DIR = r"C:\Users\sumit\Final year\stock-market-analysis\models"

# Get today's date
TODAY = TODAY = datetime.today().strftime('%Y-%m-%d')

def get_latest_folder(stock_dir):
    """Find the latest date folder inside the stock directory."""
    try:
        date_folders = sorted(os.listdir(stock_dir), reverse=True)
        for folder in date_folders:
            if folder <= TODAY:
                return folder
    except Exception as e:
        print(f"Error finding latest folder for {stock_dir}: {e}")
    return None

def load_scaler(scaler_path):
    """Load the scaler from the given path."""
    with open(scaler_path, 'rb') as scaler_file:
        return pickle.load(scaler_file)

def fetch_latest_data(stock_name):
    """Fetch the last 60 days of stock data, including today's data if available."""
    
    # Fetch last 90 days (to ensure market open days are covered)
    stock_data = yf.download(stock_name, period="90d")  # Ensures at least 60 market days
    
    if stock_data.empty:
        print(f" No data found for {stock_name}. Skipping update.")
        return None
    # Check if the last row's date is today
    last_row_date = stock_data.index[-1].strftime('%Y-%m-%d')
    if last_row_date != TODAY:
        print(f" Last row date {last_row_date} is not today's date {TODAY}. Skipping update.")
        return

    # Select only the required columns
    stock_data = stock_data[['Close', 'High', 'Low', 'Volume']]

    # Get exactly the last 60 available trading days
    if len(stock_data) < 61:
        print(f" Not enough data for {stock_name} (Only {len(stock_data)} rows). Skipping update.")
        return None
    return stock_data.tail(61)  # Ensure exactly 60 time steps



def retrain_models(stock_name):
    """Retrains and updates models for a given stock."""
    stock_dir = os.path.join(BASE_DIR, stock_name)
    latest_folder = get_latest_folder(stock_dir)
    if not latest_folder:
        print(f" No previous models found for {stock_name}. Skipping.")
        return

    latest_path = os.path.join(stock_dir, latest_folder)
    today_path = os.path.join(stock_dir, TODAY)
    os.makedirs(today_path, exist_ok=True)

    print(f" Using models from directory: {latest_path} for {stock_name}")  #  Print Model Directory

    # Load previous models & scaler
    try:
        #  Fix loss function issue
        lstm_model = load_model(os.path.join(latest_path, "lstm_model.h5"), compile=False)
        gru_model = load_model(os.path.join(latest_path, "gru_model.h5"), compile=False)

        #  Explicitly compile models
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        gru_model.compile(optimizer='adam', loss='mean_squared_error')

        with open(os.path.join(latest_path, "meta_model.pkl"), "rb") as meta_file:
            meta_model = pickle.load(meta_file)
        scaler = load_scaler(os.path.join(latest_path, "scaler.pkl"))

    except Exception as e:
        print(f" Error loading models/scaler for {stock_name}: {e}")
        return

    # Fetch new data
    stock_data = fetch_latest_data(stock_name)
    if stock_data is None or stock_data.empty:
        print(f" No new data fetched for {stock_name}. Skipping.")
        return

    stock_data_scaled = scaler.transform(stock_data)

    # Extract X (features) and y (target)
    X = stock_data_scaled[:-1]  # First 60 rows
    y = stock_data_scaled[-1, 0]  # 61st row's 'Close' price (Target)

    # Reshape X for LSTM (batch_size, time_steps, features)
    X_train = X.reshape(1, 60, 4)
    y_train = np.array([y])  # Ensure NumPy format

    try:
        lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
        gru_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

        y_pred_lstm = lstm_model.predict(X_train)
        y_pred_gru = gru_model.predict(X_train)
        stacked_predictions = np.hstack([y_pred_lstm, y_pred_gru])

        meta_model.fit(stacked_predictions, y_train)

        # Save updated models
        lstm_model.save(os.path.join(today_path, "lstm_model.h5"))
        gru_model.save(os.path.join(today_path, "gru_model.h5"))
        with open(os.path.join(today_path, "meta_model.pkl"), "wb") as meta_file:
            pickle.dump(meta_model, meta_file)
        with open(os.path.join(today_path, "scaler.pkl"), "wb") as scaler_file:
            pickle.dump(scaler, scaler_file)

        print(f"Models successfully updated for {stock_name} on {TODAY}!")

    except Exception as e:
        print(f" Error during model training/saving for {stock_name}: {e}")




# Get all stock names from BASE_DIR
stock_list = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

# Run updates for all stocks
for stock in stock_list:
    retrain_models(stock)