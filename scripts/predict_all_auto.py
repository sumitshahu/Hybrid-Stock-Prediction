import os
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Base directory where models are stored
BASE_DIR = r"C:\Users\sumit\Final year\stock-market-analysis\models"
# Get today's date
TODAY = datetime.today().strftime('%Y-%m-%d')

def get_latest_model_directory(stock_name):
    """Finds the latest date directory for a given stock."""
    stock_dir = os.path.join(BASE_DIR, stock_name)
    if not os.path.exists(stock_dir):
        print(f"No directory found for {stock_name}. Skipping.")
        return None

    # Get all date directories and sort in descending order
    date_folders = sorted(os.listdir(stock_dir), reverse=True)

    for folder in date_folders:
        # Ensure it's a valid date format (YYYY-MM-DD)
        try:
            datetime.strptime(folder, "%Y-%m-%d")
            return os.path.join(stock_dir, folder)
        except ValueError:
            continue  # Skip non-date folders

    print(f"No valid date folders found for {stock_name}.")
    return None

def load_models_and_scaler(stock_name):
    """Dynamically loads the latest models and scaler for a given stock."""
    latest_path = get_latest_model_directory(stock_name)
    if not latest_path:
        return None, None, None, None, None  # Skip if no latest directory found

    try:
        # Load models
        lstm_model = load_model(os.path.join(latest_path, "lstm_model.h5"))
        gru_model = load_model(os.path.join(latest_path, "gru_model.h5"))

        # Load the meta-model
        with open(os.path.join(latest_path, "meta_model.pkl"), "rb") as f:
            meta_model = pickle.load(f)

        # Load the scaler
        with open(os.path.join(latest_path, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)

        print(f"Loaded models and scaler from {latest_path} for {stock_name}")
        return lstm_model, gru_model, meta_model, scaler, latest_path

    except Exception as e:
        print(f" Error loading models/scaler for {stock_name}: {e}")
        return None, None, None, None, None

def load_stock_data(stock_symbol, time_steps=60):
    """Load last 60 days of stock data from Yahoo Finance."""
    try:
        data = yf.download(stock_symbol, period="60d", interval="1d")[['Close', 'High', 'Low', 'Volume']]
        if data.empty:
            print(f"No data available for {stock_symbol}")
            return None
        return data
    except Exception as e:
        print(f"Error fetching data for {stock_symbol}: {e}")
        return None

def prepare_input_data(data, scaler, time_steps=60):
    """Prepares the last 60 days of data for model input."""
    data_scaled = scaler.transform(data[['Close', 'High', 'Low', 'Volume']])
    last_60_days = np.array([data_scaled[-time_steps:]])  # Prepare the last 60 days data
    return last_60_days

def calculate_range(data, pred, std_multiplier=0.05):
    """Calculates the range (upper and lower bound) for predictions."""
    recent_std = np.std(data['Close'][-60:].values)  # Convert to NumPy array
    range_margin = recent_std * std_multiplier  # 5% of the standard deviation
    lower_bound = pred - range_margin
    upper_bound = pred + range_margin
    return lower_bound, upper_bound

def get_prediction_date():
    """Returns the correct date for which predictions are made."""
    now = datetime.now()
    if now.hour < 17:  # Before 5 PM, predict for today
        prediction_date = now.strftime('%Y-%m-%d')
    else:  # After 5 PM, predict for tomorrow
        prediction_date = (now + timedelta(days=1)).strftime('%Y-%m-%d')
    return prediction_date

def predict_tomorrow(stock_symbol):
    """Predict tomorrow's stock price using the latest models and save results."""
    lstm_model, gru_model, meta_model, scaler, latest_path = load_models_and_scaler(stock_symbol)
    if not latest_path:
        return None  # Skip if no models available

    # Load the stock data for the last 60 days
    data = load_stock_data(stock_symbol)
    if data is None:
        return None

    # Prepare input data for predictions
    last_60_days = prepare_input_data(data, scaler)

    # Get predictions from LSTM and GRU models
    pred_lstm = lstm_model.predict(last_60_days)[0][0]
    pred_gru = gru_model.predict(last_60_days)[0][0]

    # Stack predictions and predict using the ensemble model
    ensemble_input = np.array([[pred_lstm, pred_gru]])
    pred_ensemble = meta_model.predict(ensemble_input)[0]

    # Convert predictions back to actual price scale
    def inverse_transform(value):
        return scaler.inverse_transform([[value, 0, 0, 0]])[0, 0]  # Extract scalar value

    pred_lstm_actual = inverse_transform(pred_lstm)
    pred_gru_actual = inverse_transform(pred_gru)
    pred_ensemble_actual = inverse_transform(pred_ensemble)

    # Calculate range for each model's prediction
    lower_bound_lstm, upper_bound_lstm = calculate_range(data, pred_lstm_actual)
    lower_bound_gru, upper_bound_gru = calculate_range(data, pred_gru_actual)
    lower_bound_ensemble, upper_bound_ensemble = calculate_range(data, pred_ensemble_actual)

    # Store results in a dictionary
    prediction_results = {
        "Model": ["LSTM", "GRU", "Ensemble"],
        "Specific Price (INR)": [pred_lstm_actual, pred_gru_actual, pred_ensemble_actual],
        "Range": [
            f"{lower_bound_lstm:.2f} - {upper_bound_lstm:.2f}",
            f"{lower_bound_gru:.2f} - {upper_bound_gru:.2f}",
            f"{lower_bound_ensemble:.2f} - {upper_bound_ensemble:.2f}"
        ]
    }

    # Convert to DataFrame
    df_results = pd.DataFrame(prediction_results)

    # Save to Excel in the same folder where models are stored
    excel_path = os.path.join(latest_path, f"{stock_symbol}_predictions.xlsx")
    df_results.to_excel(excel_path, index=False)
    
    print(f" Predictions saved for {stock_symbol}: {excel_path}")

# Get all stock names from BASE_DIR
stock_list = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

# Run predictions for all stocks
for stock in stock_list:
    print(f"\nProcessing: {stock}")
    predict_tomorrow(stock)
