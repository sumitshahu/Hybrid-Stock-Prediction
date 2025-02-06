
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import yfinance as yf

# Function to load stock data
def load_stock_data(stock_symbol, time_steps=60):
    data = yf.download(stock_symbol, period="60d", interval="1d")[['Close', 'High', 'Low', 'Volume']]  # Load last 60 days
    if data.empty:
        print(f"No data available for {stock_symbol}")
        return None
    return data

# Function to prepare input data
def prepare_input_data(data, scaler, time_steps=60):
    data_scaled = scaler.transform(data[['Close', 'High', 'Low', 'Volume']])
    last_60_days = np.array([data_scaled[-time_steps:]])  # Prepare the last 60 days data
    return last_60_days

# Load models and scaler
def load_models_and_scaler():
    # Load LSTM and GRU models
    lstm_model = load_model("/content/drive/MyDrive/Temp/lstm_model.h5")
    gru_model = load_model("/content/drive/MyDrive/Temp/gru_model.h5")

    # Load the meta-model (Linear Regression)
    with open("/content/drive/MyDrive/Temp/meta_model.pkl", "rb") as f:
        meta_model = pickle.load(f)

    # Load the scaler
    with open("/content/drive/MyDrive/Temp/scaler_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return lstm_model, gru_model, meta_model, scaler

# Function to calculate the range (upper and lower bound)

# Function to calculate the range (upper and lower bound)
def calculate_range(data, pred, std_multiplier=0.05):
    # Calculate the standard deviation of the last 60 closing prices
    recent_std = np.std(data['Close'][-60:].values) # Convert to NumPy array
    range_margin = recent_std * std_multiplier  # 5% of the standard deviation
    lower_bound = pred - range_margin
    upper_bound = pred + range_margin
    return lower_bound, upper_bound

# ... (rest of the code remains the same)# Function to predict tomorrow's stock price with range
def predict_tomorrow(stock_symbol):
    # Load the models and scaler
    lstm_model, gru_model, meta_model, scaler = load_models_and_scaler()

    # Load the stock data for the last 60 days
    data = load_stock_data(stock_symbol)
    if data is None:
        return

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
        return scaler.inverse_transform([[value, 0, 0, 0]])[0, 0]  # Corrected to extract scalar value

    pred_lstm_actual = inverse_transform(pred_lstm)
    pred_gru_actual = inverse_transform(pred_gru)
    pred_ensemble_actual = inverse_transform(pred_ensemble)

    # Calculate range for each model's prediction
    lower_bound_lstm, upper_bound_lstm = calculate_range(data, pred_lstm_actual)
    lower_bound_gru, upper_bound_gru = calculate_range(data, pred_gru_actual)
    lower_bound_ensemble, upper_bound_ensemble = calculate_range(data, pred_ensemble_actual)

    # Print the predictions and the predicted range for each model
    print("\nTomorrow's Predicted Closing Prices for", stock_symbol)
    print(f"LSTM Model Prediction: {pred_lstm_actual:.2f} INR (Range: {lower_bound_lstm:.2f} - {upper_bound_lstm:.2f} INR)")
    print(f"GRU Model Prediction: {pred_gru_actual:.2f} INR (Range: {lower_bound_gru:.2f} - {upper_bound_gru:.2f} INR)")
    print(f"Ensemble Model Prediction: {pred_ensemble_actual:.2f} INR (Range: {lower_bound_ensemble:.2f} - {upper_bound_ensemble:.2f} INR)")

# Example Usage
predict_tomorrow("RELIANCE.NS")