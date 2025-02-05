import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
import yfinance as yf
import pickle

# Function to load stock data
def load_stock_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)[['Close', 'High', 'Low', 'Volume']]
    return data

# Function to prepare input data
def prepare_input_data(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps, 0])  # Predicting 'Close' price
    return np.array(X), np.array(y)

# Stock details
stock_symbol = "RELIANCE.NS"
start_date = "2022-01-01"
end_date = "2025-02-05"

# Load and preprocess data
data = load_stock_data(stock_symbol, start_date, end_date)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Prepare data
time_steps = 60
X, y = prepare_input_data(data_scaled, time_steps)

# Split data
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Optimized LSTM model
lstm_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(time_steps, 4)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32),
    Dense(1)
])
lstm_model.compile(optimizer="adam", loss="mean_squared_error")
lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Save LSTM model
lstm_model.save("lstm_model.h5")

# Optimized GRU model
gru_model = Sequential([
    GRU(128, return_sequences=True, input_shape=(time_steps, 4)),
    Dropout(0.2),
    GRU(64, return_sequences=False),
    Dropout(0.2),
    Dense(32),
    Dense(1)
])
gru_model.compile(optimizer="adam", loss="mean_squared_error")
gru_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Save GRU model
gru_model.save("gru_model.h5")

# Make predictions
y_pred_lstm = lstm_model.predict(X_test)
y_pred_gru = gru_model.predict(X_test)

# Stacking predictions for ensemble learning
stacked_predictions = np.hstack([y_pred_lstm, y_pred_gru])

# Train meta-model
meta_model = LinearRegression()
meta_model.fit(stacked_predictions, y_test)

# Make final predictions using the ensemble model
ensemble_predictions = meta_model.predict(stacked_predictions)

# Evaluate models
mse_lstm = mean_squared_error(y_test, y_pred_lstm)
mse_gru = mean_squared_error(y_test, y_pred_gru)
mse_ensemble = mean_squared_error(y_test, ensemble_predictions)

r2_lstm = r2_score(y_test, y_pred_lstm) * 100
r2_gru = r2_score(y_test, y_pred_gru) * 100
r2_ensemble = r2_score(y_test, ensemble_predictions) * 100

print(f"LSTM Model Accuracy: {r2_lstm:.2f}% (MSE: {mse_lstm:.4f})")
print(f"GRU Model Accuracy: {r2_gru:.2f}% (MSE: {mse_gru:.4f})")
print(f"Ensemble Model Accuracy: {r2_ensemble:.2f}% (MSE: {mse_ensemble:.4f})")

# Function to plot predictions
def plot_predictions(y_test, y_pred, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual Prices", color="blue")
    plt.plot(y_pred, label=f"{model_name} Predictions", color="red")
    plt.title(f"{model_name} Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

# Plot predictions
plot_predictions(y_test, y_pred_lstm, "LSTM")
plot_predictions(y_test, y_pred_gru, "GRU")
plot_predictions(y_test, ensemble_predictions, "Ensemble Model")

# ** Predict Tomorrow’s Stock Price **
def predict_tomorrow():
    last_60_days = data_scaled[-time_steps:]  # Get last 60 days data
    last_60_days = np.array([last_60_days])  # Reshape for model input

    # Get predictions
    pred_lstm = lstm_model.predict(last_60_days)[0][0]
    pred_gru = gru_model.predict(last_60_days)[0][0]
    ensemble_input = np.array([[pred_lstm, pred_gru]])
    pred_ensemble = meta_model.predict(ensemble_input)[0]

    # Convert predictions back to actual price scale
    pred_lstm_actual = scaler.inverse_transform([[pred_lstm, 0, 0, 0]])[0][0]
    pred_gru_actual = scaler.inverse_transform([[pred_gru, 0, 0, 0]])[0][0]
    pred_ensemble_actual = scaler.inverse_transform([[pred_ensemble, 0, 0, 0]])[0][0]

    print("\nTomorrow's Predicted Closing Prices:")
    print(f"LSTM Model Prediction: {pred_lstm_actual:.2f} INR")
    print(f"GRU Model Prediction: {pred_gru_actual:.2f} INR")
    print(f"Ensemble Model Prediction: {pred_ensemble_actual:.2f} INR")

predict_tomorrow()
