import os
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Directory setup
BASE_DIR = r"C:\Users\sumit\Final year\stock-market-analysis\models"

def get_latest_model_directory(stock_name):
    stock_dir = os.path.join(BASE_DIR, stock_name)
    if not os.path.exists(stock_dir):
        return None
    date_folders = sorted(os.listdir(stock_dir), reverse=True)
    for folder in date_folders:
        try:
            datetime.strptime(folder, "%Y-%m-%d")
            return os.path.join(stock_dir, folder)
        except ValueError:
            continue
    return None

def load_models_and_scaler(stock_name):
    latest_path = get_latest_model_directory(stock_name)
    if not latest_path:
        return None, None, None, None
    lstm_model = load_model(os.path.join(latest_path, "lstm_model.h5"))
    gru_model = load_model(os.path.join(latest_path, "gru_model.h5"))
    with open(os.path.join(latest_path, "meta_model.pkl"), "rb") as f:
        meta_model = pickle.load(f)
    with open(os.path.join(latest_path, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return lstm_model, gru_model, meta_model, scaler

def fetch_stock_data(stock_name):
    df = yf.download(stock_name, period="120d", interval="1d")
    df = df[['Close', 'High', 'Low', 'Volume']].dropna()
    return df

def prepare_sequences(df, scaler, time_steps=60):
    data_scaled = scaler.transform(df[['Close', 'High', 'Low', 'Volume']])
    X, y = [], []
    for i in range(time_steps, len(data_scaled)):
        X.append(data_scaled[i-time_steps:i])
        y.append(df['Close'].values[i])
    return np.array(X), np.array(y)

def predict_models(stock_name):
    lstm_model, gru_model, meta_model, scaler = load_models_and_scaler(stock_name)
    if not all([lstm_model, gru_model, meta_model, scaler]):
        print(f"Models missing for {stock_name}")
        return

    df = fetch_stock_data(stock_name)
    X, y_actual = prepare_sequences(df, scaler)
    dates = df.index[-len(y_actual):].strftime('%Y-%m-%d')

    lstm_preds = lstm_model.predict(X).flatten()
    gru_preds = gru_model.predict(X).flatten()
    ensemble_input = np.vstack((lstm_preds, gru_preds)).T
    ensemble_preds = meta_model.predict(ensemble_input)

    # Inverse transform only the Close column
    def inverse(value):
        return scaler.inverse_transform([[value, 0, 0, 0]])[0, 0]

    lstm_preds_actual = [inverse(v) for v in lstm_preds]
    gru_preds_actual = [inverse(v) for v in gru_preds]
    ensemble_preds_actual = [inverse(v) for v in ensemble_preds]

    plot_comparisons(stock_name, dates, y_actual, lstm_preds_actual, gru_preds_actual, ensemble_preds_actual)

def plot_comparisons(stock_name, dates, actual, lstm, gru, ensemble):
    output_dir = os.path.join("comparison_plots", stock_name)
    os.makedirs(output_dir, exist_ok=True)

    # Plot individual comparisons
    def save_plot(title, actual, pred, label, filename):
        plt.figure(figsize=(12, 5))
        plt.plot(dates, actual, label='Actual', linewidth=2)
        plt.plot(dates, pred, label=label, linestyle='--')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price (INR)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    save_plot('LSTM vs Actual', actual, lstm, 'LSTM', 'lstm_vs_actual.png')
    save_plot('GRU vs Actual', actual, gru, 'GRU', 'gru_vs_actual.png')
    save_plot('Ensemble vs Actual', actual, ensemble, 'Ensemble', 'ensemble_vs_actual.png')

    # Combined comparison
    plt.figure(figsize=(14, 6))
    plt.plot(dates, actual, label='Actual', linewidth=2)
    plt.plot(dates, lstm, label='LSTM', linestyle='--')
    plt.plot(dates, gru, label='GRU', linestyle='-.')
    plt.plot(dates, ensemble, label='Ensemble', linestyle=':')
    plt.title('Model Comparison (LSTM, GRU, Ensemble) vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_all.png'))
    plt.close()

    print(f"Plots saved in: {output_dir}")


# Specify the stock you want to process
stock_name = "RELIANCE.NS"  # Change this to any stock you want

print(f"Processing {stock_name}...")
predict_models(stock_name)
