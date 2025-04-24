from django.shortcuts import render
from django.http import JsonResponse
import yfinance as yf
import requests
import numpy as np
import os
import pandas as pd
import pytz
from datetime import datetime
import matplotlib.pyplot as plt

# Paths
DATA_DIR = r"C:\Users\sumit\Final year\stock-market-analysis\models"

# List of stocks
STOCKS = [
    'SUNPHARMA.NS', 'DRREDDY.NS', 'ITC.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS',
    'ASIANPAINT.NS', 'BRITANNIA.NS', 'TCS.NS', 'PIDILITIND.NS', 'HDFCBANK.NS',
    'INFY.NS', 'RELIANCE.NS', 'BHARTIARTL.NS', 'SHREECEM.NS', 'BAJAJFINSV.NS',
]

def index(request):
    return render(request, 'my_app/index.html')

def search_stocks(request):
    query = request.GET.get('query', '').lower()
    matching_stocks = [stock for stock in STOCKS if query in stock.lower()]
    return JsonResponse(matching_stocks, safe=False)

def is_market_open():
    now = datetime.now(pytz.timezone("Asia/Kolkata"))
    return now.weekday() < 5 and 9 <= now.hour < 15

def get_live_price(request, stock_name):
    price = yf.Ticker(stock_name).history(period='1d')['Close'].iloc[-1]
    return JsonResponse({'price': round(price, 2)})

def get_stock_data(request, stock_name):
    range_option = request.GET.get('range', '1D')
    stock = yf.Ticker(stock_name)

    period_interval_map = {
        "1D": ("1d", "1m"),
        "1W": ("5d", "15m"),
        "1M": ("1mo", "1d"),
        "1Y": ("1y", "1d"),
        "max": ("max", "1d")
    }
    date_format_map = {
        "1D": '%H:%M',
        "1W": '%Y-%m-%d %H:%M',
        "1M": '%Y-%m-%d',
        "1Y": '%Y-%m-%d',
        "max": '%Y-%m-%d'
    }

    period, interval = period_interval_map.get(range_option, ("1d", "1m"))
    date_format = date_format_map.get(range_option, '%Y-%m-%d')

    data = stock.history(period=period, interval=interval)
    if data.empty:
        return JsonResponse({'error': 'No data found'}, status=404)

    timestamps = data.index.strftime(date_format).tolist()
    prices = data['Close'].tolist()

    def select_x_labels(timestamps, prices, num_labels):
        step = max(1, len(timestamps) // num_labels)
        return [timestamps[i] for i in range(0, len(timestamps), step)], \
               [prices[i] for i in range(0, len(prices), step)]

    x_labels, x_prices = select_x_labels(timestamps, prices, 6)
    return JsonResponse({
        'timestamps': timestamps,
        'prices': prices,
        'x_labels': x_labels,
        'x_prices': x_prices
    })

def get_latest_range_price(stock_name, model_name):
    try:
        stock_dir = os.path.join(DATA_DIR, stock_name)
        if not os.path.exists(stock_dir):
            return None

        folders = [f for f in os.listdir(stock_dir) if os.path.isdir(os.path.join(stock_dir, f))]
        if not folders:
            return None

        latest_folder = max(folders)
        file_path = os.path.join(stock_dir, latest_folder, f"{stock_name}_predictions.xlsx")
        if not os.path.exists(file_path):
            return None

        df = pd.read_excel(file_path, usecols=['Model', 'Range'])
        model_data = df[df['Model'] == model_name]
        if not model_data.empty:
            return model_data.iloc[-1]['Range']
    except Exception as e:
        print(f"Error reading prediction file: {e}")
        return None
    return None







def stock_details(request, stock_name):
    try:
        live_price_url = f"http://127.0.0.1:8000/api/live_price/{stock_name}/"
        response = requests.get(live_price_url)
        stock_price = response.json()
        current_price = stock_price.get('price')
        market_status = stock_price.get('market_status', "Unknown")
    except Exception:
        current_price, market_status = None, "Unknown"

    selected_model = request.GET.get('model', 'Ensemble')
    latest_range_price = get_latest_range_price(stock_name, selected_model)


    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({
            'range_price': latest_range_price
        })

    return render(request, 'my_app/stock_details.html', {
        'stock': stock_name,
        'current_price': current_price,
        'market_status': market_status,
        'models': ['LSTM', 'GRU', 'Ensemble'],
        'selected_model': selected_model,
        'latest_range_price': latest_range_price,

    })
