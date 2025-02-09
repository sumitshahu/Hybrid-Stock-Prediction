from django.shortcuts import render
from django.http import JsonResponse
import yfinance as yf
import subprocess
import json
from datetime import datetime
import pytz
import requests
import numpy as np
import os
import pandas as pd
# List of 100 stocks
STOCKS = [
    'SUNPHARMA.NS', 'DRREDDY.NS', 'ITC.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS',
    'ASIANPAINT.NS', 'BRITANNIA.NS', 'TCS.NS', 'PIDILITIND.NS', 'HDFCBANK.NS',
    'INFY.NS', 'RELIANCE.NS', 'BHARTIARTL.NS', 'SHREECEM.NS', 'BAJAJFINSV.NS',
    # (Add remaining stocks...)
]

def index(request):
    return render(request, 'my_app/index.html')

def search_stocks(request):
    """Search stocks dynamically"""
    query = request.GET.get('query', '').lower()
    matching_stocks = [stock for stock in STOCKS if query in stock.lower()]
    return JsonResponse(matching_stocks, safe=False)

def is_market_open():
    """Check if the stock market is currently open (Indian Market Hours)"""
    now = datetime.now(pytz.timezone("Asia/Kolkata"))
    return now.weekday() < 5 and now.hour >= 9 and now.hour < 15  # 9:00 AM - 3:30 PM IST

def get_live_price(request, stock_name):
    """Fetch live stock price or latest closing price"""
    stock = yf.Ticker(stock_name)
    data = stock.history(period='1d')

    market_status = "Closed"
    price = None

    if not data.empty:
        now = datetime.now(pytz.timezone("Asia/Kolkata"))
        market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)

        if market_open_time <= now <= market_close_time:
            try:
                price = stock.history(period='1m', interval='5m').iloc[-1]['Close']
                market_status = "Open"
            except:
                price = data['Close'].iloc[-1]  # Fallback to last close
        else:
            price = data['Close'].iloc[-1]

    return JsonResponse({'price': round(price, 2) if price else None, 'market_status': market_status})


def get_stock_data(request, stock_name):
    """Fetch stock price history for different time ranges with optimized X-axis intervals"""
    range_option = request.GET.get('range', '1D')
    stock = yf.Ticker(stock_name)

    # Fetch stock history based on range
    if range_option == "1D":
        data = stock.history(period="1d", interval="1m")  # Intraday prices
        date_format = '%H:%M'  # Only show time for intraday
    elif range_option == "1W":
        data = stock.history(period="5d", interval="15m")  # Hourly data over a week
        date_format = '%Y-%m-%d %H:%M'
    elif range_option == "1M":
        data = stock.history(period="1mo", interval="1d")  # Daily closing prices
        date_format = '%Y-%m-%d'
    elif range_option == "1Y":
        data = stock.history(period="1y", interval="1d")  # Daily closing prices
        date_format = '%Y-%m-%d'
    else:
        data = stock.history(period="max", interval="1d")  # Max available data
        date_format = '%Y-%m-%d'

    if data.empty:
        return JsonResponse({'error': 'No data found'}, status=404)

    timestamps = data.index.strftime(date_format).tolist()
    prices = data['Close'].tolist()

    # Select X-axis labels dynamically
    def select_x_labels(timestamps, prices, num_labels):
        """
        Dynamically selects X-axis labels while keeping data aligned.
        Ensures all timestamps are covered without overcrowding.
        """
        num_data_points = len(timestamps)
        
        if num_data_points <= num_labels:  
            return timestamps, prices  # Return all points if few data points exist

        step = max(1, num_data_points // num_labels)  # Avoid step = 0

        return [timestamps[i] for i in range(0, num_data_points, step)], \
               [prices[i] for i in range(0, num_data_points, step)]

    # Define number of X-axis labels dynamically
    num_labels_map = {
        "1D": 6,   # Show 6 spaced-out time points (e.g., every 1-2 hours)
        "1W": 6,   # Show 6 spaced-out days
        "1M": 6,   # Show 6 spaced-out days
        "1Y": 6,   # Show 6 months
        "max": 6   # Show 6 years
    }

    num_labels = num_labels_map.get(range_option, 6)
    x_labels, x_prices = select_x_labels(timestamps, prices, num_labels)

    return JsonResponse({
        'timestamps': timestamps,  # Full timestamps for chart data
        'prices': prices,          # Full price data
        'x_labels': x_labels,      # Optimized X-axis labels
        'x_prices': x_prices       # Corresponding prices for labels
    })

def get_forecast(request, stock_name, model):
    """Fetch predictions from predict.py & get latest range price from Excel"""
    try:
        # Run the prediction script with the selected model
        result = subprocess.run(
            ['python', 'predict.py', stock_name, model],
            capture_output=True, text=True
        )
        prediction = json.loads(result.stdout)

        # Fetch latest Range price from Excel
        latest_range_price = get_latest_range_price(stock_name, model)

        # Return both predicted price & range price
        return JsonResponse({
            'prediction': round(prediction.get('predicted_price', 0), 2),
            'range_price': latest_range_price if latest_range_price is not None else "N/A"
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# Path to stock model directory (update as needed)
DATA_DIR = r"C:\Users\sumit\Final year\stock-market-analysis\models"

def get_latest_range_price(stock_name, model_name):
    """Find the latest date folder, read the Excel file, and return the latest 'Range' price for the selected model."""
    try:
        # Construct the path for the stock's base directory
        stock_dir = os.path.join(DATA_DIR, stock_name)
        print(f"Checking directory: {stock_dir}")

        # Ensure the stock directory exists
        if not os.path.exists(stock_dir):
            print("Error: Stock directory does not exist.")
            return None  # No data available

        # Get the latest folder (sorted by date)
        folders = [f for f in os.listdir(stock_dir) if os.path.isdir(os.path.join(stock_dir, f))]
        print(f"Available date folders: {folders}")

        if not folders:
            print("Error: No date folders found in stock directory.")
            return None

        # Select the latest folder based on the date
        latest_folder = max(folders, key=lambda d: d)  # Latest date folder
        folder_path = os.path.join(stock_dir, latest_folder)
        print(f"Latest folder selected: {latest_folder}")

        # Find the Excel file for the predictions (it will be named <stock_name>_predictions.xlsx)
        excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
        print(f"Excel files found: {excel_files}")

        if not excel_files:
            print("Error: No Excel files found in the latest folder.")
            return None

        # Check if the correct file exists (e.g., <stock_name>_predictions.xlsx)
        latest_file = [f for f in excel_files if f.startswith(stock_name + '_predictions')]
        if not latest_file:
            print("Error: No prediction file found.")
            return None

        latest_file = latest_file[0]  # Use the first match
        file_path = os.path.join(folder_path, latest_file)
        print(f"Latest Excel file selected: {latest_file}")

        # Read the Excel file
        df = pd.read_excel(file_path)
        print("Excel file read successfully.")

        # Ensure 'Range' and 'Model' columns exist
        if 'Range' not in df.columns or 'Model' not in df.columns:
            print(f"Error: 'Range' or 'Model' column not found in {file_path}")
            print(f"Available columns: {df.columns.tolist()}")
            return None

        # Filter the dataframe for the selected model
        model_data = df[df['Model'] == model_name]
        if model_data.empty:
            print(f"Error: No data found for model {model_name}")
            return None

        # Get the Range value for the selected model
        latest_range = model_data.iloc[-1]['Range']  # Fetch the last row for the selected model
        print(f"Latest 'Range' value for model {model_name}: {latest_range}")

        # Return the range price as a string, without any modification
        return latest_range

    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None



from django.http import JsonResponse

def stock_details(request, stock_name):
    """Render stock details page with live price & model-based Range price."""
    live_price_url = f"http://127.0.0.1:8000/api/live_price/{stock_name}/"

    try:
        response = requests.get(live_price_url)
        stock_price = response.json()
    except Exception:
        stock_price = {'price': None, 'market_status': "Unknown"}

    # Get selected model from request (default: Ensemble)
    selected_model = request.GET.get('model', 'Ensemble')

    # Fetch latest Range price from Excel based on the selected model
    latest_range_price = get_latest_range_price(stock_name, selected_model)

    # If the request is an AJAX request, send back the range price
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        if latest_range_price:
            return JsonResponse({'range_price': latest_range_price})
        else:
            return JsonResponse({'error': 'Range price not available'}, status=400)

    return render(request, 'my_app/stock_details.html', {
        'stock': stock_name,
        'current_price': stock_price.get('price'),
        'market_status': stock_price.get('market_status'),
        'models': ['LSTM', 'GRU', 'Ensemble'],
        'selected_model': selected_model,
        'latest_range_price': latest_range_price
    })
