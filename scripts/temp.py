import os
import pandas as pd

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


# Test the function for LSTM, GRU, and Ensemble
print(get_latest_range_price("RELIANCE.NS", "LSTM"))
print(get_latest_range_price("RELIANCE.NS", "GRU"))
print(get_latest_range_price("RELIANCE.NS", "Ensemble"))
