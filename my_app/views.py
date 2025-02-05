from django.shortcuts import render
from django.http import JsonResponse

# List of 100 stocks
STOCKS = [
    'SUNPHARMA.NS', 'DRREDDY.NS', 'ITC.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 
    'ASIANPAINT.NS', 'BRITANNIA.NS', 'TCS.NS', 'PIDILITIND.NS', 'HDFCBANK.NS', 
    'INFY.NS', 'RELIANCE.NS', 'BHARTIARTL.NS', 'SHREECEM.NS', 'BAJAJFINSV.NS', 
    'MARUTI.NS', 'KOTAKBANK.NS', 'HDFCLIFE.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 
    'TATACONSUM.NS', 'SBILIFE.NS', 'GRASIM.NS', 'AXISBANK.NS', 'DIVISLAB.NS', 
    'CIPLA.NS', 'EICHERMOT.NS', 'BAJFINANCE.NS', 'JSWSTEEL.NS', 'WIPRO.NS', 
    'LT.NS', 'SBIN.NS', 'HDFC.NS', 'HEROMOTOCO.NS', 'TECHM.NS', 'ICICIPRULI.NS', 
    'GODREJCP.NS', 'BAJAJ-AUTO.NS', 'TATASTEEL.NS', 'DMART.NS', 'UPL.NS', 
    'POWERGRID.NS', 'NTPC.NS', 'M&M.NS', 'INDUSINDBK.NS', 'HINDALCO.NS', 
    'COALINDIA.NS', 'DLF.NS', 'AMBUJACEM.NS', 'SIEMENS.NS', 'BPCL.NS', 
    'VEDL.NS', 'ONGC.NS', 'ZOMATO.NS', 'GAIL.NS', 'ADANIPORTS.NS', 
    'ADANIGREEN.NS', 'ADANITRANS.NS', 'PGHH.NS', 'APOLLOHOSP.NS', 'DABUR.NS', 
    'PIIND.NS', 'BERGEPAINT.NS', 'HAVELLS.NS', 'BOSCHLTD.NS', 'IDFCFIRSTB.NS', 
    'MARICO.NS', 'LTTS.NS', 'JKCEMENT.NS', 'TVSMOTOR.NS', 'ASTRAL.NS', 
    'AUROPHARMA.NS', 'ESCORTS.NS', 'SRF.NS', 'TATAMOTORS.NS', 'INDIGO.NS', 
    'AUBANK.NS', 'IRCTC.NS', 'MPHASIS.NS', 'NAUKRI.NS', 'JINDALSTEL.NS', 
    'GLAND.NS', 'TATAELXSI.NS', 'PETRONET.NS', 'ACC.NS', 'PERSISTENT.NS', 
    'VBL.NS', 'MOTHERSON.NS', 'TATAPOWER.NS', 'CONCOR.NS', 'PNB.NS', 
    'POLYCAB.NS', 'BIOCON.NS', 'IEX.NS', 'ABB.NS', 'AARTIIND.NS', 
    'NMDC.NS', 'BANDHANBNK.NS', 'ADANIENT.NS', 'MCDOWELL-N.NS'
]

def index(request):
    return render(request, 'my_app/index.html')


def search_stocks(request):
    query = request.GET.get('query', '').lower()
    matching_stocks = [stock for stock in STOCKS if query in stock.lower()]
    return JsonResponse(matching_stocks, safe=False)




def stock_details(request, stock_symbol):
    # Fetch stock data using yfinance
    stock = yf.Ticker(stock_symbol)
    stock_info = stock.info
    historical_data = stock.history(period="max")
    
    # Example stock data (you can customize the data you need)
    data = {
        'symbol': stock_info['symbol'],
        'name': stock_info['shortName'],
        'current_price': stock_info['currentPrice'],
        'high_price': stock_info['dayHigh'],
        'low_price': stock_info['dayLow'],
        'historical_data': historical_data.to_dict(orient='list')
    }

    return render(request, 'my_app/stock_details.html', {'stock': data})

def get_historical_data(request, stock_symbol):
    period = request.GET.get('period', '1mo')
    stock = yf.Ticker(stock_symbol)
    historical_data = stock.history(period=period)
    return JsonResponse(historical_data.to_dict(orient='list'))

