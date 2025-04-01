# BYBIT PEPE BOT - CLOUD EDITION
import os
import time
from pybit.unified_trading import HTTP

# Configuration
SYMBOL = "PEPEUSDT"
bybit = HTTP(
    api_key=os.getenv('BYBIT_KEY'),
    api_secret=os.getenv('BYBIT_SECRET'),
    testnet=False  # Change to True for testing
)

def trade():
    ticker = bybit.get_tickers(category="linear", symbol=SYMBOL)
    price = float(ticker['result']['list'][0]['lastPrice'])
    print(f"Current PEPE Price: {price}")
    
    # Add your strategy here
    if price < 0.00001200:
        print("BUY SIGNAL")
        # bybit.place_order(...)

if __name__ == "__main__":
    while True:
        trade()
        time.sleep(60)