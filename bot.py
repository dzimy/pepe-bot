#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BYBIT PEPE/USDT ULTIMATE BOT - PRO TRADER EDITION
Version 7.1 - With Embedded Pro Tips
"""
import os
import time
import numpy as np
import pandas as pd
import ta
import xgboost as xgb
import joblib
from pybit.unified_trading import HTTP
from datetime import datetime
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import pytz

# ===== CONFIGURATION =====
SYMBOL = "PEPEUSDT"

# Pro Tip: Adjust these based on PEPE's current volatility (0.000008-0.000015 range)
RISK_PER_TRADE = 0.02          # 2% of equity per trade
MAX_EXPOSURE = 0.3             # Max 30% account exposure
MIN_PUMP_VOLATILITY = 0.07     # 7% price spike triggers long
MIN_DUMP_VOLATILITY = -0.05    # -5% drop triggers short 
VOLUME_MULTIPLIER = 2.5        # 2.5x average volume required
LIQ_PROFIT_TARGET = 0.04       # 4% TP for liquidation plays
LIQ_STOP_LOSS = 0.02           # 2% SL for liquidation plays

# ===== INITIALIZATION =====
bybit = HTTP(
    api_key=os.getenv('BYBIT_KEY'),
    api_secret=os.getenv('BYBIT_SECRET'),
    testnet=os.getenv('TESTNET', 'True').lower() == 'true'
)

# ===== ENHANCED TELEGRAM ALERTS =====
def send_alert(msg):
    """Pro Tip: Use Telegram for real-time alerts even on mobile"""
    if 'TG_TOKEN' in os.environ and 'TG_CHAT_ID' in os.environ:
        requests.post(
            f"https://api.telegram.org/bot{os.getenv('TG_TOKEN')}/sendMessage",
            json={
                'chat_id': os.getenv('TG_CHAT_ID'),
                'text': msg,
                'parse_mode': 'HTML'
            },
            timeout=5
        )

# ===== LIQUIDATION HUNTING ENGINE =====
class LiquidationHunter:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_liq_zones(self):
        """Pro Tip: Focus on clusters of 5+ liquidations for high-probability trades"""
        liqs = bybit.get_liq_records(category="linear", symbol=SYMBOL, limit=200)['result']['list']
        buy_liqs = [float(x['price']) for x in liqs if x['side'] == 'Buy'][-5:]  # Last 5 buy liqs
        sell_liqs = [float(x['price']) for x in liqs if x['side'] == 'Sell'][-5:] # Last 5 sell liqs
        
        return {
            'support': np.mean(buy_liqs) if buy_liqs else None,
            'resistance': np.mean(sell_liqs) if sell_liqs else None
        }

# ===== AI PREDICTOR =====
class PEPEPredictor:
    def __init__(self):
        """Pro Tip: Retrain model weekly with fresh PEPE data"""
        self.model = joblib.load('models/pepe_xgb_v2.joblib')
        self.scaler = joblib.load('models/pepe_scaler.joblib')
        
    def predict(self, rsi, volume_ratio, price_change):
        features = self.scaler.transform([[rsi, volume_ratio, price_change]])
        return self.model.predict_proba(features)[0][1]  # Probability of success

# ===== CORE TRADING ENGINE =====
class TradingEngine:
    def __init__(self):
        self.hunter = LiquidationHunter()
        self.ai = PEPEPredictor()
        
    def get_market_data(self):
        """Pro Tip: Use 15m candles for optimal PEPE signal quality"""
        data = bybit.get_kline(category="linear", symbol=SYMBOL, interval="15", limit=100)
        return data['result']['list']
    
    def calculate_indicators(self, klines):
        closes = [float(x[4]) for x in klines]
        volumes = [float(x[5]) for x in klines]
        
        # Pro Tip: Use 14-period RSI for PEPE's volatility
        rsi = ta.momentum.RSIIndicator(pd.Series(closes), window=14).rsi()[-1]
        volume_ratio = volumes[-1] / np.mean(volumes[-20:])  # vs 20-period avg
        price_change = (closes[-1] - closes[-5]) / closes[-5]  # 5-period change
        
        return {
            'current_price': closes[-1],
            'rsi': rsi,
            'volume_ratio': volume_ratio, 
            'price_change': price_change
        }

    def execute_trade(self, signal, confidence):
        """Pro Tip: TWAP orders for >10M PEPE to avoid slippage"""
        price = float(bybit.get_tickers(category="linear", symbol=SYMBOL)['result']['list'][0]['lastPrice'])
        equity = float(bybit.get_wallet_balance(accountType="UNIFIED")['result']['list'][0]['totalEquity'])
        
        # Dynamic position sizing
        risk_amount = min(
            equity * RISK_PER_TRADE * (0.5 + confidence/2),  # Scale with confidence
            equity * MAX_EXPOSURE
        )
        qty = int(risk_amount / price)
        
        try:
            if signal == "long":
                tp = price * (1 + LIQ_PROFIT_TARGET)
                sl = price * (1 - LIQ_STOP_LOSS)
                
                # Pro Tip: Immediate execution for small orders
                if qty <= 10000000:  # <10M PEPE
                    bybit.place_order(
                        category="linear",
                        symbol=SYMBOL,
                        side="Buy",
                        orderType="Market",
                        qty=str(qty),
                        takeProfit=str(tp),
                        stopLoss=str(sl),
                        positionIdx=1
                    )
                else:
                    self.twap_execute("Buy", qty, minutes=3)
                
                alert = f"🚀 LIQ LONG {qty:,} PEPE\nEntry: {price:.8f}\nTP: {tp:.8f} | SL: {sl:.8f}"
            
            elif signal == "short":
                tp = price * (1 - LIQ_PROFIT_TARGET)
                sl = price * (1 + LIQ_STOP_LOSS)
                
                if qty <= 10000000:
                    bybit.place_order(
                        category="linear",
                        symbol=SYMBOL,
                        side="Sell",
                        orderType="Market",
                        qty=str(qty),
                        takeProfit=str(tp),
                        stopLoss=str(sl),
                        positionIdx=2
                    )
                else:
                    self.twap_execute("Sell", qty, minutes=3)
                
                alert = f"📉 LIQ SHORT {qty:,} PEPE\nEntry: {price:.8f}\nTP: {tp:.8f} | SL: {sl:.8f}"
            
            send_alert(alert)
            return True
            
        except Exception as e:
            send_alert(f"❌ Trade failed: {str(e)[:200]}")
            return False

    def twap_execute(self, side, total_qty, minutes=3):
        """Time-Weighted Average Price execution"""
        chunks = np.linspace(0, total_qty, num=minutes*2)  # Split into intervals
        for chunk in chunks:
            if chunk > 0:
                bybit.place_order(
                    category="linear",
                    symbol=SYMBOL,
                    side=side,
                    orderType="Limit",
                    qty=str(int(chunk)),
                    price=str(self.get_vwap()),  # VWAP reference price
                    positionIdx=1 if side == "Buy" else 2
                )
            time.sleep(30)  # 30s between chunks

    def get_vwap(self):
        """Volume-Weighted Average Price"""
        data = bybit.get_kline(category="linear", symbol=SYMBOL, interval="1", limit=100)['result']['list']
        closes = np.array([float(x[4]) for x in data])
        volumes = np.array([float(x[5]) for x in data])
        return np.sum(closes * volumes) / np.sum(volumes)

    def run(self):
        """Pro Tip: Best trading times for PEPE - UTC 8-10am and 2-4pm"""
        send_alert(f"🤖 PEPE Pro Bot Activated\n"
                  f"Strategy: Liquidation Hunting + AI\n"
                  f"Volatility Thresholds: PUMP {MIN_PUMP_VOLATILITY*100:.1f}% | DUMP {MIN_DUMP_VOLATILITY*100:.1f}%\n"
                  f"Volume Multiplier: {VOLUME_MULTIPLIER}x")
        
        while True:
            try:
                # 1. Get data
                klines = self.get_market_data()
                indicators = self.calculate_indicators(klines)
                
                # 2. Liquidation signals
                liq_zones = self.hunter.get_liq_zones()
                liq_signal = None
                
                if liq_zones['support'] and indicators['current_price'] <= liq_zones['support'] * 1.03:
                    liq_signal = "long"
                elif liq_zones['resistance'] and indicators['current_price'] >= liq_zones['resistance'] * 0.97:
                    liq_signal = "short"
                
                # 3. AI confirmation
                ai_confidence = self.ai.predict(
                    indicators['rsi'],
                    indicators['volume_ratio'],
                    indicators['price_change']
                )
                
                # 4. Check all conditions
                if (liq_signal and 
                    indicators['volume_ratio'] >= VOLUME_MULTIPLIER and
                    ai_confidence >= 0.65 and
                    ((liq_signal == "long" and indicators['price_change'] >= MIN_PUMP_VOLATILITY) or
                     (liq_signal == "short" and indicators['price_change'] <= MIN_DUMP_VOLATILITY))):
                    
                    self.execute_trade(liq_signal, ai_confidence)
                
                time.sleep(15)  # Check every 15 seconds
                
            except Exception as e:
                send_alert(f"⚠️ System error: {str(e)[:200]}")
                time.sleep(60)

# ===== MAIN =====
if __name__ == "__main__":
    bot = TradingEngine()
    bot.run()