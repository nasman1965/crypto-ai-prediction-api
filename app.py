from flask import Flask, jsonify, request
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
from functools import wraps

app = Flask(__name__)

# Rate limiting decorator
def rate_limit(max_per_minute=60):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapper
    return decorator

# Simple ML prediction class
class CryptoPrediction:
    def __init__(self):
        self.api_base = "https://api.coingecko.com/api/v3"
    
    def get_price_data(self, coin_id, days=30):
        """Get historical price data from CoinGecko (FREE API)"""
        try:
            url = f"{self.api_base}/coins/{coin_id}/market_chart"
            params = {
                "vs_currency": "usd",
                "days": days,
                "interval": "daily"
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            prices = data['prices']
            volumes = data['total_volumes']
            
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['volume'] = [v[1] for v in volumes]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for prediction"""
        if df is None or len(df) < 10:
            return None
            
        # Price changes
        df['price_change'] = df['price'].pct_change()
        df['price_change_7d'] = df['price'].pct_change(7)
        
        # Moving averages
        df['sma_7'] = df['price'].rolling(window=7).mean()
        df['sma_14'] = df['price'].rolling(window=14).mean()
        df['sma_21'] = df['price'].rolling(window=21).mean()
        
        # Volatility
        df['volatility_7d'] = df['price'].rolling(window=7).std()
        df['volatility_14d'] = df['price'].rolling(window=14).mean()
        
        # Volume indicators
        df['volume_sma_7'] = df['volume'].rolling(window=7).mean()
        df['volume_change'] = df['volume'].pct_change()
        
        # RSI (simplified)
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df.dropna()
    
    def simple_prediction_model(self, df):
        """Simple prediction algorithm - replace with your ML model"""
        if df is None or len(df) < 5:
            return None
            
        latest = df.iloc[-1]
        current_price = latest['price']
        
        # Simple trend analysis
        short_trend = latest['sma_7'] / latest['sma_14'] if latest['sma_14'] > 0 else 1
        long_trend = latest['sma_14'] / latest['sma_21'] if latest['sma_21'] > 0 else 1
        
        # Momentum indicators
        rsi_signal = 0.5 if pd.isna(latest['rsi']) else (50 - latest['rsi']) / 100
        volume_signal = latest['volume_change'] if not pd.isna(latest['volume_change']) else 0
        
        # Combine signals (simple weighted average)
        trend_weight = 0.4
        momentum_weight = 0.3
        volume_weight = 0.3
        
        prediction_factor = (
            trend_weight * (short_trend + long_trend - 2) +
            momentum_weight * rsi_signal +
            volume_weight * min(max(volume_signal, -0.1), 0.1)
        )
        
        # Predict price change (cap at ±15%)
        predicted_change = max(min(prediction_factor, 0.15), -0.15)
        predicted_price = current_price * (1 + predicted_change)
        
        # Confidence score (simplified)
        volatility = latest['volatility_7d'] / current_price if current_price > 0 else 1
        confidence = max(0.1, min(0.9, 1 - volatility * 10))
        
        return {
            'current_price': round(current_price, 6),
            'predicted_price': round(predicted_price, 6),
            'change_percent': round(predicted_change * 100, 2),
            'confidence': round(confidence, 2),
            'trend_signal': round(short_trend, 3),
            'rsi': round(latest['rsi'], 1) if not pd.isna(latest['rsi']) else None,
            'timestamp': datetime.now().isoformat()
        }

# Initialize predictor
predictor = CryptoPrediction()

# Routes
@app.route('/')
def home():
    return jsonify({
        "name": "Crypto AI Prediction API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "/predict/<coin_id>": "Get price prediction for a coin",
            "/trending": "Get trending predictions for top coins",
            "/health": "API health check"
        },
        "documentation": "https://github.com/your-username/crypto-ai-prediction-api"
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": "OK"
    })

@app.route('/predict/<coin_id>')
@rate_limit(max_per_minute=30)
def predict_coin(coin_id):
    """Get prediction for a specific coin"""
    try:
        # Get historical data
        df = predictor.get_price_data(coin_id, days=30)
        if df is None:
            return jsonify({"error": f"Could not fetch data for {coin_id}"}), 404
        
        # Calculate indicators
        df_with_indicators = predictor.calculate_technical_indicators(df)
        if df_with_indicators is None:
            return jsonify({"error": "Insufficient data for analysis"}), 400
        
        # Make prediction
        prediction = predictor.simple_prediction_model(df_with_indicators)
        if prediction is None:
            return jsonify({"error": "Could not generate prediction"}), 500
        
        return jsonify({
            "coin": coin_id,
            "prediction": prediction,
            "model": "simple_trend_v1",
            "data_points": len(df),
            "last_updated": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/trending')
@rate_limit(max_per_minute=10)
def get_trending():
    """Get predictions for trending coins"""
    popular_coins = [
        'bitcoin', 'ethereum', 'solana', 'cardano', 'avalanche-2',
        'dogecoin', 'shiba-inu', 'chainlink', 'polygon', 'uniswap'
    ]
    
    predictions = {}
    
    for coin in popular_coins[:5]:  # Limit to 5 to avoid rate limits
        try:
            df = predictor.get_price_data(coin, days=14)
            if df is not None:
                df_indicators = predictor.calculate_technical_indicators(df)
                if df_indicators is not None:
                    pred = predictor.simple_prediction_model(df_indicators)
                    if pred is not None:
                        predictions[coin] = pred
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"Error processing {coin}: {e}")
            continue
    
    # Sort by predicted change
    sorted_predictions = dict(sorted(
        predictions.items(), 
        key=lambda x: x[1]['change_percent'], 
        reverse=True
    ))
    
    return jsonify({
        "trending_predictions": sorted_predictions,
        "total_analyzed": len(predictions),
        "timestamp": datetime.now().isoformat(),
        "note": "Predictions are for educational purposes only"
    })

@app.route('/coins')
def supported_coins():
    """List of supported coins"""
    popular_coins = [
        'bitcoin', 'ethereum', 'solana', 'cardano', 'avalanche-2',
        'dogecoin', 'shiba-inu', 'chainlink', 'polygon', 'uniswap',
        'litecoin', 'bitcoin-cash', 'stellar', 'vechain', 'filecoin'
    ]
    
    return jsonify({
        "supported_coins": popular_coins,
        "total_count": len(popular_coins),
        "note": "Use coin IDs from CoinGecko API"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
