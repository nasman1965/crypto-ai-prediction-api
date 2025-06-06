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

# Enhanced ML prediction class
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
            print(f"Error fetching data for {coin_id}: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        if df is None or len(df) < 10:
            return None
            
        # Price changes and momentum
        df['price_change'] = df['price'].pct_change()
        df['price_change_3d'] = df['price'].pct_change(3)
        df['price_change_7d'] = df['price'].pct_change(7)
        
        # Moving averages (multiple timeframes)
        df['sma_3'] = df['price'].rolling(window=3).mean()
        df['sma_7'] = df['price'].rolling(window=7).mean()
        df['sma_14'] = df['price'].rolling(window=14).mean()
        df['sma_21'] = df['price'].rolling(window=21).mean()
        
        # Exponential Moving Averages
        df['ema_7'] = df['price'].ewm(span=7).mean()
        df['ema_14'] = df['price'].ewm(span=14).mean()
        
        # Volatility measures
        df['volatility_3d'] = df['price'].rolling(window=3).std()
        df['volatility_7d'] = df['price'].rolling(window=7).std()
        df['volatility_14d'] = df['price'].rolling(window=14).std()
        
        # Volume indicators
        df['volume_sma_7'] = df['volume'].rolling(window=7).mean()
        df['volume_sma_14'] = df['volume'].rolling(window=14).mean()
        df['volume_change'] = df['volume'].pct_change()
        df['volume_spike'] = df['volume'] / df['volume_sma_7']
        
        # RSI (Relative Strength Index)
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['sma_14']
        bb_std = df['price'].rolling(window=14).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # MACD
        ema_12 = df['price'].ewm(span=12).mean()
        ema_26 = df['price'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df.dropna()
    
    def enhanced_prediction_model(self, df):
        """Enhanced prediction algorithm with multiple factors"""
        if df is None or len(df) < 5:
            return None
            
        latest = df.iloc[-1]
        current_price = latest['price']
        
        # Trend Analysis (40% weight)
        short_trend = latest['ema_7'] / latest['ema_14'] if latest['ema_14'] > 0 else 1
        medium_trend = latest['sma_7'] / latest['sma_14'] if latest['sma_14'] > 0 else 1
        long_trend = latest['sma_14'] / latest['sma_21'] if latest['sma_21'] > 0 else 1
        trend_score = (short_trend + medium_trend + long_trend) / 3
        
        # Momentum Analysis (30% weight)
        rsi_momentum = (50 - latest['rsi']) / 100 if not pd.isna(latest['rsi']) else 0
        macd_momentum = latest['macd_histogram'] / current_price if not pd.isna(latest['macd_histogram']) else 0
        price_momentum = latest['price_change_3d'] if not pd.isna(latest['price_change_3d']) else 0
        momentum_score = (rsi_momentum + macd_momentum + price_momentum) / 3
        
        # Volume Analysis (20% weight)
        volume_signal = latest['volume_spike'] - 1 if not pd.isna(latest['volume_spike']) else 0
        volume_trend = latest['volume_change'] if not pd.isna(latest['volume_change']) else 0
        volume_score = (volume_signal + volume_trend) / 2
        
        # Support/Resistance Analysis (10% weight)
        bb_position = latest['bb_position'] if not pd.isna(latest['bb_position']) else 0.5
        support_resistance_score = (0.5 - bb_position) * 2  # Oversold = positive, Overbought = negative
        
        # Combine all signals with weights
        prediction_factor = (
            0.40 * (trend_score - 1) +
            0.30 * momentum_score +
            0.20 * min(max(volume_score, -0.2), 0.2) +
            0.10 * support_resistance_score
        )
        
        # Cap prediction at ±20%
        predicted_change = max(min(prediction_factor, 0.20), -0.20)
        predicted_price = current_price * (1 + predicted_change)
        
        # Enhanced confidence calculation
        volatility = latest['volatility_7d'] / current_price if current_price > 0 else 1
        data_quality = min(len(df) / 30, 1.0)  # More data = higher confidence
        
        # Base confidence on signal strength and data quality
        signal_strength = abs(prediction_factor)
        base_confidence = min(signal_strength * 2, 0.9)
        volatility_penalty = min(volatility * 5, 0.4)
        
        confidence = max(0.1, min(0.95, base_confidence * data_quality - volatility_penalty))
        
        # Calculate timeframe (24-48 hours from now)
        prediction_time = datetime.now()
        expiry_time = prediction_time + timedelta(hours=36)  # 36 hours average
        
        return {
            'current_price': round(current_price, 6),
            'predicted_price': round(predicted_price, 6),
            'change_percent': round(predicted_change * 100, 2),
            'confidence': round(confidence, 2),
            'trend_signal': round(trend_score, 3),
            'momentum_score': round(momentum_score, 3),
            'volume_score': round(volume_score, 3),
            'rsi': round(latest['rsi'], 1) if not pd.isna(latest['rsi']) else None,
            'macd': round(latest['macd'], 4) if not pd.isna(latest['macd']) else None,
            'bb_position': round(latest['bb_position'], 3) if not pd.isna(latest['bb_position']) else None,
            'timestamp': prediction_time.isoformat(),
            'timeframe': '24-48 hours',
            'prediction_horizon': 'next_day',
            'expires_at': expiry_time.isoformat(),
            'target_date': expiry_time.strftime('%Y-%m-%d %H:%M UTC'),
            'volatility_7d': round(volatility * 100, 2) if not pd.isna(volatility) else None
        }

# Initialize predictor
predictor = CryptoPrediction()

# Routes
@app.route('/')
def home():
    return jsonify({
        "name": "Crypto AI Prediction API",
        "version": "2.0.0",
        "status": "active",
        "description": "Enhanced AI-powered cryptocurrency price predictions with timeframe clarity",
        "endpoints": {
            "/predict/<coin_id>": "Get price prediction for a coin (24-48 hour timeframe)",
            "/trending": "Get trending predictions for top coins",
            "/coins": "List supported cryptocurrencies",
            "/health": "API health check"
        },
        "features": [
            "Technical analysis (RSI, MACD, Bollinger Bands)",
            "Multiple timeframe moving averages",
            "Volume analysis and momentum indicators",
            "Confidence scoring with volatility adjustment",
            "24-48 hour prediction timeframe"
        ],
        "documentation": "https://github.com/nasman1965/crypto-ai-prediction-api",
        "timeframe": "All predictions target 24-48 hours ahead",
        "disclaimer": "For educational purposes only. Not financial advice."
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": "OK",
        "api_version": "2.0.0",
        "prediction_timeframe": "24-48 hours"
    })

@app.route('/predict/<coin_id>')
@rate_limit(max_per_minute=30)
def predict_coin(coin_id):
    """Get enhanced prediction for a specific coin"""
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
        prediction = predictor.enhanced_prediction_model(df_with_indicators)
        if prediction is None:
            return jsonify({"error": "Could not generate prediction"}), 500
        
        return jsonify({
            "coin": coin_id,
            "prediction": prediction,
            "model": "enhanced_trend_v2",
            "data_points": len(df),
            "analysis_features": [
                "Moving averages (3, 7, 14, 21 days)",
                "Exponential moving averages (7, 14 days)",
                "RSI (14-day)",
                "MACD with signal line",
                "Bollinger Bands",
                "Volume analysis",
                "Momentum indicators"
            ],
            "last_updated": datetime.now().isoformat(),
            "disclaimer": "24-48 hour prediction timeframe. Not financial advice."
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/trending')
@rate_limit(max_per_minute=10)
def get_trending():
    """Get predictions for trending coins with enhanced analysis"""
    popular_coins = [
        'bitcoin', 'ethereum', 'solana', 'cardano', 'avalanche-2',
        'dogecoin', 'shiba-inu', 'chainlink', 'polygon', 'uniswap'
    ]
    
    predictions = {}
    
    for coin in popular_coins[:6]:  # Analyze 6 coins
        try:
            df = predictor.get_price_data(coin, days=21)
            if df is not None:
                df_indicators = predictor.calculate_technical_indicators(df)
                if df_indicators is not None:
                    pred = predictor.enhanced_prediction_model(df_indicators)
                    if pred is not None:
                        # Add risk/reward ratio
                        pred['risk_reward_ratio'] = abs(pred['change_percent']) / max(pred['volatility_7d'], 1)
                        predictions[coin] = pred
            
            time.sleep(1.2)  # Rate limiting for free API
            
        except Exception as e:
            print(f"Error processing {coin}: {e}")
            continue
    
    # Sort by a combination of predicted change and confidence
    def score_prediction(item):
        pred = item[1]
        return pred['change_percent'] * pred['confidence'] * 0.01
    
    sorted_predictions = dict(sorted(
        predictions.items(), 
        key=score_prediction, 
        reverse=True
    ))
    
    return jsonify({
        "trending_predictions": sorted_predictions,
        "total_analyzed": len(predictions),
        "ranking_criteria": "Predicted change % weighted by confidence score",
        "timeframe": "24-48 hours",
        "timestamp": datetime.now().isoformat(),
        "next_update": (datetime.now() + timedelta(hours=6)).isoformat(),
        "disclaimer": "Predictions are for educational purposes only. High volatility = high risk."
    })

@app.route('/coins')
def supported_coins():
    """List of supported coins with categories"""
    coins_by_category = {
        "major_coins": ['bitcoin', 'ethereum', 'solana'],
        "altcoins": ['cardano', 'avalanche-2', 'chainlink', 'polygon', 'uniswap'],
        "meme_coins": ['dogecoin', 'shiba-inu'],
        "defi_tokens": ['uniswap', 'chainlink', 'avalanche-2'],
        "layer1_blockchains": ['bitcoin', 'ethereum', 'solana', 'cardano']
    }
    
    all_coins = []
    for category_coins in coins_by_category.values():
        all_coins.extend(category_coins)
    
    unique_coins = list(set(all_coins))
    
    return jsonify({
        "supported_coins": unique_coins,
        "total_count": len(unique_coins),
        "categories": coins_by_category,
        "note": "Use coin IDs from CoinGecko API",
        "prediction_timeframe": "24-48 hours for all coins",
        "data_source": "CoinGecko Free API"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
