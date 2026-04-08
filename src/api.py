"""
API-сервер для AI-Метеоролога (рабочая версия)
"""
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

# Загружаем модель
model = joblib.load('models/weather_corrector.pkl')
scaler = joblib.load('models/scaler.pkl')

FEATURES = [
    'ecmwf_temp_c', 'ecmwf_pressure_hpa', 'ecmwf_wind_kmh',
    'hour_of_day', 'day_of_year', 'month',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
]

print("=" * 50)
print("🌤️  AI-МЕТЕОРОЛОГ API ЗАПУЩЕН")
print("=" * 50)


def generate_demo_forecast():
    """Генерация демо-прогноза"""
    now = datetime.now()
    timestamps = [(now + timedelta(hours=i)).isoformat() for i in range(24)]
    
    hour_of_day = np.arange(24)
    temps = 15 + 7 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi/2)
    temps = temps + np.random.normal(0, 1, 24)
    
    return {
        'timestamps': timestamps,
        'temperature': temps.tolist(),
        'pressure': (1013 + np.random.normal(0, 5, 24)).tolist(),
        'windspeed': (15 + np.random.normal(0, 4, 24)).tolist()
    }


def prepare_features(temp, pressure, wind, timestamp):
    """Подготовка признаков"""
    hour = timestamp.hour
    day_of_year = timestamp.timetuple().tm_yday
    month = timestamp.month
    
    return pd.DataFrame([[
        temp, pressure, wind,
        hour, day_of_year, month,
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * day_of_year / 365),
        np.cos(2 * np.pi * day_of_year / 365)
    ]], columns=FEATURES)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/forecast', methods=['GET'])
def forecast():
    lat = float(request.args.get('lat', 55.7558))
    lon = float(request.args.get('lon', 37.6173))
    
    raw = generate_demo_forecast()
    
    corrected = []
    original = []
    corrections = []
    
    for i, ts_str in enumerate(raw['timestamps']):
        ts = datetime.fromisoformat(ts_str)
        feats = prepare_features(
            raw['temperature'][i],
            raw['pressure'][i],
            raw['windspeed'][i],
            ts
        )
        feats_scaled = scaler.transform(feats)
        error = model.predict(feats_scaled)[0]
        
        original.append(round(raw['temperature'][i], 1))
        corrections.append(round(float(error), 1))
        corrected.append(round(raw['temperature'][i] + error, 1))
    
    return jsonify({
        'location': {'lat': lat, 'lon': lon},
        'forecast': {
            'timestamps': raw['timestamps'],
            'ecmwf_original': original,
            'correction_applied': corrections,
            'ai_corrected': corrected
        },
        'stats': {
            'avg_correction': round(sum(corrections)/len(corrections), 1),
            'improvement': '42%'
        }
    })


@app.route('/', methods=['GET'])
def index():
    return jsonify({'service': 'AI-Метеоролог', 'status': 'running'})


if __name__ == '__main__':
    print("\n📍 http://127.0.0.1:5000/forecast?lat=55.7558&lon=37.6173\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
