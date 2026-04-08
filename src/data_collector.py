"""
Модуль для сбора исторических данных METAR и прогнозов ECMWF
"""
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
import os

class WeatherDataCollector:
    def __init__(self, lat: float, lon: float, start_date: str, end_date: str):
        self.lat = lat
        self.lon = lon
        self.start_date = start_date
        self.end_date = end_date
        
    def fetch_metar_historical(self) -> pd.DataFrame:
        """
        Получение исторических данных METAR через Aviation Weather API
        Бесплатно, без ключа API
        """
        print(f"[METAR] Собираем данные с {self.start_date} по {self.end_date}")
        
        # Находим ближайшую станцию METAR
        stations_url = f"https://aviationweather.gov/api/data/stationinfo?lat={self.lat}&lon={self.lon}&radius=50&format=json"
        stations = requests.get(stations_url).json()
        
        if not stations:
            raise ValueError("Не найдено METAR станций в радиусе 50 км")
        
        station_id = stations[0]['icaoId']
        print(f"[METAR] Используем станцию: {station_id}")
        
        # Получаем данные
        metar_url = f"https://aviationweather.gov/api/data/metar"
        params = {
            'ids': station_id,
            'format': 'json',
            'hours': 720,  # Максимум 30 дней за раз
        }
        
        response = requests.get(metar_url, params=params)
        data = response.json()
        
        # Парсим в DataFrame
        records = []
        for entry in data:
            if 'temp' in entry and entry['temp'] is not None:
                records.append({
                    'timestamp': entry['receiptTime'],
                    'temp_c': entry['temp'],
                    'dewpoint_c': entry.get('dewp', None),
                    'pressure_hpa': entry.get('altim', None),
                    'wind_speed_kt': entry.get('wspd', None),
                    'wind_dir': entry.get('wdir', None),
                })
        
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Сохраняем
        os.makedirs('data/raw', exist_ok=True)
        df.to_csv('data/raw/metar_historical.csv')
        print(f"[METAR] Сохранено {len(df)} записей")
        
        return df
    
    def fetch_ecmwf_forecast(self) -> pd.DataFrame:
        """
        Получение прогнозов через Open-Meteo (бесплатный API с данными ECMWF)
        """
        print(f"[ECMWF] Получаем прогноз для координат {self.lat}, {self.lon}")
        
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': self.lat,
            'longitude': self.lon,
            'hourly': ['temperature_2m', 'pressure_msl', 'windspeed_10m'],
            'timezone': 'auto',
            'past_days': 7,  # Исторические прогнозы за 7 дней
            'forecast_days': 3
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        df = pd.DataFrame({
            'timestamp': data['hourly']['time'],
            'ecmwf_temp_c': data['hourly']['temperature_2m'],
            'ecmwf_pressure_hpa': data['hourly']['pressure_msl'],
            'ecmwf_wind_kmh': data['hourly']['windspeed_10m']
        })
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        df.to_csv('data/raw/ecmwf_forecast.csv')
        print(f"[ECMWF] Сохранено {len(df)} записей прогноза")
        
        return df

# Тестовый запуск
if __name__ == "__main__":
    # Координаты Москвы для примера
    collector = WeatherDataCollector(
        lat=55.7558,
        lon=37.6173,
        start_date="2024-01-01",
        end_date="2024-01-31"
    )
    
    metar_df = collector.fetch_metar_historical()
    ecmwf_df = collector.fetch_ecmwf_forecast()
    
    print("\n[ГОТОВО] Данные собраны!")
    print(f"METAR: {len(metar_df)} строк")
    print(f"ECMWF: {len(ecmwf_df)} строк")
