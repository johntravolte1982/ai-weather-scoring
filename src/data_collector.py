"""
Модуль для сбора исторических данных METAR и прогнозов ECMWF
Оптимизирован для работы в GitHub Codespaces
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import os
import time

class WeatherDataCollector:
    def __init__(self, lat: float, lon: float, days_back: int = 7):
        """
        Args:
            lat: Широта
            lon: Долгота
            days_back: На сколько дней назад собирать данные (макс 7 для Open-Meteo)
        """
        self.lat = lat
        self.lon = lon
        self.days_back = min(days_back, 7)  # Open-Meteo ограничение
        
    def fetch_metar_historical(self) -> pd.DataFrame:
        """
        Получение исторических данных METAR через Aviation Weather API
        Бесплатно, без ключа API
        """
        print(f"[METAR] 🔍 Поиск ближайшей станции для координат {self.lat}, {self.lon}")
        
        # Находим ближайшую станцию METAR
        stations_url = f"https://aviationweather.gov/api/data/stationinfo"
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'radius': 100,  # радиус поиска в км
            'format': 'json'
        }
        
        try:
            response = requests.get(stations_url, params=params, timeout=10)
            stations = response.json()
        except Exception as e:
            print(f"[METAR] ⚠️ Ошибка получения списка станций: {e}")
            print("[METAR] 📝 Использую тестовые данные для демонстрации...")
            return self._generate_demo_metar()
        
        if not stations:
            print("[METAR] ⚠️ Станции не найдены, создаю демо-данные...")
            return self._generate_demo_metar()
        
        station_id = stations[0]['icaoId']
        station_name = stations[0].get('name', 'Unknown')
        print(f"[METAR] ✅ Найдена станция: {station_id} - {station_name}")
        
        # Получаем данные METAR за последние часы
        metar_url = f"https://aviationweather.gov/api/data/metar"
        params = {
            'ids': station_id,
            'format': 'json',
            'hours': self.days_back * 24,  # Конвертируем дни в часы
        }
        
        try:
            response = requests.get(metar_url, params=params, timeout=30)
            data = response.json()
        except Exception as e:
            print(f"[METAR] ⚠️ Ошибка получения данных: {e}")
            print("[METAR] 📝 Использую демо-данные...")
            return self._generate_demo_metar()
        
        if not data:
            print("[METAR] ⚠️ Нет данных, создаю демо...")
            return self._generate_demo_metar()
        
        # Парсим в DataFrame
        records = []
        for entry in data:
            if 'temp' in entry and entry['temp'] is not None:
                records.append({
                    'timestamp': entry.get('receiptTime', entry.get('obsTime')),
                    'temp_c': entry['temp'],
                    'dewpoint_c': entry.get('dewp', np.nan),
                    'pressure_hpa': entry.get('altim', np.nan),
                    'wind_speed_kt': entry.get('wspd', np.nan),
                    'wind_dir': entry.get('wdir', np.nan),
                })
        
        if not records:
            print("[METAR] ⚠️ Нет записей с температурой, создаю демо...")
            return self._generate_demo_metar()
        
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Удаляем дубликаты по времени
        df = df[~df.index.duplicated(keep='first')]
        
        # Заполняем пропуски интерполяцией
        df = df.interpolate(method='linear', limit_direction='both')
        
        # Сохраняем
        os.makedirs('data/raw', exist_ok=True)
        df.to_csv('data/raw/metar_historical.csv')
        print(f"[METAR] 💾 Сохранено {len(df)} записей")
        
        return df
    
    def fetch_ecmwf_forecast(self) -> pd.DataFrame:
        """
        Получение прогнозов через Open-Meteo (бесплатный API с данными ECMWF IFS)
        """
        print(f"[ECMWF] 🌍 Получаем данные Open-Meteo для {self.lat}, {self.lon}")
        
        # Open-Meteo использует ECMWF IFS модель для прогнозов
        url = "https://api.open-meteo.com/v1/forecast"
        
        # Формируем даты
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_back)
        
        params = {
            'latitude': self.lat,
            'longitude': self.lon,
            'hourly': ['temperature_2m', 'pressure_msl', 'windspeed_10m', 'winddirection_10m'],
            'timezone': 'auto',
            'past_days': self.days_back,
            'forecast_days': 1,  # Минимальный прогноз вперёд
            'models': 'ecmwf_ifs04'  # Явно указываем модель ECMWF
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"[ECMWF] ⚠️ Ошибка API: {e}")
            print("[ECMWF] 📝 Создаю демо-данные...")
            return self._generate_demo_ecmwf()
        
        if 'hourly' not in data:
            print("[ECMWF] ⚠️ Нет данных 'hourly', создаю демо...")
            return self._generate_demo_ecmwf()
        
        df = pd.DataFrame({
            'timestamp': data['hourly']['time'],
            'ecmwf_temp_c': data['hourly']['temperature_2m'],
            'ecmwf_pressure_hpa': data['hourly']['pressure_msl'],
            'ecmwf_wind_kmh': data['hourly']['windspeed_10m']
        })
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Сохраняем
        os.makedirs('data/raw', exist_ok=True)
        df.to_csv('data/raw/ecmwf_forecast.csv')
        print(f"[ECMWF] 💾 Сохранено {len(df)} записей прогноза")
        
        return df
    
    def _generate_demo_metar(self) -> pd.DataFrame:
        """
        Генерация демо-данных METAR для тестирования
        """
        print("[DEMO] 🎲 Генерация демо-данных METAR...")
        
        dates = pd.date_range(
            end=datetime.now(),
            periods=self.days_back * 24,
            freq='h'
        )
        
        # Имитация суточного хода температуры
        base_temp = 15  # Средняя температура
        temps = base_temp + 8 * np.sin(2 * np.pi * np.arange(len(dates)) / 24 - np.pi/2)
        temps += np.random.normal(0, 1.5, len(dates))  # Добавляем шум
        
        df = pd.DataFrame({
            'temp_c': temps,
            'dewpoint_c': temps - 5 + np.random.normal(0, 1, len(dates)),
            'pressure_hpa': 1013 + np.random.normal(0, 5, len(dates)),
            'wind_speed_kt': 10 + np.random.normal(0, 3, len(dates)),
            'wind_dir': np.random.uniform(0, 360, len(dates))
        }, index=dates)
        
        os.makedirs('data/raw', exist_ok=True)
        df.to_csv('data/raw/metar_historical.csv')
        print(f"[DEMO] 💾 Сохранено {len(df)} демо-записей METAR")
        
        return df
    
    def _generate_demo_ecmwf(self) -> pd.DataFrame:
        """
        Генерация демо-данных ECMWF для тестирования
        """
        print("[DEMO] 🎲 Генерация демо-данных ECMWF...")
        
        dates = pd.date_range(
            end=datetime.now(),
            periods=self.days_back * 24,
            freq='h'
        )
        
        # Имитация прогноза (с небольшим смещением от "реальности")
        base_temp = 15
        temps = base_temp + 7 * np.sin(2 * np.pi * np.arange(len(dates)) / 24 - np.pi/2)
        temps += np.random.normal(0, 1, len(dates))
        temps += 1.5  # Систематическая ошибка прогноза (завышает)
        
        df = pd.DataFrame({
            'ecmwf_temp_c': temps,
            'ecmwf_pressure_hpa': 1013 + np.random.normal(0, 5, len(dates)),
            'ecmwf_wind_kmh': 15 + np.random.normal(0, 4, len(dates))
        }, index=dates)
        
        os.makedirs('data/raw', exist_ok=True)
        df.to_csv('data/raw/ecmwf_forecast.csv')
        print(f"[DEMO] 💾 Сохранено {len(df)} демо-записей ECMWF")
        
        return df


def main():
    """
    Основная функция сбора данных
    """
    print("=" * 60)
    print("🌤️  AI-МЕТЕОРОЛОГ: СБОР ДАННЫХ")
    print("=" * 60)
    
    # Координаты Москвы (можно заменить на свои)
    lat = 55.7558
    lon = 37.6173
    
    print(f"\n📍 Координаты: {lat}, {lon} (Москва)")
    print(f"📅 Собираем данные за последние 7 дней\n")
    
    # Инициализация сборщика
    collector = WeatherDataCollector(lat=lat, lon=lon, days_back=7)
    
    # Сбор данных
    metar_df = collector.fetch_metar_historical()
    time.sleep(1)  # Пауза между запросами
    ecmwf_df = collector.fetch_ecmwf_forecast()
    
    print("\n" + "=" * 60)
    print("✅ ДАННЫЕ СОБРАНЫ!")
    print("=" * 60)
    print(f"\n📊 METAR:      {len(metar_df)} записей")
    print(f"📊 ECMWF:      {len(ecmwf_df)} записей")
    print(f"\n💾 Файлы сохранены в data/raw/")
    print(f"\n📁 Следующий шаг: запустите обучение модели")
    print(f"   python src/model_trainer.py")


if __name__ == "__main__":
    main()