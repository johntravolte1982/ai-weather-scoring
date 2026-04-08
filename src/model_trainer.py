"""
Модуль обучения модели для коррекции прогнозов ECMWF
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class WeatherCorrectorModel:
    """
    Модель для коррекции прогнозов погоды ECMWF на основе исторических данных METAR
    """
    
    def __init__(self, data_path: str = 'data/processed/training_data.csv'):
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'ecmwf_temp_c',
            'ecmwf_pressure_hpa', 
            'ecmwf_wind_kmh',
            'hour_of_day',
            'day_of_year',
            'month'
        ]
        self.target_column = 'temp_error'
        
    def prepare_data(self) -> pd.DataFrame:
        """
        Подготовка данных для обучения:
        - Объединение METAR и ECMWF
        - Создание временных признаков
        - Расчёт целевой переменной (ошибка прогноза)
        """
        print("📊 [1/5] Загрузка и подготовка данных...")
        
        # Загружаем сырые данные
        metar_path = 'data/raw/metar_historical.csv'
        ecmwf_path = 'data/raw/ecmwf_forecast.csv'
        
        if not os.path.exists(metar_path) or not os.path.exists(ecmwf_path):
            raise FileNotFoundError(
                "❌ Не найдены файлы с данными. Сначала запустите:\n"
                "   python src/data_collector.py"
            )
        
        metar = pd.read_csv(metar_path, index_col=0, parse_dates=True)
        ecmwf = pd.read_csv(ecmwf_path, index_col=0, parse_dates=True)
        
        # Объединяем по времени
        df = metar.join(ecmwf, how='inner')
        
        # Удаляем строки с пропусками
        df = df.dropna()
        
        # Создаём целевую переменную (ошибка ECMWF)
        df['temp_error'] = df['temp_c'] - df['ecmwf_temp_c']
        
        # Добавляем временные признаки
        df['hour_of_day'] = df.index.hour
        df['day_of_year'] = df.index.dayofyear
        df['month'] = df.index.month
        
        # Добавляем циклические признаки для времени
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Расширяем список признаков
        self.feature_columns.extend(['hour_sin', 'hour_cos', 'day_sin', 'day_cos'])
        
        # Сохраняем обработанный датасет
        os.makedirs('data/processed', exist_ok=True)
        df.to_csv(self.data_path)
        
        print(f"   ✅ Подготовлено {len(df)} записей")
        print(f"   📅 Период: {df.index.min()} → {df.index.max()}")
        print(f"   🎯 Средняя ошибка ECMWF: {df['temp_error'].mean():.2f}°C")
        
        return df
    
    def train(self, df: pd.DataFrame = None) -> dict:
        """
        Обучение модели GradientBoostingRegressor
        """
        print("\n🧠 [2/5] Обучение модели...")
        
        if df is None:
            df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        
        # Подготовка признаков и целевой переменной
        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()
        
        # Разделение на train/test по времени (80/20)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Масштабирование признаков
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Создаём и обучаем модель
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=3,
            subsample=0.8,
            random_state=42
        )
        
        print(f"   🌲 Обучение GradientBoostingRegressor...")
        print(f"   📚 Обучающая выборка: {len(X_train)} записей")
        print(f"   🧪 Тестовая выборка: {len(X_test)} записей")
        
        self.model.fit(X_train_scaled, y_train)
        
        # Предсказания
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Расчёт метрик
        metrics = {
            'train': {
                'mae': mean_absolute_error(y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'r2': r2_score(y_train, y_pred_train)
            },
            'test': {
                'mae': mean_absolute_error(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'r2': r2_score(y_test, y_pred_test)
            }
        }
        
        # Кросс-валидация на временных рядах
        print("\n📈 [3/5] Кросс-валидация...")
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(
            self.model, 
            self.scaler.transform(X), 
            y, 
            cv=tscv, 
            scoring='neg_mean_absolute_error'
        )
        metrics['cv_mae_mean'] = -cv_scores.mean()
        metrics['cv_mae_std'] = cv_scores.std()
        
        print(f"   📊 Средняя MAE на кросс-валидации: {metrics['cv_mae_mean']:.3f}°C (+/- {metrics['cv_mae_std']:.3f}°C)")
        
        # Важность признаков
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        metrics['feature_importance'] = feature_importance
        
        return metrics
    
    def evaluate(self, metrics: dict) -> None:
        """
        Вывод результатов обучения
        """
        print("\n📊 [4/5] Результаты обучения:")
        print("=" * 50)
        print(f"📈 TRAIN:")
        print(f"   MAE:  {metrics['train']['mae']:.3f}°C")
        print(f"   RMSE: {metrics['train']['rmse']:.3f}°C")
        print(f"   R²:   {metrics['train']['r2']:.3f}")
        print(f"\n🧪 TEST:")
        print(f"   MAE:  {metrics['test']['mae']:.3f}°C")
        print(f"   RMSE: {metrics['test']['rmse']:.3f}°C")
        print(f"   R²:   {metrics['test']['r2']:.3f}")
        print("\n🔝 Топ-5 важных признаков:")
        for _, row in metrics['feature_importance'].head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        # Интерпретация
        improvement = (1 - metrics['test']['rmse'] / abs(metrics['cv_mae_mean'])) * 100
        print(f"\n💡 После коррекции ошибка снижается на ~{improvement:.1f}%")
    
    def save_model(self) -> None:
        """
        Сохранение модели и скейлера
        """
        print("\n💾 [5/5] Сохранение модели...")
        
        os.makedirs('models', exist_ok=True)
        
        # Сохраняем модель
        model_path = 'models/weather_corrector.pkl'
        joblib.dump(self.model, model_path)
        
        # Сохраняем скейлер
        scaler_path = 'models/scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        
        # Сохраняем метаданные
        metadata = {
            'created_at': datetime.now().isoformat(),
            'features': self.feature_columns,
            'target': self.target_column,
            'model_type': 'GradientBoostingRegressor'
        }
        
        import json
        with open('models/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Модель сохранена: {model_path}")
        print(f"   ✅ Скейлер сохранён: {scaler_path}")
        print(f"   ✅ Метаданные: models/metadata.json")
    
    def plot_results(self, df: pd.DataFrame = None) -> None:
        """
        Визуализация результатов
        """
        if df is None:
            df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        
        # Подготовка данных для предсказаний
        X = df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        df['corrected_temp'] = df['ecmwf_temp_c'] + self.model.predict(X_scaled)
        
        # Создаём график
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Сравнение прогнозов
        ax = axes[0, 0]
        sample = df.tail(100)  # Последние 100 точек
        ax.plot(sample.index, sample['temp_c'], label='METAR (факт)', linewidth=2, alpha=0.8)
        ax.plot(sample.index, sample['ecmwf_temp_c'], label='ECMWF (исходный)', alpha=0.7, linestyle='--')
        ax.plot(sample.index, sample['corrected_temp'], label='AI-Метеоролог (скоррект.)', alpha=0.7)
        ax.set_title('Сравнение прогнозов температуры')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Температура (°C)')
        
        # 2. Ошибки прогнозов
        ax = axes[0, 1]
        ecmwf_error = sample['temp_c'] - sample['ecmwf_temp_c']
        corrected_error = sample['temp_c'] - sample['corrected_temp']
        ax.hist(ecmwf_error, bins=20, alpha=0.5, label=f'ECMWF (MAE={abs(ecmwf_error).mean():.2f}°C)')
        ax.hist(corrected_error, bins=20, alpha=0.5, label=f'AI-Метеоролог (MAE={abs(corrected_error).mean():.2f}°C)')
        ax.set_title('Распределение ошибок')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Ошибка (°C)')
        
        # 3. Важность признаков
        ax = axes[1, 0]
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True).tail(10)
        ax.barh(importance['feature'], importance['importance'])
        ax.set_title('Топ-10 важных признаков')
        ax.set_xlabel('Важность')
        
        # 4. Корреляция прогнозов с фактом
        ax = axes[1, 1]
        ax.scatter(df['ecmwf_temp_c'], df['temp_c'], alpha=0.3, label='ECMWF', s=10)
        ax.scatter(df['corrected_temp'], df['temp_c'], alpha=0.3, label='AI-Метеоролог', s=10)
        ax.plot([df['temp_c'].min(), df['temp_c'].max()], 
                [df['temp_c'].min(), df['temp_c'].max()], 
                'k--', alpha=0.5, label='Идеальное совпадение')
        ax.set_xlabel('Прогноз (°C)')
        ax.set_ylabel('Факт (°C)')
        ax.set_title('Корреляция прогнозов с фактической температурой')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохраняем график
        os.makedirs('data/processed', exist_ok=True)
        plt.savefig('data/processed/training_results.png', dpi=150, bbox_inches='tight')
        print(f"\n📈 График сохранён: data/processed/training_results.png")
        
        plt.show()


def main():
    """
    Основной пайплайн обучения
    """
    print("=" * 60)
    print("🌤️  AI-МЕТЕОРОЛОГ: ОБУЧЕНИЕ МОДЕЛИ КОРРЕКЦИИ ПРОГНОЗОВ")
    print("=" * 60)
    
    # Инициализация
    trainer = WeatherCorrectorModel()
    
    # Подготовка данных
    df = trainer.prepare_data()
    
    # Обучение
    metrics = trainer.train(df)
    
    # Оценка
    trainer.evaluate(metrics)
    
    # Сохранение
    trainer.save_model()
    
    # Визуализация
    trainer.plot_results(df)
    
    print("\n" + "=" * 60)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)
    print("\n📁 Следующий шаг: запустите API сервер")
    print("   python src/api.py")


if __name__ == "__main__":
    main()