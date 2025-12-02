# create_data.py - надежная версия
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import BELARUS_CONFIG, RAW_DATA_DIR
from data_provider import RealisticDataProvider

def create_backup_data():
    """Создание резервных данных если основной метод не работает"""
    provider = RealisticDataProvider()
    
    # 🔥 ИСПРАВЛЕНО: Увеличиваем до 2000 вакансий и 100000 выпускников
    vacancies_df = provider.generate_real_vacancies(2000)
    graduates_df = provider.generate_100k_graduates()  # Новый метод!
    
    # Сохраняем
    vacancies_path = RAW_DATA_DIR / 'real_vacancies.csv'
    graduates_path = RAW_DATA_DIR / 'graduates_data.csv'
    
    vacancies_df.to_csv(vacancies_path, index=False)
    graduates_df.to_csv(graduates_path, index=False)
    
    return vacancies_df, graduates_df

def main():
    """Основная функция генерации данных"""
    print("🎯 ГЕНЕРАЦИЯ 100000 РЕАЛИСТИЧНЫХ ВЫПУСКНИКОВ")
    print("=" * 70)
    
    try:
        print("🔄 Создание реалистичных данных...")
        
        # Используем новый метод для 100000 выпускников
        vacancies_df, graduates_df = create_backup_data()
        
        print(f"✅ Данные успешно созданы:")
        print(f"   - Вакансий: {len(vacancies_df)}")
        print(f"   - Выпускников: {len(graduates_df)}")
        
    except Exception as e:
        print(f"❌ Не удалось создать данные: {e}")
        print("🔄 Пробуем альтернативный метод...")
        
        try:
            provider = RealisticDataProvider()
            # 🔥 ИСПРАВЛЕНО: Альтернативный метод тоже на 100000
            vacancies_df = provider.generate_real_vacancies(2000)
            graduates_df = provider.generate_100k_graduates()
            
            vacancies_path = RAW_DATA_DIR / 'real_vacancies.csv'
            graduates_path = RAW_DATA_DIR / 'graduates_data.csv'
            
            vacancies_df.to_csv(vacancies_path, index=False)
            graduates_df.to_csv(graduates_path, index=False)
            
            print(f"✅ Альтернативные данные сохранены:")
            print(f"   - {graduates_path} ({len(graduates_df)} записей)")
            print(f"   - {vacancies_path} ({len(vacancies_df)} записей)")
            
        except Exception as e2:
            print(f"❌ Критическая ошибка: {e2}")
            return
if __name__ == "__main__":
    main()