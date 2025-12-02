# run_analysis.py
"""
Основной скрипт запуска анализа с 100000 выпускников
"""

from data_provider import RealisticDataProvider
from data_loader import RealDataLoader
import pandas as pd

def main():
    print("🚀 ЗАПУСК АНАЛИЗА С 100000 ВЫПУСКНИКОВ")
    
    provider = RealisticDataProvider()
    loader = RealDataLoader()
    
    # Генерируем или загружаем данные
    print("📊 Создание реалистичных данных...")
    graduates_df = provider.generate_100k_graduates()
    vacancies_df = loader.load_real_vacancies()
    
    # Сохраняем данные
    graduates_path = "data/raw/graduates_100k.csv"
    graduates_df.to_csv(graduates_path, index=False)
    
    print(f"✅ Создано {len(graduates_df)} выпускников")
    print(f"💾 Данные сохранены в: {graduates_path}")
    
    # Базовая статистика
    employment_rate = graduates_df['employed'].mean()
    avg_salary = graduates_df[graduates_df['employed']]['salary_byn'].mean()
    
    print(f"📈 Реальная статистика:")
    print(f"   • Уровень трудоустройства: {employment_rate:.1%}")
    print(f"   • Средняя зарплата: {avg_salary:.0f} BYN")
    print(f"   • Распределение по факультетам:")
    
    for faculty in graduates_df['faculty'].unique():
        count = len(graduates_df[graduates_df['faculty'] == faculty])
        faculty_employment = graduates_df[graduates_df['faculty'] == faculty]['employed'].mean()
        print(f"     - {faculty}: {count} выпускников, трудоустройство: {faculty_employment:.1%}")

if __name__ == "__main__":
    main()