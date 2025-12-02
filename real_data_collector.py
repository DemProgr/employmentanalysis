# real_data_collector.py - упрощенная версия без API
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import RAW_DATA_DIR
from data_provider import RealisticDataProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealDataCollector:
    """Упрощенный сборщик данных без зависимости от API"""
    
    def __init__(self):
        self.data_provider = RealisticDataProvider()
        logger.info("✅ Инициализирован сборщик реалистичных данных")
    
    def collect_real_vacancies(self):
        """Сбор реалистичных вакансий"""
        logger.info("📊 Сбор данных о вакансиях...")
        return self.data_provider.generate_real_vacancies(80)
    
    def generate_realistic_graduates(self, num_graduates=100000):  # 🔥 ИЗМЕНЕНО: 100000 по умолчанию
        """Генерация реалистичных данных о выпускниках"""
        logger.info(f"🎓 Генерация {num_graduates} данных выпускников...")
        return self.data_provider.generate_100k_graduates()
    
    def save_all_data(self):
        """Сохранение всех данных"""
        logger.info("💾 Сохранение данных...")
        
        try:
            # Сбор вакансий
            vacancies_df = self.collect_real_vacancies()
            
            # Генерация данных выпускников
            graduates_df = self.generate_realistic_graduates(400)
            
            # Сохранение
            vacancies_path = RAW_DATA_DIR / 'real_vacancies.csv'
            graduates_path = RAW_DATA_DIR / 'graduates_data.csv'
            
            vacancies_df.to_csv(vacancies_path, index=False, encoding='utf-8')
            graduates_df.to_csv(graduates_path, index=False, encoding='utf-8')
            
            logger.info(f"✅ Данные сохранены:")
            logger.info(f"   - Вакансии: {vacancies_path} ({len(vacancies_df)} записей)")
            logger.info(f"   - Выпускники: {graduates_path} ({len(graduates_df)} записей)")
            
            return vacancies_df, graduates_df
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения данных: {e}")
            # Возвращаем пустые DataFrame в случае ошибки
            return pd.DataFrame(), pd.DataFrame()

def main():
    """Основная функция сбора данных"""
    print("🚀 ЗАПУСК СБОРА РЕАЛИСТИЧНЫХ ДАННЫХ")
    print("=" * 60)
    
    try:
        collector = RealDataCollector()
        vacancies_df, graduates_df = collector.save_all_data()
        
        print(f"\n🎉 СБОР ДАННЫХ ЗАВЕРШЕН!")
        print("=" * 60)
        print(f"📊 Вакансий: {len(vacancies_df)}")
        print(f"🎓 Выпускников: {len(graduates_df)}")
        
        # Статистика
        if not vacancies_df.empty:
            print(f"\n📈 СТАТИСТИКА ВАКАНСИЙ:")
            for category in vacancies_df['category'].unique():
                cat_data = vacancies_df[vacancies_df['category'] == category]
                avg_salary = cat_data['salary_avg_byn'].mean()
                print(f"   {category}: {len(cat_data)} вакансий, средняя зарплата: {avg_salary:.0f} BYN")
        
        if not graduates_df.empty:
            print(f"\n📊 СТАТИСТИКА ВЫПУСКНИКОВ:")
            employment_rate = graduates_df['employed'].mean()
            employed = graduates_df[graduates_df['employed'] == True]
            avg_salary = employed['salary_byn'].mean() if len(employed) > 0 else 0
            print(f"   Уровень трудоустройства: {employment_rate:.1%}")
            print(f"   Средняя зарплата: {avg_salary:.0f} BYN")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()