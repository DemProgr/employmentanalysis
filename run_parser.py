# run_parser.py - отдельный запуск парсера
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hh_parser import RealDataEnhancer
from data_loader import RealDataLoader
import pandas as pd

def main():
    print("🚀 ЗАПУСК ПАРСЕРА HH.RU ДЛЯ СБОРА РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 60)
    
    try:
        # Инициализация
        enhancer = RealDataEnhancer()
        loader = RealDataLoader()
        
        # Загружаем существующие данные
        current_vacancies = loader.load_real_vacancies(use_enhancer=False)
        
        print(f"📊 Текущее количество вакансий: {len(current_vacancies) if current_vacancies is not None else 0}")
        
        # Собираем новые данные
        print("🔍 Запуск сбора данных с HH.ru...")
        updated_vacancies = enhancer.enhance_with_real_vacancies(current_vacancies, 200)
        
        print(f"✅ Собрано {len(updated_vacancies)} вакансий")
        
        # Сохраняем
        vacancies_path = Path("data/raw/real_vacancies.csv")
        updated_vacancies.to_csv(vacancies_path, index=False)
        
        # Статистика
        print(f"\n📈 СТАТИСТИКА СОБРАННЫХ ДАННЫХ:")
        print(f"• Всего вакансий: {len(updated_vacancies)}")
        print(f"• Распределение по категориям:")
        category_stats = updated_vacancies['category'].value_counts()
        for category, count in category_stats.items():
            avg_salary = updated_vacancies[updated_vacancies['category'] == category]['salary_avg_byn'].mean()
            print(f"  - {category}: {count} вакансий, средняя зарплата: {avg_salary:.0f} BYN")
        
        print(f"\n💾 Данные сохранены в: {vacancies_path}")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()