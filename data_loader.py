# data_loader.py - улучшенная версия с интеграцией парсера
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from data_provider import RealisticDataProvider
from hh_parser import data_enhancer  # Импортируем новый парсер

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealDataLoader:
    def __init__(self):
        self.raw_data_dir = RAW_DATA_DIR
        self.processed_data_dir = PROCESSED_DATA_DIR
        self.data_provider = RealisticDataProvider()
        self.data_enhancer = data_enhancer
    
    def load_real_vacancies(self, use_enhancer=True):
        """Загрузка данных о вакансиях с возможностью дополнения реальными данными"""
        try:
            file_path = self.raw_data_dir / 'real_vacancies.csv'
            vacancies_df = None
            
            if file_path.exists():
                vacancies_df = pd.read_csv(file_path)
                if len(vacancies_df) > 0:
                    logger.info(f"✅ Вакансии загружены: {len(vacancies_df)} записей")
                    
                    # Дополняем реальными данными если нужно
                    if use_enhancer and len(vacancies_df) < 50:  # Если мало данных
                        logger.info("🔄 Дополняем данные реальными вакансиями...")
                        vacancies_df = self.data_enhancer.enhance_with_real_vacancies(vacancies_df)
                        # Сохраняем обновленные данные
                        vacancies_df.to_csv(file_path, index=False)
                    
                    return vacancies_df
            
            # Если файла нет или он пустой, создаем данные
            logger.warning("❌ Файл вакансий не найден или пуст. Создаем данные...")
            vacancies_df = self.data_provider.generate_real_vacancies(80)
            
            # Дополняем реальными данными
            if use_enhancer:
                logger.info("🔄 Дополняем данные реальными вакансиями...")
                vacancies_df = self.data_enhancer.enhance_with_real_vacancies(vacancies_df, 50)
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            vacancies_df.to_csv(file_path, index=False)
            
            logger.info(f"✅ Создано {len(vacancies_df)} вакансий")
            return vacancies_df
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки вакансий: {e}")
            # Возвращаем базовые данные в случае ошибки
            return self.data_provider.generate_real_vacancies(50)
    
    def load_graduates_data(self):
        """Загрузка данных о выпускниках"""
        try:
            file_path = self.raw_data_dir / 'graduates_data.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                if len(df) > 0:
                    logger.info(f"✅ Данные выпускников загружены: {len(df)} записей")
                    return df
            
            # Если файла нет или он пустой, создаем данные
            logger.warning("❌ Файл выпускников не найден или пуст. Создаем данные...")
            df = self.data_provider.generate_real_graduates(400)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(file_path, index=False)
            return df
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки выпускников: {e}")
            return self.data_provider.generate_real_graduates(200)
    
    def update_real_vacancies(self):
        """Принудительное обновление данных реальными вакансиями"""
        try:
            logger.info("🎯 Запуск обновления реальных вакансий...")
            
            # Загружаем существующие данные
            current_vacancies = self.load_real_vacancies(use_enhancer=False)
            
            # Обновляем реальными данными
            updated_vacancies = self.data_enhancer.enhance_with_real_vacancies(current_vacancies, 100)
            
            # Сохраняем
            file_path = self.raw_data_dir / 'real_vacancies.csv'
            updated_vacancies.to_csv(file_path, index=False)
            
            logger.info(f"✅ Данные вакансий обновлены: {len(updated_vacancies)} записей")
            return updated_vacancies
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления вакансий: {e}")
            return self.load_real_vacancies(use_enhancer=False)

# Создание экземпляра для импорта
data_loader = RealDataLoader()