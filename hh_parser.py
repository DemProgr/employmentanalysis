# hh_parser.py - исправленная версия парсера
import requests
import pandas as pd
import numpy as np
import time
import random
import re
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys
import json

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import BELARUS_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HHApiParser:
    """Улучшенный парсер для сбора реальных данных с HeadHunter"""
    
    def __init__(self):
        self.base_url = "https://api.hh.ru"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.8'
        })
        self.fallback_data = []  # Резервные данные на случай ошибок
    
    def search_vacancies(self, search_query="", area=16, per_page=50, pages=2):
        """Поиск вакансий через API HH с улучшенной обработкой ошибок"""
        all_vacancies = []
        
        for page in range(pages):
            try:
                logger.info(f"🔍 Поиск вакансий: '{search_query}', страница {page + 1}")
                
                params = {
                    'text': search_query,
                    'area': area,  # 16 - Беларусь
                    'page': page,
                    'per_page': per_page,
                    'only_with_salary': False,  # Изменили на False чтобы получить больше вакансий
                    'search_field': 'name'  # Ищем только в названиях
                }
                
                response = self.session.get(f"{self.base_url}/vacancies", params=params, timeout=15)
                
                # Проверяем статус ответа
                if response.status_code != 200:
                    logger.warning(f"⚠️ API вернул статус {response.status_code}")
                    continue
                
                data = response.json()
                vacancies = data.get('items', [])
                
                logger.info(f"✅ Найдено {len(vacancies)} вакансий на странице {page + 1}")
                
                for vacancy in vacancies:
                    try:
                        vacancy_data = self.parse_vacancy(vacancy)
                        if vacancy_data and vacancy_data.get('name'):  # Проверяем что есть название
                            all_vacancies.append(vacancy_data)
                    except Exception as e:
                        logger.error(f"Ошибка парсинга вакансии: {e}")
                        continue
                
                # Проверяем лимиты API
                if page >= data.get('pages', 1) - 1:
                    break
                    
                # Задержка для соблюдения лимитов API
                time.sleep(0.3)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Ошибка сети: {e}")
                break
            except Exception as e:
                logger.error(f"❌ Неожиданная ошибка: {e}")
                break
        
        return pd.DataFrame(all_vacancies) if all_vacancies else pd.DataFrame()
    
    def parse_vacancy(self, vacancy):
        """Парсинг данных вакансии с улучшенной обработкой"""
        try:
            # Основная информация
            vacancy_id = vacancy.get('id', f"unknown_{int(time.time())}")
            name = vacancy.get('name', 'Не указано').strip()
            
            if not name or name == 'Не указано':
                return None
                
            company_info = vacancy.get('employer', {})
            company = company_info.get('name', 'Не указана')
            
            # Зарплата
            salary_data = vacancy.get('salary')
            salary_from = salary_data.get('from') if salary_data else None
            salary_to = salary_data.get('to') if salary_data else None
            salary_currency = salary_data.get('currency', 'RUR') if salary_data else 'RUR'
            
            # Конвертируем в BYN
            salary_avg = self.calculate_avg_salary(salary_from, salary_to)
            salary_avg_byn = self.convert_to_byn(salary_avg, salary_currency) if salary_avg else None
            
            # Опыт работы
            experience = vacancy.get('experience', {}).get('name', 'Не указан')
            
            # Навыки
            key_skills = self.extract_skills(vacancy)
            
            # Местоположение
            area_info = vacancy.get('area', {})
            area = area_info.get('name', 'Не указано')
            
            # Дата публикации
            published_at = vacancy.get('published_at', '')
            
            # Описание
            description = self.clean_description(vacancy.get('description', ''))
            
            # Определяем категорию
            category = self.determine_category(name, description)
            
            vacancy_data = {
                'id': f"hh_{vacancy_id}",
                'name': name,
                'company': company,
                'category': category,
                'salary_from': salary_from,
                'salary_to': salary_to,
                'salary_currency': salary_currency,
                'salary_avg': salary_avg,
                'salary_avg_byn': salary_avg_byn,
                'experience': experience,
                'employment': vacancy.get('employment', {}).get('name', 'Не указана'),
                'schedule': vacancy.get('schedule', {}).get('name', 'Не указан'),
                'description': description[:500] if description else '',  # Ограничиваем длину
                'key_skills': key_skills,
                'skills_count': len(key_skills),
                'area': area,
                'published_at': published_at,
                'alternate_url': vacancy.get('alternate_url', ''),
                'archived': vacancy.get('archived', False),
                'source': 'hh_api'
            }
            
            return vacancy_data
            
        except Exception as e:
            logger.error(f"Ошибка парсинга вакансии {vacancy.get('id', '')}: {e}")
            return None
    
    def calculate_avg_salary(self, salary_from, salary_to):
        """Расчет средней зарплаты"""
        try:
            if salary_from and salary_to:
                return (salary_from + salary_to) / 2
            elif salary_from:
                return salary_from
            elif salary_to:
                return salary_to
            return None
        except:
            return None
    
    def convert_to_byn(self, amount, original_currency):
        """Конвертация в BYN (упрощенные курсы)"""
        try:
            if not amount:
                return None
                
            conversion_rates = {
                'RUR': 0.035,  # RUB to BYN
                'RUB': 0.035,  # RUB to BYN
                'USD': 3.2,    # USD to BYN
                'EUR': 3.4,    # EUR to BYN
                'BYR': 1,      # Старые BYN
                'BYN': 1       # Текущие BYN
            }
            
            rate = conversion_rates.get(original_currency, 1)
            converted = amount * rate
            
            # Округляем до 50
            return round(converted / 50) * 50
        except:
            return None
    
    def extract_skills(self, vacancy):
        """Извлечение навыков из вакансии"""
        skills = []
        
        try:
            # Из ключевых навыков
            if 'key_skills' in vacancy:
                skills.extend([skill['name'] for skill in vacancy['key_skills']])
            
            # Из описания (базовый анализ)
            description = (vacancy.get('snippet', {}).get('requirement', '') + 
                          ' ' + vacancy.get('description', '')).lower()
            
            # Популярные навыки для разных категорий
            common_skills = {
                'programming': ['python', 'java', 'javascript', 'c#', 'php', 'ruby', 'go', 'sql'],
                'web': ['html', 'css', 'react', 'vue', 'angular', 'node.js', 'django', 'flask'],
                'devops': ['docker', 'kubernetes', 'aws', 'linux', 'git', 'jenkins'],
                'databases': ['postgresql', 'mongodb', 'mysql', 'redis'],
                'tools': ['git', 'jira', 'confluence', 'figma', 'photoshop']
            }
            
            for category, skill_list in common_skills.items():
                for skill in skill_list:
                    if skill in description and skill not in skills:
                        skills.append(skill.title())
            
            return list(set(skills))[:8]  # Уникальные навыки, максимум 8
            
        except Exception as e:
            logger.error(f"Ошибка извлечения навыков: {e}")
            return []
    
    def clean_description(self, description):
        """Очистка описания от HTML тегов"""
        try:
            if not description:
                return ""
            # Удаляем HTML теги
            clean = re.compile('<.*?>')
            cleaned = re.sub(clean, '', description)
            # Удаляем лишние пробелы
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            return cleaned
        except:
            return ""
    
    def determine_category(self, title, description):
        """Определение категории вакансии с улучшенной логикой"""
        try:
            text = (title + ' ' + description).lower()
            
            category_keywords = {
                'ИТ': [
                    'разработчик', 'программист', 'developer', 'software', 'it', 'айти',
                    'python', 'java', 'javascript', 'c#', 'php', 'ruby', 'go', 'sql',
                    'devops', 'frontend', 'backend', 'fullstack', 'web', 'mobile',
                    'системный администратор', 'сетевой инженер', 'qa', 'тестировщик',
                    'android', 'ios', '1c', 'базы данных', 'админ'
                ],
                'Медицина': [
                    'врач', 'медсестра', 'фельдшер', 'стоматолог', 'хирург', 'терапевт',
                    'педиатр', 'гинеколог', 'кардиолог', 'невролог', 'офтальмолог',
                    'медицинский', 'здравоохранение', 'больница', 'поликлиника'
                ],
                'Инженерия': [
                    'инженер', 'строитель', 'проектировщик', 'технолог', 'конструктор',
                    'энергетик', 'механик', 'электрик', 'прораб', 'архитектор',
                    'техник', 'монтаж', 'наладка', 'оборудование'
                ],
                'Экономика': [
                    'экономист', 'бухгалтер', 'финансовый', 'аналитик', 'маркетолог',
                    'менеджер', 'аудитор', 'кредитный', 'банк', 'финансы',
                    'accountant', 'finance', 'анализ', 'отчетность'
                ],
                'Педагогика': [
                    'учитель', 'преподаватель', 'педагог', 'воспитатель', 'методист',
                    'образование', 'школа', 'университет', 'курс', 'обучение'
                ],
                'Юриспруденция': [
                    'юрист', 'адвокат', 'юрисконсульт', 'нотариус', 'следователь',
                    'правовед', 'закон', 'договор', 'суд'
                ]
            }
            
            # Подсчет совпадений для каждой категории
            category_scores = {}
            for category, keywords in category_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text)
                if score > 0:
                    category_scores[category] = score
            
            if category_scores:
                # Возвращаем категорию с наибольшим количеством совпадений
                return max(category_scores.items(), key=lambda x: x[1])[0]
            else:
                return 'Другое'
                
        except Exception as e:
            logger.error(f"Ошибка определения категории: {e}")
            return 'Другое'

class RealDataEnhancer:
    """Улучшение реальных данных с помощью парсера"""
    
    def __init__(self):
        self.parser = HHApiParser()
    
    def enhance_with_real_vacancies(self, existing_vacancies=None, num_vacancies=50):
        """Дополнение данных реальными вакансиями с резервным планом"""
        logger.info("🎯 Сбор реальных вакансий с HH.ru...")
        
        all_real_vacancies = []
        
        # Основные поисковые запросы
        search_queries = [
            "разработчик", "программист", "it",
            "врач", "медицинский", 
            "инженер", 
            "бухгалтер", "экономист",
            "учитель", "преподаватель",
            "юрист"
        ]
        
        successful_queries = 0
        
        for query in search_queries:
            try:
                logger.info(f"🔍 Поиск по запросу: '{query}'")
                vacancies_df = self.parser.search_vacancies(
                    search_query=query, 
                    per_page=20,  # Уменьшили для тестирования
                    pages=1
                )
                
                if not vacancies_df.empty:
                    all_real_vacancies.append(vacancies_df)
                    successful_queries += 1
                    logger.info(f"✅ Собрано {len(vacancies_df)} вакансий по запросу '{query}'")
                else:
                    logger.warning(f"⚠️ Не найдено вакансий по запросу '{query}'")
                
                # Увеличиваем задержку между запросами
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Ошибка при поиске '{query}': {e}")
                continue
        
        # Объединяем результаты
        if all_real_vacancies:
            real_vacancies_df = pd.concat(all_real_vacancies, ignore_index=True)
            
            # Обработка дубликатов
            if not real_vacancies_df.empty and 'id' in real_vacancies_df.columns:
                real_vacancies_df = real_vacancies_df.drop_duplicates(subset=['id'])
            
            logger.info(f"🎉 Всего собрано {len(real_vacancies_df)} уникальных реальных вакансий")
            
            # Объединение с существующими данными
            if existing_vacancies is not None and not existing_vacancies.empty:
                combined_df = pd.concat([existing_vacancies, real_vacancies_df], ignore_index=True)
                if 'id' in combined_df.columns:
                    combined_df = combined_df.drop_duplicates(subset=['id'])
                return combined_df
            else:
                return real_vacancies_df
        else:
            logger.warning("⚠️ Не удалось собрать реальные вакансии. Используем резервные данные.")
            return self.get_fallback_vacancies(existing_vacancies)
    
    def get_fallback_vacancies(self, existing_vacancies=None):
        """Резервные данные на случай недоступности API"""
        try:
            # Создаем реалистичные данные на основе провайдера
            from data_provider import RealisticDataProvider
            provider = RealisticDataProvider()
            fallback_df = provider.generate_real_vacancies(30)
            
            # Помечаем как реальные данные
            fallback_df['source'] = 'fallback'
            
            if existing_vacancies is not None and not existing_vacancies.empty:
                combined_df = pd.concat([existing_vacancies, fallback_df], ignore_index=True)
                return combined_df
            else:
                return fallback_df
                
        except Exception as e:
            logger.error(f"❌ Ошибка создания резервных данных: {e}")
            return pd.DataFrame()

# Глобальный экземпляр для импорта
data_enhancer = RealDataEnhancer()

def test_api_connection():
    """Тест подключения к API HH"""
    try:
        parser = HHApiParser()
        test_response = parser.session.get("https://api.hh.ru/vacancies?text=test&per_page=1", timeout=10)
        
        if test_response.status_code == 200:
            logger.info("✅ Подключение к API HH успешно")
            return True
        else:
            logger.warning(f"⚠️ API вернул статус {test_response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к API: {e}")
        return False

if __name__ == "__main__":
    # Тестирование парсера
    print("🧪 ТЕСТИРОВАНИЕ ПАРСЕРА HH.RU")
    print("=" * 50)
    
    # Тест подключения
    if test_api_connection():
        enhancer = RealDataEnhancer()
        vacancies = enhancer.enhance_with_real_vacancies(num_vacancies=20)
        
        print(f"✅ Собрано вакансий: {len(vacancies)}")
        if not vacancies.empty:
            print(f"📊 Распределение по категориям:")
            print(vacancies['category'].value_counts())
            print(f"📋 Примеры вакансий:")
            for i, (_, row) in enumerate(vacancies.head(3).iterrows()):
                print(f"{i+1}. {row.get('name', 'N/A')} - {row.get('company', 'N/A')} - {row.get('salary_avg_byn', 'N/A')} BYN")
    else:
        print("❌ Не удалось подключиться к API HH")