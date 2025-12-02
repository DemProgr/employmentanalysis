# data_provider.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import BELARUS_CONFIG, RAW_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealisticDataProvider:
    """Поставщик реалистичных данных на основе реальной статистики Беларуси"""
    
    def __init__(self):
        # 🔥 ИСПРАВЛЕНИЕ: Инициализируем real_stats перед использованием
        self._init_real_stats()
        
        self.real_salary_data = {
            'ИТ': {
                'min': 1800, 'max': 5000, 'avg': 3200, 
                'employment_rate': 0.92,  # Высокий спрос
                'growth_rate': 0.12,      # Быстрый рост
                'premium_universities': ['БГУИР', 'БГУ', 'БНТУ']  # Лучшие для IT
            },
            'Медицина': {
                'min': 1400, 'max': 3500, 'avg': 2200, 
                'employment_rate': 0.95,  # Очень высокий
                'growth_rate': 0.08,
                'premium_universities': ['БГМУ']
            },
            'Инженерия': {
                'min': 1500, 'max': 4000, 'avg': 2500, 
                'employment_rate': 0.88,
                'growth_rate': 0.07,
                'premium_universities': ['БНТУ', 'БГТУ']
            },
            'Экономика': {
                'min': 1200, 'max': 3000, 'avg': 1900, 
                'employment_rate': 0.82,
                'growth_rate': 0.05,
                'premium_universities': ['БГЭУ', 'БГУ']
            },
            'Педагогика': {
                'min': 1000, 'max': 2500, 'avg': 1600, 
                'employment_rate': 0.96,  # Очень высокий из-за дефицита
                'growth_rate': 0.15,      # Ускоренный рост
                'premium_universities': ['БГПУ', 'МГЛУ']
            },
            'Юриспруденция': {
                'min': 1300, 'max': 3500, 'avg': 2100, 
                'employment_rate': 0.78,
                'growth_rate': 0.06,
                'premium_universities': ['БГУ', 'ГрГУ']
            }
        }
        
        # 🔥 ОБНОВЛЕНО: Престиж университетов на основе реальных рейтингов
        self.university_prestige = {
            'БГУ': {'score': 9.5, 'coefficient': 1.25, 'city': 'Минск'},
            'БГУИР': {'score': 9.2, 'coefficient': 1.20, 'city': 'Минск'},
            'БГМУ': {'score': 9.0, 'coefficient': 1.18, 'city': 'Минск'},
            'БНТУ': {'score': 8.8, 'coefficient': 1.15, 'city': 'Минск'},
            'БГЭУ': {'score': 8.5, 'coefficient': 1.12, 'city': 'Минск'},
            'БГПУ': {'score': 8.2, 'coefficient': 1.10, 'city': 'Минск'},
            'ГрГУ': {'score': 7.8, 'coefficient': 1.05, 'city': 'Гродно'},
            'ВГУ': {'score': 7.5, 'coefficient': 1.03, 'city': 'Витебск'},
            'ГГТУ': {'score': 7.3, 'coefficient': 1.02, 'city': 'Гомель'},
            'ПГУ': {'score': 7.0, 'coefficient': 1.00, 'city': 'Могилев'}
        }
        
        # 🔥 ОБНОВЛЕНО: Региональные коэффициенты на основе уровня жизни
        self.real_city_salaries = {
            'Минск': {'coefficient': 1.3, 'base_salary': 2200, 'employment_rate': 0.85},
            'Гродно': {'coefficient': 1.05, 'base_salary': 1800, 'employment_rate': 0.80},
            'Брест': {'coefficient': 1.02, 'base_salary': 1750, 'employment_rate': 0.78},
            'Гомель': {'coefficient': 1.00, 'base_salary': 1700, 'employment_rate': 0.77},
            'Витебск': {'coefficient': 0.98, 'base_salary': 1650, 'employment_rate': 0.76},
            'Могилев': {'coefficient': 0.95, 'base_salary': 1600, 'employment_rate': 0.75}
        }
        
        # 🔥 ДОБАВЛЕНО: Коэффициенты влияния на зарплату
        self.salary_factors = {
            'gpa': 100,          # За каждый балл GPA выше 6.0
            'internships': 150,  # За каждую стажировку
            'projects': 80,      # За каждый проект
            'certificates': 60,  # За каждый сертификат
            'university_prestige': 0.1,  # Процент от базовой зарплаты за балл престижа
            'city_coefficient': 1.0      # Региональный коэффициент
        }

    def generate_future_trends_analysis(self, faculty, current_data):
        """Анализ будущих трендов для конкретной специальности - ИСПРАВЛЕННЫЙ"""
        
        # 🔥 ОБНОВЛЕННЫЕ ПРОГНОЗЫ ДО 2035 ГОДА
        future_trends = {
            'ИТ': {
                '2025': {'salary_growth': 1.04, 'demand_growth': 1.05, 'key_skills': ['AI/ML', 'Cloud', 'Cybersecurity']},
                '2030': {'salary_growth': 1.22, 'demand_growth': 1.28, 'key_skills': ['AI Ethics', 'Quantum Security', 'Bio-IT']},
                '2035': {'salary_growth': 1.48, 'demand_growth': 1.48, 'key_skills': ['Neuro-interfaces', 'Space Tech', 'Sustainable AI']}
            },
            'Медицина': {
                '2025': {'salary_growth': 1.07, 'demand_growth': 1.08, 'key_skills': ['Telemedicine', 'Genomics', 'Precision Medicine']},
                '2030': {'salary_growth': 1.40, 'demand_growth': 1.47, 'key_skills': ['AI Diagnostics', 'Regenerative Medicine', 'Digital Health']},
                '2035': {'salary_growth': 1.93, 'demand_growth': 2.16, 'key_skills': ['Longevity Tech', 'Personalized Vaccines', 'Bio-printing']}
            },
            'Инженерия': {
                '2025': {'salary_growth': 1.05, 'demand_growth': 1.06, 'key_skills': ['Green Tech', 'Robotics', 'Smart Cities']},
                '2030': {'salary_growth': 1.28, 'demand_growth': 1.34, 'key_skills': ['Sustainable Engineering', 'Space Infrastructure', 'Advanced Materials']},
                '2035': {'salary_growth': 1.63, 'demand_growth': 1.79, 'key_skills': ['Quantum Engineering', 'Terraforming', 'Ocean Engineering']}
            },
            'Экономика': {
                '2025': {'salary_growth': 1.04, 'demand_growth': 1.03, 'key_skills': ['FinTech', 'Data Analytics', 'ESG']},
                '2030': {'salary_growth': 1.22, 'demand_growth': 1.16, 'key_skills': ['AI Finance', 'Blockchain', 'Digital Economy']},
                '2035': {'salary_growth': 1.48, 'demand_growth': 1.41, 'key_skills': ['Quantum Finance', 'Space Economics', 'Bio-Economics']}
            },
            'Педагогика': {
                '2025': {'salary_growth': 1.09, 'demand_growth': 1.10, 'key_skills': ['EdTech', 'Digital Pedagogy', 'Inclusive Education']},
                '2030': {'salary_growth': 1.54, 'demand_growth': 1.61, 'key_skills': ['AI Tutoring', 'VR Learning', 'Personalized Education']},
                '2035': {'salary_growth': 2.17, 'demand_growth': 2.59, 'key_skills': ['Neuro-Education', 'Quantum Learning', 'Space Education']}
            },
            'Юриспруденция': {
                '2025': {'salary_growth': 1.045, 'demand_growth': 1.04, 'key_skills': ['LegalTech', 'Digital Law', 'AI Regulation']},
                '2030': {'salary_growth': 1.25, 'demand_growth': 1.22, 'key_skills': ['Blockchain Law', 'AI Ethics Law', 'Space Law']},
                '2035': {'salary_growth': 1.50, 'demand_growth': 1.48, 'key_skills': ['Quantum Law', 'Interplanetary Law', 'Bio-Law']}
            }
        }
        
        return future_trends.get(faculty, {
            '2025': {'salary_growth': 1.04, 'demand_growth': 1.03, 'key_skills': ['Digital Literacy', 'Adaptability']},
            '2030': {'salary_growth': 1.22, 'demand_growth': 1.16, 'key_skills': ['Lifelong Learning', 'Cross-discipline']},
            '2035': {'salary_growth': 1.48, 'demand_growth': 1.41, 'key_skills': ['Future Skills', 'Innovation']}
        })
    
    
    def _init_real_stats(self):
        """Инициализация статистики с обработкой ошибок"""
        try:
            from real_statistics import real_stats_provider
            self.real_stats = real_stats_provider
            logger.info("✅ Real statistics initialized")
        except ImportError as e:
            logger.warning(f"⚠️ Could not import real_statistics: {e}")
            # Создаем базовую структуру как fallback
            self.real_stats = self._create_fallback_stats()
    
    def _create_fallback_stats(self):
        """Создание fallback статистики если модуль не доступен"""
        class FallbackStats:
            def calculate_graduate_distribution(self, target_total=100000):
                return [
                    {'faculty': 'ИТ', 'university': 'БГУИР', 'region': 'Минск', 'count': 18000},
                    {'faculty': 'Медицина', 'university': 'БГМУ', 'region': 'Минск', 'count': 12000},
                    {'faculty': 'Инженерия', 'university': 'БНТУ', 'region': 'Минск', 'count': 22000},
                    {'faculty': 'Экономика', 'university': 'БГЭУ', 'region': 'Минск', 'count': 25000},
                    {'faculty': 'Педагогика', 'university': 'БГПУ', 'region': 'Минск', 'count': 15000},
                    {'faculty': 'Юриспруденция', 'university': 'БГУ', 'region': 'Минск', 'count': 8000}
                ]
        
        class FallbackStatsProvider:
            def __init__(self):
                self.stats = FallbackStats()
        
        return FallbackStatsProvider()

        # В data_provider.py ДОБАВЛЯЕМ в класс RealisticDataProvider:

    def generate_future_trends_analysis(self, faculty, current_data):
        """Анализ будущих трендов для конкретной специальности"""
        
        future_trends = {
            'ИТ': {
                '2025': {'salary_growth': 1.08, 'demand_growth': 1.12, 'key_skills': ['AI/ML', 'Cloud', 'Cybersecurity']},
                '2030': {'salary_growth': 1.47, 'demand_growth': 1.76, 'key_skills': ['Quantum Computing', 'AI Ethics', 'Bioinformatics']},
                '2035': {'salary_growth': 2.00, 'demand_growth': 2.48, 'key_skills': ['Neuro-interfaces', 'Space Tech', 'Sustainable AI']}
            },
            'Медицина': {
                '2025': {'salary_growth': 1.06, 'demand_growth': 1.08, 'key_skills': ['Telemedicine', 'Genomics', 'Precision Medicine']},
                '2030': {'salary_growth': 1.34, 'demand_growth': 1.47, 'key_skills': ['AI Diagnostics', 'Regenerative Medicine', 'Digital Health']},
                '2035': {'salary_growth': 1.79, 'demand_growth': 2.16, 'key_skills': ['Longevity Tech', 'Personalized Vaccines', 'Bio-printing']}
            },
            'Инженерия': {
                '2025': {'salary_growth': 1.05, 'demand_growth': 1.06, 'key_skills': ['Green Tech', 'Robotics', 'Smart Cities']},
                '2030': {'salary_growth': 1.28, 'demand_growth': 1.34, 'key_skills': ['Sustainable Engineering', 'Space Infrastructure', 'Advanced Materials']},
                '2035': {'salary_growth': 1.63, 'demand_growth': 1.79, 'key_skills': ['Quantum Engineering', 'Terraforming', 'Ocean Engineering']}
            }
        }
        
        return future_trends.get(faculty, {
            '2025': {'salary_growth': 1.04, 'demand_growth': 1.03, 'key_skills': ['Digital Literacy', 'Adaptability']},
            '2030': {'salary_growth': 1.22, 'demand_growth': 1.16, 'key_skills': ['Lifelong Learning', 'Cross-discipline']},
            '2035': {'salary_growth': 1.48, 'demand_growth': 1.41, 'key_skills': ['Future Skills', 'Innovation']}
        })
        

    def generate_100k_graduates(self):
        """Генерация 100000 реалистичных выпускников на основе статистики"""
        logger.info("🎓 Генерация 100000 реалистичных выпускников...")
        
        # 🔥 ИСПРАВЛЕНИЕ: Проверяем что real_stats инициализирован
        if not hasattr(self, 'real_stats'):
            self._init_real_stats()
        
        graduates = []
        student_id = 1
        
        # Используем распределение из статистики
        distribution = self.real_stats.calculate_graduate_distribution(100000)
        
        for item in distribution:
            for i in range(item['count']):
                if student_id > 100000:
                    break
                    
                graduate = self._create_realistic_graduate(
                    student_id, item['faculty'], item['university'], item['region']
                )
                graduates.append(graduate)
                student_id += 1
        
        df = pd.DataFrame(graduates)
        
        # Проверяем результат
        if len(df) < 100000:
            logger.warning(f"⚠️ Сгенерировано только {len(df)} выпускников. Догенерируем...")
            # Догенерируем недостающих
            remaining = 100000 - len(df)
            for i in range(remaining):
                graduate = self._create_realistic_graduate(
                    student_id + i, 'ИТ', 'БГУИР', 'Минск'
                )
                graduates.append(graduate)
            
            df = pd.DataFrame(graduates)
        
        logger.info(f"✅ Успешно сгенерировано {len(df)} выпускников")
        return df

    def _create_realistic_graduate(self, student_id, faculty, university, region):
        """Создает одного реалистичного выпускника с учетом престижа университета"""
        # 🔥 ОБНОВЛЕНО: Реалистичное распределение GPA с учетом факультета
        if faculty == 'ИТ':
            gpa = np.random.normal(7.9, 0.7)  # Более высокий средний для IT
        elif faculty == 'Медицина':
            gpa = np.random.normal(8.4, 0.5)  # Самый высокий средний
        elif faculty == 'Инженерия':
            gpa = np.random.normal(7.6, 0.8)
        elif faculty == 'Педагогика':
            gpa = np.random.normal(7.8, 0.6)
        else:
            gpa = np.random.normal(7.5, 0.7)
        
        gpa = max(5.0, min(10.0, gpa))
        
        # 🔥 ОБНОВЛЕНО: Активность на основе GPA и престижа университета
        prestige = self.university_prestige.get(university, {'coefficient': 1.0})['coefficient']
        base_activity = (gpa - 6.0) / 4.0 * prestige
        
        internships = np.random.poisson(1 + base_activity * 3)  # Увеличили влияние
        projects = np.random.poisson(2 + base_activity * 4)
        certificates = np.random.poisson(1 + base_activity * 3)
        
        # 🔥 ОБНОВЛЕНО: Вероятность трудоустройства с учетом престижа университета
        base_employment_rate = self.real_salary_data[faculty]['employment_rate']
        faculty_growth = self.real_salary_data[faculty]['growth_rate']
        
        # Влияние факторов на трудоустройство
        gpa_factor = (gpa - 7.0) * 0.05 if gpa > 7.0 else 0
        experience_factor = internships * 0.08 + projects * 0.05 + certificates * 0.03
        prestige_factor = (prestige - 1.0) * 0.1  # Престиж университета добавляет до +10%
        
        employment_prob = min(0.98, base_employment_rate + gpa_factor + experience_factor + prestige_factor)
        employed = np.random.random() < employment_prob
        
        # 🔥 ОБНОВЛЕНО: Зарплата с учетом престижа университета и региона
        if employed:
            base_salary = self.real_salary_data[faculty]['avg']
            
            # Надбавки за навыки
            salary_boost = (
                (gpa - 6.0) * self.salary_factors['gpa'] +
                internships * self.salary_factors['internships'] +
                projects * self.salary_factors['projects'] +
                certificates * self.salary_factors['certificates']
            )
            
            # Коэффициент престижа университета
            prestige_multiplier = 1.0 + (prestige - 1.0) * 0.15
            
            # Региональный коэффициент
            city_data = self.real_city_salaries.get(region, {'coefficient': 1.0})
            regional_multiplier = city_data['coefficient']
            
            # Итоговая зарплата
            salary = base_salary + salary_boost
            salary = salary * prestige_multiplier * regional_multiplier
            salary *= np.random.uniform(0.95, 1.05)  # Небольшой случайный разброс
            
            # Ограничения по рынку
            salary = max(self.real_salary_data[faculty]['min'], 
                       min(self.real_salary_data[faculty]['max'], salary))
        else:
            salary = 0
        
        # 🔥 ОБНОВЛЕНО: Время поиска работы в зависимости от престижа
        base_search = 90  # Базовое время поиска
        search_reduction = (prestige - 1.0) * 20 + (gpa - 7.0) * 5 + internships * 7
        search_duration = max(15, int(np.random.normal(base_search - search_reduction, 20)))
        
        return {
            'student_id': student_id,
            'university': university,
            'faculty': faculty,
            'specialization': f'{faculty} специализация',
            'gpa': round(gpa, 2),
            'internships': int(internships),
            'projects': int(projects),
            'certificates': int(certificates),
            'graduation_year': np.random.choice([2022, 2023, 2024], p=[0.3, 0.4, 0.3]),
            'employed': employed,
            'salary_byn': round(salary, 2) if employed else 0,
            'job_search_duration': search_duration,
            'field_related': employed and np.random.random() < 0.92,  # Выше вероятность работы по специальности
            'location': region,
            'university_prestige': round(prestige, 2)  # 🔥 НОВЫЙ ПРИЗНАК
        }

    # 🔥 СОХРАНЯЕМ СТАРЫЕ МЕТОДЫ ДЛЯ СОВМЕСТИМОСТИ
    def generate_real_vacancies(self, num_vacancies=100):
        """Генерация реалистичных вакансий"""
        logger.info("🎯 Генерация реалистичных вакансий...")
        
        vacancies = []
        for i in range(num_vacancies):
            category = np.random.choice(list(self.real_salary_data.keys()))
            salary_data = self.real_salary_data[category]
            
            salary_avg = np.random.normal(salary_data['avg'], salary_data['avg'] * 0.2)
            salary_avg = max(salary_data['min'], min(salary_data['max'], salary_avg))
            
            salary_from = salary_avg * np.random.uniform(0.8, 0.95)
            salary_to = salary_avg * np.random.uniform(1.05, 1.2)
            
            skills = np.random.choice(
                self.real_skills_data[category], 
                size=min(5, len(self.real_skills_data[category])),
                replace=False
            ).tolist()
            
            experience_options = ['без опыта', 'от 1 года до 3 лет', 'от 3 до 6 лет', 'более 6 лет']
            experience_weights = [0.2, 0.4, 0.3, 0.1]
            
            vacancy = {
                'id': f'real_vac_{i+1:04d}',
                'name': self._generate_vacancy_name(category),
                'company': np.random.choice(self.companies[category]),
                'category': category,
                'salary_from': round(salary_from),
                'salary_to': round(salary_to),
                'salary_currency': 'BYN',
                'salary_avg': round(salary_avg),
                'salary_avg_byn': round(salary_avg),
                'experience': np.random.choice(experience_options, p=experience_weights),
                'employment': np.random.choice(['полная занятость', 'частичная занятость', 'проектная работа'], 
                                             p=[0.8, 0.15, 0.05]),
                'schedule': np.random.choice(['полный день', 'сменный график', 'гибкий график'], 
                                           p=[0.7, 0.2, 0.1]),
                'description': f'Реальная вакансия в сфере {category}. Требования соответствуют рынку труда Беларуси.',
                'key_skills': skills,
                'skills_count': len(skills),
                'area': np.random.choice(['Минск', 'Гродно', 'Витебск', 'Гомель', 'Могилев', 'Брест'],
                                       p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.05]),
                'published_at': (datetime.now() - timedelta(days=np.random.randint(1, 30))).strftime('%Y-%m-%dT%H:%M:%S%z'),
                'alternate_url': f'https://real-job.by/vacancy/{i+1}',
                'archived': False
            }
            
            vacancies.append(vacancy)
        
        df = pd.DataFrame(vacancies)
        logger.info(f"✅ Сгенерировано {len(df)} реалистичных вакансий")
        return df

    def _generate_vacancy_name(self, category):
        """Генерация реалистичных названий вакансий"""
        names = {
            'ИТ': [
                'Python разработчик', 'Java разработчик', 'Frontend разработчик', 
                'Backend разработчик', 'Fullstack разработчик', 'DevOps инженер',
                'Data Scientist', 'QA инженер', 'Системный администратор', 'Аналитик данных'
            ],
            'Медицина': [
                'Врач-терапевт', 'Врач-хирург', 'Медсестра', 'Фельдшер', 'Стоматолог',
                'Педиатр', 'Кардиолог', 'Невролог', 'Офтальмолог', 'Реаниматолог'
            ],
            'Инженерия': [
                'Инженер-проектировщик', 'Инженер-строитель', 'Инженер-энергетик',
                'Инженер-механик', 'Инженер-технолог', 'Архитектор', 'Геодезист'
            ],
            'Экономика': [
                'Экономист', 'Бухгалтер', 'Финансовый аналитик', 'Аудитор',
                'Менеджер по продажам', 'Маркетолог', 'Бизнес-аналитик'
            ],
            'Педагогика': [
                'Учитель математики', 'Учитель английского', 'Преподаватель вуза',
                'Воспитатель детского сада', 'Методист', 'Педагог-психолог'
            ],
            'Юриспруденция': [
                'Юрист', 'Юрисконсульт', 'Адвокат', 'Нотариус', 'Следователь',
                'Корпоративный юрист', 'Помощник юриста'
            ]
        }
        return np.random.choice(names[category])

    def generate_real_graduates(self, num_graduates=500):
        """Старый метод для совместимости"""
        if num_graduates >= 100000:
            return self.generate_100k_graduates()
        else:
            # Для обратной совместимости
            return self.generate_100k_graduates().head(num_graduates)

# Глобальный экземпляр для импорта
data_provider = RealisticDataProvider()