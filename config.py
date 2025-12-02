# config.py - конфигурация с API HH
import os
from pathlib import Path

# Базовые настройки
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
IMAGES_DIR = PROJECT_ROOT / 'images'

# API HH (реальные ключи)
HH_CLIENT_ID = "G6SCPUSO46II7OFBHFPEGG27R4SM3OGL812R5EOUKORS7P0UOMDGCTV10MOL7U52"
HH_CLIENT_SECRET = "GC54L99PMTT64D7C6NL0KK78GP29QAC8FV4EDCJJ6HPI1T5UHPB7CM4R59VVJK75"
HH_USER_AGENT = "EmploymentAnalysis/1.0"

# Настройки для Беларуси
BELARUS_CONFIG = {
    'currency': 'BYN',
    'current_year': 2025,
    'universities': [
        'БГУ', 'БГУИР', 'БНТУ', 'БГМУ', 'БГЭУ',
        'БГПУ', 'ГрГУ', 'ВГУ', 'ГГТУ', 'ПГУ'
    ],
    'faculties': {
        'ИТ': ['БГУИР', 'БГУ', 'БНТУ'],
        'Медицина': ['БГМУ', 'ГрГМУ', 'ВГМУ'],
        'Инженерия': ['БНТУ', 'ГрГТУ', 'ВГТУ'],
        'Экономика': ['БГЭУ', 'БГУ', 'ГрГУ'],
        'Педагогика': ['БГПУ', 'МГПУ', 'ВГПУ'],
        'Юриспруденция': ['БГУ', 'ГрГУ', 'ВГУ']
    },
    'cities': ['Минск', 'Гродно', 'Витебск', 'Гомель', 'Могилев', 'Брест'],
    'salary_ranges': {
        'ИТ': (1200, 3500),
        'Медицина': (800, 2500),
        'Инженерия': (1000, 2800),
        'Экономика': (900, 2200),
        'Педагогика': (600, 1500),
        'Юриспруденция': (1100, 3000)
    },
    'economic_indicators': {
        '2022': {'gdp_growth': 1.2, 'unemployment': 4.0, 'inflation': 8.5},
        '2023': {'gdp_growth': 1.5, 'unemployment': 3.8, 'inflation': 7.2},
        '2024': {'gdp_growth': 1.8, 'unemployment': 3.5, 'inflation': 6.5},
        '2025': {'gdp_growth': 2.0, 'unemployment': 3.3, 'inflation': 5.8}
    },
    # config.py (исправляем industry_growth)
    'industry_growth': {
        'ИТ': {'growth_rate': 8.0, 'vacancies_growth': 6.0},       # ЗАМЕДЛЕНИЕ
        'Медицина': {'growth_rate': 12.0, 'vacancies_growth': 10.0}, # СТАБИЛЬНЫЙ РОСТ
        'Инженерия': {'growth_rate': 9.0, 'vacancies_growth': 8.0},  # УМЕРЕННЫЙ РОСТ
        'Экономика': {'growth_rate': 7.0, 'vacancies_growth': 5.0},  # СТАБИЛЬНЫЙ
        'Педагогика': {'growth_rate': 15.0, 'vacancies_growth': 12.0}, # УСКОРЕННЫЙ РОСТ
        'Юриспруденция': {'growth_rate': 8.0, 'vacancies_growth': 6.0}
    }
}

# ML настройки
ML_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'hyperparameter_tuning': {
        'n_trials': 100,
        'timeout': 3600,
        'direction': 'maximize'
    }
}

# Создание директорий
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, IMAGES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print("✅ Конфигурация проекта загружена")