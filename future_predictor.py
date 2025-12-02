# future_predictor.py
"""
Модуль для долгосрочного прогнозирования до 2035 года
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FutureTrendsPredictor:
    """Предиктор будущих трендов на рынке труда Беларуси"""
    
    def __init__(self):
        self.current_year = datetime.now().year
        self.industry_trends = self._load_industry_trends()
    
    def _load_industry_trends(self):
        """Загрузка данных о трендах по отраслям - ИСПРАВЛЕННЫЕ КОЭФФИЦИЕНТЫ"""
        return {
            'ИТ': {
                'base_growth': 0.04,  # ЗАМЕДЛЕНИЕ РОСТА
                'disruption_factor': 0.10,
                'global_demand': 0.08,
                'skills_evolution': ['AI/ML', 'Cloud', 'Cybersecurity', 'Quantum', 'Bio-IT']
            },
            'Медицина': {
                'base_growth': 0.07,  # СТАБИЛЬНЫЙ РОСТ
                'disruption_factor': 0.12,
                'global_demand': 0.10,
                'skills_evolution': ['Digital Health', 'Genomics', 'Telemedicine', 'Longevity']
            },
            'Инженерия': {
                'base_growth': 0.05,
                'disruption_factor': 0.08,
                'global_demand': 0.06,
                'skills_evolution': ['Green Tech', 'Smart Infrastructure', 'Space', 'Advanced Materials']
            },
            'Экономика': {
                'base_growth': 0.04,
                'disruption_factor': 0.05,
                'global_demand': 0.03,
                'skills_evolution': ['FinTech', 'Data Analytics', 'ESG', 'Digital Economy']
            },
            'Педагогика': {
                'base_growth': 0.09,  # УСКОРЕННЫЙ РОСТ ИЗ-ЗА ДЕФИЦИТА
                'disruption_factor': 0.08,
                'global_demand': 0.06,
                'skills_evolution': ['EdTech', 'Lifelong Learning', 'Personalized Education', 'Digital Pedagogy']
            },
            'Юриспруденция': {
                'base_growth': 0.045,
                'disruption_factor': 0.06,
                'global_demand': 0.04,
                'skills_evolution': ['LegalTech', 'Digital Law', 'AI Regulation', 'Space Law']
            }
        }
    
    def predict_salary_trajectory(self, faculty, current_salary, target_year):
        """Прогноз траектории зарплаты до target_year - ИСПРАВЛЕННЫЙ РАСЧЕТ"""
        if target_year <= self.current_year:
            return current_salary
        
        trends = self.industry_trends.get(faculty, self.industry_trends['Экономика'])
        years_ahead = target_year - self.current_year
        
        # Базовый рост с учетом насыщения рынка
        if faculty == 'ИТ' and years_ahead > 5:
            # ИТ: быстрый рост первые 5 лет, затем замедление
            early_growth = (1 + trends['base_growth']) ** min(years_ahead, 5)
            late_growth = (1 + trends['base_growth'] * 0.6) ** max(years_ahead - 5, 0)
            base_growth = early_growth * late_growth
        elif faculty == 'Педагогика':
            # Педагогика: ускоренный рост из-за дефицита
            base_growth = (1 + trends['base_growth']) ** years_ahead
        else:
            # Остальные: стабильный рост
            base_growth = (1 + trends['base_growth']) ** years_ahead
        
        # Корректировка на глобальный спрос
        global_adjustment = (1 + trends['global_demand']) ** (years_ahead * 0.5)
        
        # Случайный фактор прорыва
        disruption_boost = 1.0
        if np.random.random() < trends['disruption_factor']:
            disruption_boost = 1.2
        
        predicted_salary = current_salary * base_growth * global_adjustment * disruption_boost
        
        # Ограничения максимального роста
        max_growth_limits = {
            'ИТ': 2.2,      # +120% к 2035
            'Медицина': 2.5, # +150% к 2035  
            'Инженерия': 2.0, # +100% к 2035
            'Экономика': 1.8, # +80% к 2035
            'Педагогика': 2.8, # +180% к 2035 - НАИБОЛЬШИЙ РОСТ
            'Юриспруденция': 1.9 # +90% к 2035
        }
        
        max_salary = current_salary * max_growth_limits.get(faculty, 2.0)
        return min(predicted_salary, max_salary)

# Глобальный экземпляр
future_predictor = FutureTrendsPredictor()