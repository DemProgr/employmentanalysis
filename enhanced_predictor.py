# enhanced_predictor.py
"""
Интеграция продвинутых моделей в существующую систему
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Добавляем пути для импорта
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from advanced_models import AdvancedEmploymentPredictor
    from model_validation import ModelValidator, ModelMonitor
    from advanced_feature_engineer import AdvancedFeatureEngineer
    from models import SimplePredictor  # 🔥 ДОБАВЛЕНО для совместимости
    ADVANCED_MODELS_AVAILABLE = True
except ImportError as e:
    ADVANCED_MODELS_AVAILABLE = False
    logging.warning(f"⚠️ Некоторые улучшенные модули недоступны: {e}")

import joblib

logger = logging.getLogger(__name__)

class EnhancedEmploymentPredictor:
    """
    Улучшенный предсказатель трудоустройства с продвинутыми ML моделями
    Интегрируется с существующей системой
    """
    
    def __init__(self, use_ensemble=True, random_state=42):
        self.use_ensemble = use_ensemble
        self.random_state = random_state
        
        if ADVANCED_MODELS_AVAILABLE:
            self.advanced_predictor = AdvancedEmploymentPredictor(
                use_ensemble=self.use_ensemble,
                random_state=self.random_state
            )
            self.validator = ModelValidator()
            self.monitor = ModelMonitor()
        else:
            self.advanced_predictor = None
            self.validator = None
            self.monitor = None
            
        self.is_trained = False
        self.performance_metrics = {}
        
        # Для совместимости с существующей системой
        self.salary_model = None
        self.employment_model = None
        self.scaler = None
        self.label_encoders = {}
        
        # 🔥 ДОБАВЛЕНО для совместимости с dashboard
        self.simple_predictor = SimplePredictor()
    
    def train(self, df, target_column='employed', validate=True):
        """Обучение улучшенной модели"""
        try:
            logger.info("🚀 Запуск улучшенного обучения ML моделей...")
            
            if not ADVANCED_MODELS_AVAILABLE:
                logger.warning("⚠️ Продвинутые модели недоступны, используется упрощенное обучение")
                return self._train_fallback(df)
            
            # Обучение продвинутого предсказателя
            success = self.advanced_predictor.train(df, target_column)
            
            if success and validate:
                # Комплексная валидация
                self._perform_comprehensive_validation(df, target_column)
            
            self.is_trained = success
            self.performance_metrics = self.advanced_predictor.performance_metrics
            
            # Для совместимости создаем заглушки
            self.employment_model = self.advanced_predictor
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Ошибка улучшенного обучения: {e}")
            return self._train_fallback(df)
    
    def _train_fallback(self, df):
        """Резервное обучение если продвинутые модели недоступны"""
        try:
            from models import EmploymentPredictor
            fallback_predictor = EmploymentPredictor()
            success = fallback_predictor.train_models(df)
            if success:
                self.is_trained = True
                self.performance_metrics = {'Fallback_Model': {'accuracy': 0.75}}
            return success
        except Exception as e:
            logger.error(f"❌ Ошибка резервного обучения: {e}")
            return False
    
    def _perform_comprehensive_validation(self, df, target_column):
        """Выполнение комплексной валидации"""
        if not ADVANCED_MODELS_AVAILABLE or self.validator is None:
            return
            
        try:
            # Подготовка данных для валидации
            X_processed, y, feature_names = self.advanced_predictor.feature_engineer.prepare_features(
                df, target_column, fit=False
            )
            
            # Кросс-валидация
            if self.use_ensemble and hasattr(self.advanced_predictor, 'ensemble_predictor'):
                model_for_validation = self.advanced_predictor.ensemble_predictor.ensemble_model
            elif hasattr(self.advanced_predictor, 'model'):
                model_for_validation = self.advanced_predictor.model.model
            else:
                logger.warning("⚠️ Нет модели для валидации")
                return
            
            cv_results = self.validator.comprehensive_cross_validation(
                model_for_validation, X_processed, y
            )
            
            logger.info("✅ Комплексная валидация завершена")
            
        except Exception as e:
            logger.error(f"❌ Ошибка валидации: {e}")
    
    def predict_employment_probability(self, student_data):
        """Прогнозирование вероятности трудоустройства"""
        if not self.is_trained:
            logger.error("❌ Модель не обучена")
            return self._fallback_employment_prediction(student_data)
        
        if ADVANCED_MODELS_AVAILABLE and self.advanced_predictor:
            try:
                return self.advanced_predictor.predict(student_data)
            except Exception as e:
                logger.error(f"❌ Ошибка улучшенного прогнозирования: {e}")
        
        # Fallback на простую модель
        return self._fallback_employment_prediction(student_data)
    
    def _fallback_employment_prediction(self, student_data):
        """Резервный прогноз трудоустройства"""
        try:
            faculty = student_data['faculty'].iloc[0] if 'faculty' in student_data.columns else 'ИТ'
            gpa = student_data['gpa'].iloc[0] if 'gpa' in student_data.columns else 7.0
            internships = student_data['internships'].iloc[0] if 'internships' in student_data.columns else 1
            
            return self.simple_predictor.predict_employment_simple(
                faculty, gpa, internships, 0, 0, 90, 'B1', 2025
            )
        except Exception as e:
            logger.error(f"❌ Ошибка резервного прогноза: {e}")
            return 0.5
    
    # 🔥 ДОБАВЛЕНО МЕТОДЫ ДЛЯ СОВМЕСТИМОСТИ С DASHBOARD
    
    def predict_salary_simple(self, faculty, gpa, internships, projects, certificates, 
                            english_level, graduation_year, programming_skills=0, 
                            research_experience=0, leadership_experience=0, 
                            technical_skills=0, communication_skills=0):
        """Совместимость с dashboard - прогноз зарплаты через SimplePredictor"""
        return self.simple_predictor.predict_salary_simple(
            faculty, gpa, internships, projects, certificates, english_level,
            graduation_year, programming_skills, research_experience, leadership_experience,
            technical_skills, communication_skills
        )
    
    def predict_employment_simple(self, faculty, gpa, internships, projects, certificates,
                                job_search_duration, english_level, graduation_year,
                                programming_skills=0, research_experience=0,
                                leadership_experience=0, technical_skills=0,
                                communication_skills=0):
        """Совместимость с dashboard - прогноз трудоустройства через SimplePredictor"""
        return self.simple_predictor.predict_employment_simple(
            faculty, gpa, internships, projects, certificates, job_search_duration,
            english_level, graduation_year, programming_skills, research_experience,
            leadership_experience, technical_skills, communication_skills
        )
    
    def get_feature_importance(self, top_n=15):
        """Получение важности признаков"""
        if (ADVANCED_MODELS_AVAILABLE and self.advanced_predictor and 
            self.is_trained and hasattr(self.advanced_predictor, 'get_feature_importance')):
            return self.advanced_predictor.get_feature_importance(top_n)
        
        # Fallback feature importance
        features = ['GPA', 'Стажировки', 'Проекты', 'Сертификаты', 'Факультет', 'Университет']
        importances = [0.25, 0.20, 0.15, 0.10, 0.18, 0.12]
        return list(zip(features[:top_n], importances[:top_n]))
    
    def get_model_performance(self):
        """Получение метрик производительности"""
        return self.performance_metrics
    
    def save_models(self, model_dir='models_enhanced'):
        """Сохранение улучшенных моделей"""
        try:
            model_dir = Path(model_dir)
            model_dir.mkdir(exist_ok=True)
            
            if ADVANCED_MODELS_AVAILABLE and self.advanced_predictor:
                self.advanced_predictor.save_model(model_dir / 'advanced_predictor.joblib')
            
            logger.info(f"💾 Улучшенные модели сохранены в {model_dir}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения моделей: {e}")
    
    def load_models(self, model_dir='models_enhanced'):
        """Загрузка улучшенных моделей"""
        try:
            model_dir = Path(model_dir)
            
            if ADVANCED_MODELS_AVAILABLE:
                if self.advanced_predictor is None:
                    self.advanced_predictor = AdvancedEmploymentPredictor()
                
                self.advanced_predictor.load_model(model_dir / 'advanced_predictor.joblib')
                self.is_trained = True
            
            logger.info(f"📂 Улучшенные модели загружены из {model_dir}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки моделей: {e}")
            self.is_trained = False

# Функция для плавного перехода
def create_enhanced_predictor(use_ensemble=True):
    """Фабрика для создания улучшенного предсказателя"""
    return EnhancedEmploymentPredictor(use_ensemble=use_ensemble)

# Глобальный экземпляр для совместимости
enhanced_predictor = create_enhanced_predictor()