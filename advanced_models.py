"""
Продвинутые ML модели для прогнозирования трудоустройства с оптимизацией гиперпараметров
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                   StratifiedKFold, RandomizedSearchCV,
                                   cross_validate)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                            HistGradientBoostingClassifier, VotingClassifier,
                            StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report,
                           confusion_matrix, precision_recall_curve, average_precision_score)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedEmploymentClassifier(BaseEstimator, ClassifierMixin):
    """Продвинутый классификатор для прогнозирования трудоустройства"""
    
    def __init__(self, model_type='xgboost', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.base_model = None  # 🔥 ДОБАВЛЕНО: храним базовую модель отдельно
        self.is_calibrated = False
        self.feature_importance_ = None
        self.classes_ = None
        
    def _get_base_model(self, model_type):
        """Получение базовой модели по типу"""
        models = {
            'xgboost': XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
            ),
            'lightgbm': LGBMClassifier(
                random_state=self.random_state,
                verbose=-1
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_state
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=self.random_state
            ),
            'logistic': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            )
        }
        return models.get(model_type, models['xgboost'])
    
    def _get_param_distribution(self, model_type):
        """Параметры для RandomizedSearchCV"""
        param_distributions = {
            'xgboost': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1],
                'reg_lambda': [0, 0.1, 0.5, 1]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 9, -1],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [31, 63, 127, 255],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1],
                'reg_lambda': [0, 0.1, 0.5, 1]
            },
            'random_forest': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        return param_distributions.get(model_type, {})
    
    def fit(self, X, y, optimize_hyperparams=True, cv_folds=5):
        """Обучение модели с оптимизацией гиперпараметров"""
        try:
            logger.info(f"🎯 Обучение модели {self.model_type}...")
            
            base_model = self._get_base_model(self.model_type)
            self.classes_ = np.unique(y)
            
            if optimize_hyperparams and self._get_param_distribution(self.model_type):
                # Оптимизация гиперпараметров
                param_dist = self._get_param_distribution(self.model_type)
                
                search = RandomizedSearchCV(
                    base_model,
                    param_dist,
                    n_iter=20,  # Уменьшено для скорости
                    cv=StratifiedKFold(n_splits=min(cv_folds, 3), shuffle=True, random_state=self.random_state),
                    scoring='roc_auc',
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=0
                )
                
                search.fit(X, y)
                self.base_model = search.best_estimator_  # 🔥 ИСПРАВЛЕНО: сохраняем базовую модель
                logger.info(f"✅ Лучшие параметры: {search.best_params_}")
                logger.info(f"✅ Лучший ROC-AUC: {search.best_score_:.4f}")
                
            else:
                # Простое обучение
                self.base_model = base_model
                self.base_model.fit(X, y)
            
            # 🔥 ИСПРАВЛЕНО: Сохранение важности признаков ДО калибровки
            if hasattr(self.base_model, 'feature_importances_'):
                self.feature_importance_ = self.base_model.feature_importances_
            elif hasattr(self.base_model, 'coef_'):
                self.feature_importance_ = np.abs(self.base_model.coef_[0])
            
            # Калибровка вероятностей
            self.model = CalibratedClassifierCV(
                self.base_model,  # 🔥 ИСПРАВЛЕНО: используем базовую модель
                cv=min(3, cv_folds),
                method='isotonic'
            )
            self.model.fit(X, y)
            self.is_calibrated = True
            
            logger.info(f"✅ Модель {self.model_type} успешно обучена и откалибрована")
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения модели: {e}")
            raise
    
    def predict(self, X):
        """Предсказание классов"""
        if self.model is None:
            raise ValueError("Модель не обучена")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Предсказание вероятностей"""
        if self.model is None:
            raise ValueError("Модель не обучена")
        return self.model.predict_proba(X)
    
    def predict_employment_probability(self, X):
        """Предсказание вероятности трудоустройства"""
        probabilities = self.predict_proba(X)
        return probabilities[:, 1]  # Вероятность класса 1 (трудоустроен)

class EnsembleEmploymentPredictor:
    """Ансамблевый предсказатель для повышения надежности"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.ensemble_model = None
        self.feature_names = []
        self.is_trained = False
        
    def create_ensemble(self, X, y, feature_names):
        """Создание ансамбля моделей"""
        self.feature_names = feature_names
        
        # Базовые модели
        base_models = [
            ('xgboost', AdvancedEmploymentClassifier('xgboost', self.random_state)),
            ('lightgbm', AdvancedEmploymentClassifier('lightgbm', self.random_state)),
            ('random_forest', AdvancedEmploymentClassifier('random_forest', self.random_state))
        ]
        
        # Обучаем базовые модели
        for name, model in base_models:
            logger.info(f"🔧 Обучение {name}...")
            model.fit(X, y, optimize_hyperparams=True, cv_folds=3)
            self.models[name] = model
        
        # 🔥 ИСПРАВЛЕНО: используем базовые модели для стекинга
        meta_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        
        # Создание стекинга с базовыми моделями
        self.ensemble_model = StackingClassifier(
            estimators=[(name, model.base_model) for name, model in base_models],  # 🔥 ИСПРАВЛЕНО: используем base_model
            final_estimator=meta_model,
            cv=3,
            passthrough=False,
            n_jobs=-1
        )
        
        # Обучение ансамбля
        logger.info("🏗️ Обучение ансамблевой модели...")
        self.ensemble_model.fit(X, y)
        
        self.is_trained = True
        logger.info("✅ Ансамблевая модель успешно обучена")
    
    def predict_proba(self, X):
        """Предсказание вероятностей ансамблем"""
        if not self.is_trained:
            raise ValueError("Ансамбль не обучен")
        return self.ensemble_model.predict_proba(X)
    
    def predict_employment_probability(self, X):
        """Предсказание вероятности трудоустройства"""
        probabilities = self.predict_proba(X)
        return probabilities[:, 1]
    
    def get_model_weights(self):
        """Получение весов моделей в ансамбле"""
        if hasattr(self.ensemble_model.final_estimator_, 'coef_'):
            return self.ensemble_model.final_estimator_.coef_[0]
        return None

class AdvancedEmploymentPredictor:
    """Продвинутый предсказатель трудоустройства"""
    
    def __init__(self, use_ensemble=True, random_state=42):
        self.use_ensemble = use_ensemble
        self.random_state = random_state
        self.feature_engineer = None
        self.model = None
        self.ensemble_predictor = None
        self.performance_metrics = {}
        self.is_trained = False
        
    def train(self, df, target_column='employed', test_size=0.2):
        """Обучение продвинутой модели с обработкой ошибок"""
        try:
            # Импорт здесь чтобы избежать циклических импортов
            from advanced_feature_engineer import AdvancedFeatureEngineer
            
            # Инициализация feature engineer
            self.feature_engineer = AdvancedFeatureEngineer()
            
            # 🔥 ПРОВЕРКА ДАННЫХ ПЕРЕД ОБУЧЕНИЕМ
            required_columns = ['gpa', 'internships', 'projects', 'certificates', 
                            'graduation_year', 'salary_byn', 'job_search_duration',
                            'faculty', 'university', 'location', 'employed']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"⚠️ В данных отсутствуют колонки: {missing_columns}")
                # Попробуем создать недостающие колонки с базовыми значениями
                for col in missing_columns:
                    if col == 'salary_byn':
                        df[col] = df.get('salary', 0)
                    elif col == 'job_search_duration':
                        df[col] = 90  # Среднее значение
                    elif col == 'location':
                        df[col] = 'Минск'  # Значение по умолчанию
                    else:
                        df[col] = 0
            
            # Подготовка данных
            X_processed, y, feature_names = self.feature_engineer.prepare_features(
                df, target_column, fit=True
            )
            
            if len(X_processed) < 50:
                logger.warning("⚠️ Мало данных для продвинутого обучения")
                return False
            
            # Разделение на train/test
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, 
                test_size=test_size, 
                random_state=self.random_state, 
                stratify=y
            )
            
            # 🔥 ЗАПИСЫВАЕМ ИНФОРМАЦИЮ О ПРИЗНАКАХ
            logger.info(f"📊 Используется {len(feature_names)} признаков для обучения")
            logger.info(f"📋 Признаки: {feature_names[:10]}...")  # Показываем первые 10
            
            if self.use_ensemble:
                # Используем ансамбль
                self.ensemble_predictor = EnsembleEmploymentPredictor(self.random_state)
                self.ensemble_predictor.create_ensemble(X_train, y_train, feature_names)
                
                # Оценка на тестовых данных
                y_pred_proba = self.ensemble_predictor.predict_employment_probability(X_test)
                self._evaluate_model(y_test, y_pred_proba, "Ensemble")
                
            else:
                # Используем лучшую одиночную модель
                best_model = AdvancedEmploymentClassifier('xgboost', self.random_state)
                best_model.fit(X_train, y_train, optimize_hyperparams=True, cv_folds=3)
                self.model = best_model
                
                # Оценка на тестовых данных
                y_pred_proba = self.model.predict_employment_probability(X_test)
                self._evaluate_model(y_test, y_pred_proba, "XGBoost")
            
            self.is_trained = True
            logger.info("🎉 Продвинутое обучение завершено успешно")
            
            # 🔥 СОХРАНЯЕМ ИНФОРМАЦИЮ О ПРИЗНАКАХ
            self.feature_names = feature_names
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка продвинутого обучения: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _evaluate_model(self, y_true, y_pred_proba, model_name):
        """Оценка модели"""
        try:
            # Вычисляем метрики
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            metrics = {
                'roc_auc': roc_auc_score(y_true, y_pred_proba),
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0)
            }
            
            self.performance_metrics[model_name] = metrics
            
            logger.info(f"📊 Результаты {model_name}:")
            for metric, value in metrics.items():
                logger.info(f"   {metric}: {value:.4f}")
                
        except Exception as e:
            logger.error(f"❌ Ошибка оценки модели: {e}")
    
    def predict(self, student_data):
        """Прогнозирование вероятности трудоустройства"""
        if not self.is_trained:
            logger.error("❌ Модель не обучена")
            return None
        
        try:
            # Подготовка данных
            X_processed, _, _ = self.feature_engineer.prepare_features(
                student_data, fit=False
            )
            
            # Предсказание
            if self.use_ensemble and self.ensemble_predictor:
                probability = self.ensemble_predictor.predict_employment_probability(X_processed)
            elif self.model:
                probability = self.model.predict_employment_probability(X_processed)
            else:
                logger.error("❌ Нет обученной модели")
                return None
            
            return probability[0] if len(probability) == 1 else probability
            
        except Exception as e:
            logger.error(f"❌ Ошибка прогнозирования: {e}")
            return None
    
    def get_feature_importance(self, top_n=15):
        """Получение важности признаков"""
        if not self.is_trained:
            return None
        
        try:
            feature_names = self.feature_engineer.get_feature_names()
            
            if self.model and hasattr(self.model, 'feature_importance_'):
                importances = self.model.feature_importance_
            elif self.ensemble_predictor:
                # Для ансамбля используем среднюю важность из базовых моделей
                importances = np.zeros(len(feature_names))
                count = 0
                for name, model in self.ensemble_predictor.models.items():
                    if hasattr(model, 'feature_importance_') and model.feature_importance_ is not None:
                        importances += model.feature_importance_
                        count += 1
                if count > 0:
                    importances /= count
            else:
                return None
            
            # Сортируем по важности
            indices = np.argsort(importances)[::-1]
            
            # Берем только top_n признаков
            top_features = []
            for i in indices[:min(top_n, len(importances))]:
                if i < len(feature_names):
                    top_features.append((feature_names[i], importances[i]))
            
            return top_features
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения важности признаков: {e}")
            return None
    
    def save_model(self, filepath):
        """Сохранение модели"""
        try:
            model_data = {
                'feature_engineer': self.feature_engineer,
                'model': self.model,
                'ensemble_predictor': self.ensemble_predictor,
                'performance_metrics': self.performance_metrics,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, filepath)
            logger.info(f"💾 Модель сохранена в {filepath}")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения модели: {e}")
    
    def load_model(self, filepath):
        """Загрузка модели"""
        try:
            model_data = joblib.load(filepath)
            self.feature_engineer = model_data['feature_engineer']
            self.model = model_data['model']
            self.ensemble_predictor = model_data['ensemble_predictor']
            self.performance_metrics = model_data['performance_metrics']
            self.is_trained = model_data['is_trained']
            logger.info(f"📂 Модель загружена из {filepath}")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")