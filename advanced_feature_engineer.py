# advanced_feature_engineer.py - ИСПРАВЛЕННАЯ ВЕРСИЯ

"""
Продвинутый инжиниринг признаков для системы прогнозирования трудоустройства
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Трансформер для создания расширенных признаков"""
    
    def __init__(self):
        self.faculty_employment_rates = None
        self.university_prestige_scores = None
        self.location_economic_scores = None
        
    def fit(self, X, y=None):
        # Вычисляем таргет-энкодинг для категориальных переменных
        if y is not None:
            self.faculty_employment_rates = X.groupby('faculty')['employed'].mean()
            
            # Рассчитываем престиж университета на основе зарплат выпускников
            university_stats = X.groupby('university').agg({
                'salary_byn': 'mean',
                'employed': 'mean',
                'gpa': 'mean'
            })
            
            if len(university_stats) > 1:
                salary_min = university_stats['salary_byn'].min()
                salary_max = university_stats['salary_byn'].max()
                if salary_max > salary_min:
                    self.university_prestige_scores = (
                        (university_stats['salary_byn'] - salary_min) / 
                        (salary_max - salary_min) * 10
                    )
                else:
                    self.university_prestige_scores = pd.Series(5.0, index=university_stats.index)
            else:
                self.university_prestige_scores = pd.Series(5.0, index=university_stats.index)
            
            # Экономический показатель региона
            location_stats = X.groupby('location').agg({
                'salary_byn': 'mean',
                'employed': 'mean'
            })
            
            if len(location_stats) > 1:
                salary_min = location_stats['salary_byn'].min()
                salary_max = location_stats['salary_byn'].max()
                if salary_max > salary_min:
                    self.location_economic_scores = (
                        (location_stats['salary_byn'] - salary_min) / 
                        (salary_max - salary_min) * 10
                    )
                else:
                    self.location_economic_scores = pd.Series(5.0, index=location_stats.index)
            else:
                self.location_economic_scores = pd.Series(5.0, index=location_stats.index)
            
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        # Временные признаки
        current_year = 2025
        X_transformed['years_since_graduation'] = current_year - X_transformed['graduation_year']
        X_transformed['is_recent_graduate'] = (X_transformed['years_since_graduation'] <= 1).astype(int)
        
        # 🔥 ИСПРАВЛЕНО: Правильное вычисление skills_diversity
        # Используем только существующие колонки
        if 'internships' in X_transformed.columns and 'projects' in X_transformed.columns and 'certificates' in X_transformed.columns:
            X_transformed['skills_diversity'] = (
                (X_transformed['internships'] > 0).astype(int) * 2 +
                (X_transformed['projects'] > 0).astype(int) * 1.5 +
                (X_transformed['certificates'] > 0).astype(int) * 1
            )
        else:
            X_transformed['skills_diversity'] = 0
        
        # 🔥 ИСПРАВЛЕНО: Используем skills_diversity только после его создания
        X_transformed['total_experience_score'] = (
            X_transformed['internships'] * 0.4 + 
            X_transformed['projects'] * 0.3 + 
            X_transformed['certificates'] * 0.2 +
            X_transformed['skills_diversity'] * 0.1
        )
        
        X_transformed['academic_performance_index'] = (
            X_transformed['gpa'] * 0.6 + 
            (X_transformed['projects'] / 10) * 0.4
        )
        
        # Индекс карьерной готовности
        X_transformed['career_readiness_index'] = (
            X_transformed['gpa'] * 0.25 +
            X_transformed['total_experience_score'] * 0.35 +
            X_transformed['skills_diversity'] * 0.20 +
            (X_transformed['graduation_year'] - 2010) * 0.10
        )
        
        # Добавляем проверку для job_search_duration
        if 'job_search_duration' in X_transformed.columns:
            X_transformed['career_readiness_index'] += (X_transformed['job_search_duration'] <= 30).astype(int) * 0.10
        
        # Признаки взаимодействия
        X_transformed['gpa_experience_interaction'] = X_transformed['gpa'] * X_transformed['total_experience_score']
        
        # 🔥 ИСПРАВЛЕНО: Безопасное применение location_premium
        if 'location' in X_transformed.columns:
            X_transformed['location_premium'] = X_transformed['location'].apply(
                lambda x: 1.5 if x == 'Минск' else 1.2 if x in ['Гродно', 'Брест'] else 1.0
            )
        else:
            X_transformed['location_premium'] = 1.0
        
        # Таргет-энкодинг
        if self.faculty_employment_rates is not None and 'faculty' in X_transformed.columns:
            X_transformed['faculty_employment_rate'] = X_transformed['faculty'].map(self.faculty_employment_rates)
            X_transformed['faculty_employment_rate'].fillna(0.5, inplace=True)
        else:
            X_transformed['faculty_employment_rate'] = 0.5
            
        if self.university_prestige_scores is not None and 'university' in X_transformed.columns:
            X_transformed['university_prestige_score'] = X_transformed['university'].map(self.university_prestige_scores)
            X_transformed['university_prestige_score'].fillna(5.0, inplace=True)
        else:
            X_transformed['university_prestige_score'] = 5.0
            
        if self.location_economic_scores is not None and 'location' in X_transformed.columns:
            X_transformed['location_economic_score'] = X_transformed['location'].map(self.location_economic_scores)
            X_transformed['location_economic_score'].fillna(5.0, inplace=True)
        else:
            X_transformed['location_economic_score'] = 5.0
        
        # Бинарные признаки
        X_transformed['has_high_gpa'] = (X_transformed['gpa'] >= 7.5).astype(int)
        X_transformed['has_multiple_internships'] = (X_transformed['internships'] >= 1).astype(int)
        X_transformed['has_projects'] = (X_transformed['projects'] >= 2).astype(int)
        X_transformed['has_certificates'] = (X_transformed['certificates'] >= 1).astype(int)
        
        # Индекс конкурентоспособности на рынке труда
        X_transformed['market_competitiveness_index'] = (
            X_transformed['university_prestige_score'] * 0.3 +
            X_transformed['faculty_employment_rate'] * 0.3 +
            X_transformed['career_readiness_index'] * 0.2 +
            X_transformed['location_economic_score'] * 0.2
        )
        
        return X_transformed

class AdvancedFeatureEngineer:
    """Продвинутый инжиниринг признаков для прогнозирования трудоустройства"""
    
    def __init__(self):
        self.preprocessor = None
        self.feature_transformer = FeatureEngineeringTransformer()
        self.numeric_features = []
        self.categorical_features = []
        
    def build_preprocessor(self, df):
        """Создание комплексного пайплайна предобработки"""
        
        # 🔥 ИСПРАВЛЕНО: Определяем признаки ПОСЛЕ применения трансформера
        # Сначала применяем трансформер, чтобы получить все признаки
        df_transformed = self.feature_transformer.fit_transform(df)
        
        # 🔥 ИСПРАВЛЕНО: Определяем признаки, которые действительно существуют
        all_possible_features = [
            # Базовые признаки
            'gpa', 'internships', 'projects', 'certificates', 'graduation_year',
            'salary_byn', 'job_search_duration',
            
            # Созданные признаки
            'years_since_graduation', 'total_experience_score', 'academic_performance_index',
            'gpa_experience_interaction', 'location_premium', 'faculty_employment_rate',
            'university_prestige_score', 'location_economic_score', 'career_readiness_index',
            'market_competitiveness_index', 'skills_diversity',
            
            # Бинарные признаки
            'is_recent_graduate', 'has_high_gpa', 'has_multiple_internships', 
            'has_projects', 'has_certificates'
        ]
        
        # Фильтруем только те признаки, которые действительно есть в данных
        self.numeric_features = [f for f in all_possible_features if f in df_transformed.columns]
        
        # 🔥 ИСПРАВЛЕНО: Исключаем категориальные признаки из обработки
        self.categorical_features = []
        
        # Пайплайны для числовых признаков
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # 🔥 ИСПРАВЛЕНО: Используем только существующие числовые признаки
        valid_features = [f for f in self.numeric_features if f in df.columns or f in df_transformed.columns]
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, valid_features),
            ],
            remainder='drop',
            n_jobs=-1
        )
        
        return self.preprocessor
    
    def prepare_features(self, df, target_column='employed', fit=True):
        """Подготовка признаков для обучения"""
        try:
            # Применяем feature engineering
            if fit:
                df_processed = self.feature_transformer.fit_transform(df, df[target_column] if target_column in df.columns else None)
                # Строим препроцессор на преобразованных данных
                if self.preprocessor is None:
                    self.build_preprocessor(df_processed)
            else:
                df_processed = self.feature_transformer.transform(df)
            
            # Разделяем на признаки и целевую переменную
            if target_column in df_processed.columns:
                X = df_processed.drop(columns=[target_column])
                y = df_processed[target_column]
            else:
                X = df_processed
                y = None
            
            # 🔥 ИСПРАВЛЕНО: Убедимся, что все признаки существуют
            available_features = [col for col in X.columns if col in self.numeric_features]
            
            # Применяем препроцессор
            if fit:
                X_processed = self.preprocessor.fit_transform(X[available_features])
            else:
                X_processed = self.preprocessor.transform(X[available_features])
            
            # Получаем имена признаков
            feature_names = self.get_feature_names()
            
            return X_processed, y, feature_names
            
        except Exception as e:
            logger.error(f"❌ Ошибка подготовки признаков: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def get_feature_names(self):
        """Получение имен признаков после преобразования"""
        if self.preprocessor is None:
            return []
        
        # 🔥 ИСПРАВЛЕНО: Возвращаем только числовые признаки
        feature_names = list(self.numeric_features)
        
        return feature_names