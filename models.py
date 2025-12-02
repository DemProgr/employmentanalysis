# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.svm import SVR, SVC
# from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.feature_selection import SelectFromModel
# import joblib
# from pathlib import Path
# import logging
# import sys
# import warnings
# from xgboost import XGBRegressor, XGBClassifier
# from lightgbm import LGBMRegressor, LGBMClassifier

# warnings.filterwarnings('ignore')

# # Добавляем корневую директорию проекта в путь
# project_root = Path(__file__).parent
# sys.path.insert(0, str(project_root))

# from config import MODELS_DIR, ML_CONFIG

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class EmploymentPredictor:
#     def __init__(self):
#         self.models_dir = MODELS_DIR
#         self.salary_model = None
#         self.employment_model = None
#         self.scaler = StandardScaler()
#         self.label_encoders = {}
#         self.is_trained = False
#         self.feature_names = []
    
#     def prepare_features(self, df):
#         """Подготовка признаков для ML"""
#         df_encoded = df.copy()
        
#         # Кодируем категориальные переменные
#         categorical_columns = ['faculty', 'university', 'location', 'specialization']
        
#         for col in categorical_columns:
#             if col in df_encoded.columns:
#                 if col not in self.label_encoders:
#                     self.label_encoders[col] = LabelEncoder()
#                     df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
#                 else:
#                     # Для новых данных используем transform, но обрабатываем неизвестные значения
#                     try:
#                         df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
#                     except ValueError:
#                         # Если встретилось новое значение, используем наиболее частую категорию
#                         df_encoded[col] = 0
        
#         # Выбираем числовые признаки
#         numeric_features = ['gpa', 'internships', 'projects', 'certificates', 'graduation_year']
#         self.feature_names = [col for col in numeric_features + categorical_columns if col in df_encoded.columns]
        
#         return df_encoded[self.feature_names], self.feature_names
    
#     def get_feature_importance(self):
#         """Получение важности признаков - ДОБАВЛЕННЫЙ МЕТОД"""
#         if self.salary_model and hasattr(self.salary_model, 'feature_importances_'):
#             return self.salary_model.feature_importances_
#         elif self.employment_model and hasattr(self.employment_model, 'feature_importances_'):
#             return self.employment_model.feature_importances_
#         return None
    
#     def train_salary_model(self, df):
#         """Обучение модели прогнозирования зарплаты"""
#         try:
#             logger.info("🎯 Обучение модели зарплаты...")
            
#             # Используем только трудоустроенных
#             employed_df = df[df['employed'] == True]
            
#             if len(employed_df) < 20:  # Уменьшили минимальное количество для обучения
#                 logger.warning("❌ Недостаточно данных для обучения модели зарплаты")
#                 return None
            
#             X, features = self.prepare_features(employed_df)
#             y = employed_df['salary_byn']
            
#             # Проверяем, что есть данные для обучения
#             if len(X) == 0 or len(y) == 0:
#                 logger.warning("❌ Нет данных для обучения модели зарплаты")
#                 return None
            
#             # Масштабирование
#             X_scaled = self.scaler.fit_transform(X)
            
#             # Разделение на train/test
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X_scaled, y, test_size=0.2, random_state=42
#             )
            
#             # Используем только одну модель для упрощения
#             model = RandomForestRegressor(
#                 n_estimators=50,  # Уменьшили для скорости
#                 random_state=42,
#                 min_samples_split=10,  # Добавили для избежания предупреждений
#                 min_samples_leaf=5
#             )
            
#             # Обучение модели
#             model.fit(X_train, y_train)
#             self.salary_model = model
            
#             # Оценка
#             y_pred = model.predict(X_test)
#             mae = mean_absolute_error(y_test, y_pred)
#             r2 = r2_score(y_test, y_pred)
            
#             logger.info(f"✅ Модель зарплаты обучена:")
#             logger.info(f"   MAE: {mae:.2f} BYN")
#             logger.info(f"   R²: {r2:.4f}")
            
#             return model
            
#         except Exception as e:
#             logger.error(f"❌ Ошибка обучения модели зарплаты: {e}")
#             return None
    
#     def train_employment_model(self, df):
#         """Обучение модели прогнозирования трудоустройства"""
#         try:
#             logger.info("🎯 Обучение модели трудоустройства...")
            
#             if len(df) < 50:  # Уменьшили минимальное количество
#                 logger.warning("❌ Недостаточно данных для обучения модели трудоустройства")
#                 return None
            
#             X, features = self.prepare_features(df)
#             y = df['employed']
            
#             # Проверяем, что есть данные для обучения
#             if len(X) == 0 or len(y) == 0:
#                 logger.warning("❌ Нет данных для обучения модели трудоустройства")
#                 return None
            
#             # Масштабирование
#             X_scaled = self.scaler.fit_transform(X)
            
#             # Разделение на train/test
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X_scaled, y, test_size=0.2, random_state=42, stratify=y
#             )
            
#             # Используем только одну модель для упрощения
#             model = RandomForestClassifier(
#                 n_estimators=50,  # Уменьшили для скорости
#                 random_state=42,
#                 min_samples_split=10,  # Добавили для избежания предупреждений
#                 min_samples_leaf=5
#             )
            
#             # Обучение модели
#             model.fit(X_train, y_train)
#             self.employment_model = model
            
#             # Оценка
#             y_pred = model.predict(X_test)
#             accuracy = accuracy_score(y_test, y_pred)
            
#             logger.info(f"✅ Модель трудоустройства обучена:")
#             logger.info(f"   Accuracy: {accuracy:.4f}")
            
#             return model
            
#         except Exception as e:
#             logger.error(f"❌ Ошибка обучения модели трудоустройства: {e}")
#             return None
    
#     def train_models(self, df):
#         """Обучение всех моделей"""
#         salary_model = self.train_salary_model(df)
#         employment_model = self.train_employment_model(df)
        
#         if salary_model or employment_model:  # Изменили на OR, чтобы работать даже если одна модель обучилась
#             self.is_trained = True
#             return True
#         return False
    
#     def predict_salary(self, student_data):
#         """Прогнозирование зарплаты"""
#         if not self.is_trained or self.salary_model is None:
#             logger.error("❌ Модель зарплаты не обучена")
#             return None
        
#         try:
#             X, _ = self.prepare_features(student_data)
#             X_scaled = self.scaler.transform(X)
#             prediction = self.salary_model.predict(X_scaled)[0]
#             return max(0, prediction)
#         except Exception as e:
#             logger.error(f"❌ Ошибка прогнозирования зарплаты: {e}")
#             return None
    
#     def predict_employment_probability(self, student_data):
#         """Прогнозирование вероятности трудоустройства"""
#         if not self.is_trained or self.employment_model is None:
#             logger.error("❌ Модель трудоустройства не обучена")
#             return None
        
#         try:
#             X, _ = self.prepare_features(student_data)
#             X_scaled = self.scaler.transform(X)
#             probability = self.employment_model.predict_proba(X_scaled)[0][1]
#             return probability
#         except Exception as e:
#             logger.error(f"❌ Ошибка прогнозирования трудоустройства: {e}")
#             return None
    
#     def save_models(self):
#         """Сохранение моделей"""
#         try:
#             if self.salary_model:
#                 joblib.dump(self.salary_model, self.models_dir / 'salary_model.pkl')
#             if self.employment_model:
#                 joblib.dump(self.employment_model, self.models_dir / 'employment_model.pkl')
#             if hasattr(self, 'scaler'):
#                 joblib.dump(self.scaler, self.models_dir / 'scaler.pkl')
#             if self.label_encoders:
#                 joblib.dump(self.label_encoders, self.models_dir / 'label_encoders.pkl')
            
#             logger.info("✅ Модели успешно сохранены")
#         except Exception as e:
#             logger.error(f"❌ Ошибка сохранения моделей: {e}")
    
#     def load_models(self):
#         """Загрузка моделей"""
#         try:
#             self.salary_model = joblib.load(self.models_dir / 'salary_model.pkl')
#             self.employment_model = joblib.load(self.models_dir / 'employment_model.pkl')
#             self.scaler = joblib.load(self.models_dir / 'scaler.pkl')
#             self.label_encoders = joblib.load(self.models_dir / 'label_encoders.pkl')
#             self.is_trained = True
            
#             logger.info("✅ Модели успешно загружены")
#         except Exception as e:
#             logger.error(f"❌ Ошибка загрузки моделей: {e}")

# class AdvancedEmploymentPredictor(EmploymentPredictor):
#     """Расширенный предсказатель с дополнительными функциями"""
    
#     def __init__(self):
#         super().__init__()
    
#     def create_advanced_features(self, df):
#         """Создание расширенных признаков"""
#         df_advanced = df.copy()
        
#         # Композитные признаки
#         df_advanced['total_experience'] = df_advanced['internships'] + df_advanced['projects'] * 0.5
#         df_advanced['skills_score'] = df_advanced['certificates'] * 10 + df_advanced['projects'] * 5
#         df_advanced['academic_performance'] = (df_advanced['gpa'] - 5) * 20
        
#         # Временные признаки
#         current_year = 2025
#         df_advanced['years_since_graduation'] = current_year - df_advanced['graduation_year']
        
#         return df_advanced

# # В models.py ДОБАВЛЯЕМ в класс SimplePredictor:

#         growth_multiplier = 1.09 ** years_ahead
#                     elif faculty == 'ИТ':
#                         # ИТ: замедление после 5 лет
#                         if years_ahead <= 5:
#                             growth_multiplier = 1.06 ** years_ahead
#                         else:
#                             early_growth = 1.06 ** 5
#                             late_growth = 1.03 ** (years_ahead - 5)
#                             growth_multiplier = early_growth * late_growth
#                     elif faculty == 'Медицина':
#                         # Медицина: стабильный высокий рост
#                         growth_multiplier = 1.07 ** years_ahead
#                     else:
#                         # Остальные: стандартный рост
#                         growth_rate = self.yearly_growth_rates.get(faculty, {'salary': 1.04})['salary']
#                         growth_multiplier = growth_rate ** years_ahead
                    
#                     predicted_salary = predicted_salary * growth_multiplier
                
#                 # Ограничения по рынку
#                 salary_limits = {
#                     'ИТ': (800, 6000),
#                     'Медицина': (700, 5000), 
#                     'Инженерия': (800, 4500),
#                     'Экономика': (600, 3500),
#                     'Педагогика': (500, 4000),  # ВЫШЕ ПРЕДЕЛ ИЗ-ЗА РОСТА СПРОСА
#                     'Юриспруденция': (700, 3800)
#                 }
                
#                 min_salary, max_salary = salary_limits.get(faculty, (600, 3000))
#                 return max(min_salary, min(predicted_salary, max_salary))


# if __name__ == "__main__":
#     # Тестирование моделей
#     from data_loader import RealDataLoader
    
#     loader = RealDataLoader()
#     graduates = loader.load_graduates_data()
    
#     if graduates is not None:
#         predictor = EmploymentPredictor()
#         success = predictor.train_models(graduates)
        
#         if success:
#             predictor.save_models()
            
#             # Тестирование
#             test_data = graduates.iloc[:1]
#             salary_pred = predictor.predict_salary(test_data)
#             employment_prob = predictor.predict_employment_probability(test_data)
            
#             print(f"\n🧪 ТЕСТИРОВАНИЕ МОДЕЛЕЙ:")
#             print(f"   Прогноз зарплаты: {salary_pred:.0f} BYN")
#             print(f"   Вероятность трудоустройства: {employment_prob:.1%}")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
import joblib
from pathlib import Path
import logging
import sys
import warnings
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

warnings.filterwarnings('ignore')

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import MODELS_DIR, ML_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmploymentPredictor:
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.salary_model = None
        self.employment_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        self.feature_names = []
    
    def prepare_features(self, df):
        """Подготовка признаков для ML"""
        df_encoded = df.copy()
        
        # Кодируем категориальные переменные
        categorical_columns = ['faculty', 'university', 'location', 'specialization']
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    # Для новых данных используем transform, но обрабатываем неизвестные значения
                    try:
                        df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
                    except ValueError:
                        # Если встретилось новое значение, используем наиболее частую категорию
                        df_encoded[col] = 0
        
        # Выбираем числовые признаки
        numeric_features = ['gpa', 'internships', 'projects', 'certificates', 'graduation_year']
        self.feature_names = [col for col in numeric_features + categorical_columns if col in df_encoded.columns]
        
        return df_encoded[self.feature_names], self.feature_names
    
    def get_feature_importance(self):
        """Получение важности признаков - ДОБАВЛЕННЫЙ МЕТОД"""
        if self.salary_model and hasattr(self.salary_model, 'feature_importances_'):
            return self.salary_model.feature_importances_
        elif self.employment_model and hasattr(self.employment_model, 'feature_importances_'):
            return self.employment_model.feature_importances_
        return None
    
    def train_salary_model(self, df):
        """Обучение модели прогнозирования зарплаты"""
        try:
            logger.info("🎯 Обучение модели зарплаты...")
            
            # Используем только трудоустроенных
            employed_df = df[df['employed'] == True]
            
            if len(employed_df) < 20:  # Уменьшили минимальное количество для обучения
                logger.warning("❌ Недостаточно данных для обучения модели зарплаты")
                return None
            
            X, features = self.prepare_features(employed_df)
            y = employed_df['salary_byn']
            
            # Проверяем, что есть данные для обучения
            if len(X) == 0 or len(y) == 0:
                logger.warning("❌ Нет данных для обучения модели зарплаты")
                return None
            
            # Масштабирование
            X_scaled = self.scaler.fit_transform(X)
            
            # Разделение на train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Используем только одну модель для упрощения
            model = RandomForestRegressor(
                n_estimators=50,  # Уменьшили для скорости
                random_state=42,
                min_samples_split=10,  # Добавили для избежания предупреждений
                min_samples_leaf=5
            )
            
            # Обучение модели
            model.fit(X_train, y_train)
            self.salary_model = model
            
            # Оценка
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"✅ Модель зарплаты обучена:")
            logger.info(f"   MAE: {mae:.2f} BYN")
            logger.info(f"   R²: {r2:.4f}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения модели зарплаты: {e}")
            return None
    
    def train_employment_model(self, df):
        """Обучение модели прогнозирования трудоустройства"""
        try:
            logger.info("🎯 Обучение модели трудоустройства...")
            
            if len(df) < 50:  # Уменьшили минимальное количество
                logger.warning("❌ Недостаточно данных для обучения модели трудоустройства")
                return None
            
            X, features = self.prepare_features(df)
            y = df['employed']
            
            # Проверяем, что есть данные для обучения
            if len(X) == 0 or len(y) == 0:
                logger.warning("❌ Нет данных для обучения модели трудоустройства")
                return None
            
            # Масштабирование
            X_scaled = self.scaler.fit_transform(X)
            
            # Разделение на train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Используем только одну модель для упрощения
            model = RandomForestClassifier(
                n_estimators=50,  # Уменьшили для скорости
                random_state=42,
                min_samples_split=10,  # Добавили для избежания предупреждений
                min_samples_leaf=5
            )
            
            # Обучение модели
            model.fit(X_train, y_train)
            self.employment_model = model
            
            # Оценка
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"✅ Модель трудоустройства обучена:")
            logger.info(f"   Accuracy: {accuracy:.4f}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения модели трудоустройства: {e}")
            return None
    
    def train_models(self, df):
        """Обучение всех моделей"""
        salary_model = self.train_salary_model(df)
        employment_model = self.train_employment_model(df)
        
        if salary_model or employment_model:  # Изменили на OR, чтобы работать даже если одна модель обучилась
            self.is_trained = True
            return True
        return False
    
    def predict_salary(self, student_data):
        """Прогнозирование зарплаты"""
        if not self.is_trained or self.salary_model is None:
            logger.error("❌ Модель зарплаты не обучена")
            return None
        
        try:
            X, _ = self.prepare_features(student_data)
            X_scaled = self.scaler.transform(X)
            prediction = self.salary_model.predict(X_scaled)[0]
            return max(0, prediction)
        except Exception as e:
            logger.error(f"❌ Ошибка прогнозирования зарплаты: {e}")
            return None
    
    def predict_employment_probability(self, student_data):
        """Прогнозирование вероятности трудоустройства"""
        if not self.is_trained or self.employment_model is None:
            logger.error("❌ Модель трудоустройства не обучена")
            return None
        
        try:
            X, _ = self.prepare_features(student_data)
            X_scaled = self.scaler.transform(X)
            probability = self.employment_model.predict_proba(X_scaled)[0][1]
            return probability
        except Exception as e:
            logger.error(f"❌ Ошибка прогнозирования трудоустройства: {e}")
            return None
    
    def save_models(self):
        """Сохранение моделей"""
        try:
            if self.salary_model:
                joblib.dump(self.salary_model, self.models_dir / 'salary_model.pkl')
            if self.employment_model:
                joblib.dump(self.employment_model, self.models_dir / 'employment_model.pkl')
            if hasattr(self, 'scaler'):
                joblib.dump(self.scaler, self.models_dir / 'scaler.pkl')
            if self.label_encoders:
                joblib.dump(self.label_encoders, self.models_dir / 'label_encoders.pkl')
            
            logger.info("✅ Модели успешно сохранены")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения моделей: {e}")
    
    def load_models(self):
        """Загрузка моделей"""
        try:
            self.salary_model = joblib.load(self.models_dir / 'salary_model.pkl')
            self.employment_model = joblib.load(self.models_dir / 'employment_model.pkl')
            self.scaler = joblib.load(self.models_dir / 'scaler.pkl')
            self.label_encoders = joblib.load(self.models_dir / 'label_encoders.pkl')
            self.is_trained = True
            
            logger.info("✅ Модели успешно загружены")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки моделей: {e}")

class AdvancedEmploymentPredictor(EmploymentPredictor):
    """Расширенный предсказатель с дополнительными функциями"""
    
    def __init__(self):
        super().__init__()
    
    def create_advanced_features(self, df):
        """Создание расширенных признаков"""
        df_advanced = df.copy()
        
        # Композитные признаки
        df_advanced['total_experience'] = df_advanced['internships'] + df_advanced['projects'] * 0.5
        df_advanced['skills_score'] = df_advanced['certificates'] * 10 + df_advanced['projects'] * 5
        df_advanced['academic_performance'] = (df_advanced['gpa'] - 5) * 20
        
        # Временные признаки
        current_year = 2025
        df_advanced['years_since_graduation'] = current_year - df_advanced['graduation_year']
        
        return df_advanced

# В models.py ДОБАВЛЯЕМ класс SimplePredictor:
class SimplePredictor:
    """Упрощенный предсказатель без ML для использования при ошибках"""
    
    def __init__(self):
        self.faculty_salaries = {
            'ИТ': 2200, 'Медицина': 1800, 'Инженерия': 1900,
            'Экономика': 1700, 'Педагогика': 1400, 'Юриспруденция': 1600
        }
        self.faculty_employment = {
            'ИТ': 0.85, 'Медицина': 0.80, 'Инженерия': 0.75,
            'Экономика': 0.70, 'Педагогика': 0.65, 'Юриспруденция': 0.72
        }
        self.yearly_growth_rates = {
            'ИТ': {'salary': 1.06, 'employment': 1.02},
            'Медицина': {'salary': 1.07, 'employment': 1.03},
            'Инженерия': {'salary': 1.05, 'employment': 1.015},
            'Экономика': {'salary': 1.04, 'employment': 1.01},
            'Педагогика': {'salary': 1.09, 'employment': 1.04},
            'Юриспруденция': {'salary': 1.045, 'employment': 1.012}
        }
    
    def predict_salary_simple(self, faculty, gpa, internships, projects, certificates, 
                            english_level, graduation_year, programming_skills=0, 
                            research_experience=0, leadership_experience=0, 
                            technical_skills=0, communication_skills=0):
        """Упрощенный прогноз зарплаты"""
        base_salary = self.faculty_salaries.get(faculty, 1500)
        
        # Модификаторы
        gpa_bonus = (gpa - 7.0) * 50
        internships_bonus = internships * 80
        projects_bonus = projects * 50
        certificates_bonus = certificates * 60
        
        # Бонус за английский
        english_bonus = 0
        if english_level in ['B2', 'C1', 'C2']:
            english_bonus = 200
        
        # Бонус за дополнительные навыки
        skills_bonus = (programming_skills * 40 + research_experience * 30 + 
                       leadership_experience * 35 + technical_skills * 45 + 
                       communication_skills * 25)
        
        total_salary = (base_salary + gpa_bonus + internships_bonus + 
                       projects_bonus + certificates_bonus + english_bonus + skills_bonus)
        
        # Корректировка на будущие годы (если выпуск еще не наступил)
        current_year = 2025
        if graduation_year > current_year:
            years_ahead = graduation_year - current_year
            if faculty == 'Педагогика':
                growth_multiplier = 1.09 ** years_ahead
            elif faculty == 'ИТ':
                # ИТ: замедление после 5 лет
                if years_ahead <= 5:
                    growth_multiplier = 1.06 ** years_ahead
                else:
                    early_growth = 1.06 ** 5
                    late_growth = 1.03 ** (years_ahead - 5)
                    growth_multiplier = early_growth * late_growth
            elif faculty == 'Медицина':
                # Медицина: стабильный высокий рост
                growth_multiplier = 1.07 ** years_ahead
            else:
                # Остальные: стандартный рост
                growth_rate = self.yearly_growth_rates.get(faculty, {'salary': 1.04})['salary']
                growth_multiplier = growth_rate ** years_ahead
            
            total_salary = total_salary * growth_multiplier
        
        # Ограничения по рынку
        salary_limits = {
            'ИТ': (800, 6000),
            'Медицина': (700, 5000), 
            'Инженерия': (800, 4500),
            'Экономика': (600, 3500),
            'Педагогика': (500, 4000),  # ВЫШЕ ПРЕДЕЛ ИЗ-ЗА РОСТА СПРОСА
            'Юриспруденция': (700, 3800)
        }
        
        min_salary, max_salary = salary_limits.get(faculty, (600, 3000))
        return max(min_salary, min(total_salary, max_salary))
    
    def predict_employment_simple(self, faculty, gpa, internships, projects, certificates,
                                job_search_duration, english_level, graduation_year,
                                programming_skills=0, research_experience=0,
                                leadership_experience=0, technical_skills=0,
                                communication_skills=0):
        """Упрощенный прогноз трудоустройства"""
        base_prob = self.faculty_employment.get(faculty, 0.6)
        
        # Модификаторы вероятности
        gpa_effect = (gpa - 7.0) * 0.03
        internships_effect = internships * 0.04
        projects_effect = projects * 0.025
        certificates_effect = certificates * 0.03
        
        # Эффект от дополнительных навыков
        skills_effect = (programming_skills * 0.02 + research_experience * 0.015 +
                        leadership_experience * 0.018 + technical_skills * 0.022 +
                        communication_skills * 0.012)
        
        # Эффект от английского
        english_effect = 0.05 if english_level in ['B2', 'C1', 'C2'] else 0
        
        total_prob = (base_prob + gpa_effect + internships_effect + projects_effect +
                     certificates_effect + skills_effect + english_effect)
        
        return max(0.1, min(0.95, total_prob))


if __name__ == "__main__":
    # Тестирование моделей
    from data_loader import RealDataLoader
    
    loader = RealDataLoader()
    graduates = loader.load_graduates_data()
    
    if graduates is not None:
        predictor = EmploymentPredictor()
        success = predictor.train_models(graduates)
        
        if success:
            predictor.save_models()
            
            # Тестирование
            test_data = graduates.iloc[:1]
            salary_pred = predictor.predict_salary(test_data)
            employment_prob = predictor.predict_employment_probability(test_data)
            
            print(f"\n🧪 ТЕСТИРОВАНИЕ МОДЕЛЕЙ:")
            print(f"   Прогноз зарплаты: {salary_pred:.0f} BYN")
            print(f"   Вероятность трудоустройства: {employment_prob:.1%}")