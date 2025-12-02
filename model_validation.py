"""
Комплексная валидация и мониторинг ML моделей
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, learning_curve, validation_curve, StratifiedKFold
from sklearn.metrics import (roc_curve, precision_recall_curve, confusion_matrix, 
                           classification_report, roc_auc_score, accuracy_score, 
                           precision_score, recall_score, f1_score, average_precision_score)
from sklearn.calibration import calibration_curve
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ModelValidator:
    """Комплексная валидация ML моделей"""
    
    def __init__(self):
        self.validation_results = {}
        self.cv_results = {}
        
    def comprehensive_cross_validation(self, model, X, y, cv_strategy=5):
        """Комплексная кросс-валидация с multiple метриками"""
        try:
            scoring_metrics = {
                'accuracy': 'accuracy',
                'precision': 'precision',
                'recall': 'recall',
                'f1': 'f1',
                'roc_auc': 'roc_auc',
                'average_precision': 'average_precision'
            }
            
            cv_results = cross_validate(
                model, X, y,
                cv=StratifiedKFold(n_splits=min(cv_strategy, 5), shuffle=True, random_state=42),
                scoring=scoring_metrics,
                return_train_score=True,
                n_jobs=-1
            )
            
            self.cv_results = cv_results
            
            # Анализ результатов
            logger.info("📊 Результаты кросс-валидации:")
            for metric in scoring_metrics.keys():
                test_scores = cv_results[f'test_{metric}']
                train_scores = cv_results[f'train_{metric}']
                
                test_mean = np.mean(test_scores)
                test_std = np.std(test_scores)
                train_mean = np.mean(train_scores)
                train_std = np.std(train_scores)
                
                logger.info(f"   {metric.upper():20} | "
                          f"Train: {train_mean:.4f} ± {train_std:.4f} | "
                          f"Test: {test_mean:.4f} ± {test_std:.4f}")
                
                # Проверка на переобучение
                if train_mean - test_mean > 0.1:
                    logger.warning(f"   ⚠️ Возможное переобучение для {metric}")
            
            return cv_results
            
        except Exception as e:
            logger.error(f"❌ Ошибка кросс-валидации: {e}")
            return {}
    
    def plot_learning_curve(self, model, X, y, title="Кривая обучения"):
        """Построение кривой обучения"""
        try:
            # 🔥 ИСПРАВЛЕНО: правильное использование cv параметра
            n_folds = min(5, len(np.unique(y)))  # Ограничиваем количество фолдов
            cv_strategy = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, 
                cv=cv_strategy,  # 🔥 ИСПРАВЛЕНО: передаем объект StratifiedKFold
                n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 5),
                scoring='accuracy',
                random_state=42
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Кривые обучения
            ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", 
                    label="Обучающая выборка", linewidth=2)
            ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="g", 
                    label="Валидационная выборка", linewidth=2)
            ax.fill_between(train_sizes, 
                        np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                        np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                        alpha=0.1, color="r")
            ax.fill_between(train_sizes,
                        np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
                        np.mean(test_scores, axis=1) + np.std(test_scores, axis=1),
                        alpha=0.1, color="g")
            ax.set_xlabel("Размер обучающей выборки")
            ax.set_ylabel("Accuracy")
            ax.set_title(title)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Ошибка построения кривой обучения: {e}")
            return None
    
    def plot_calibration_curve(self, y_true, y_prob, model_name="Модель"):
        """Построение кривой калибровки"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Кривая калибровки
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=10
            )
            
            ax1.plot(mean_predicted_value, fraction_of_positives, "s-", 
                    label=model_name, linewidth=2)
            ax1.plot([0, 1], [0, 1], "k:", label="Идеально откалиброванная")
            ax1.set_xlabel("Средняя предсказанная вероятность")
            ax1.set_ylabel("Доля положительных классов")
            ax1.set_title("Кривая калибровки")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Гистограмма вероятностей
            ax2.hist(y_prob, bins=20, edgecolor='black', alpha=0.7)
            ax2.set_xlabel("Предсказанная вероятность")
            ax2.set_ylabel("Частота")
            ax2.set_title("Распределение предсказанных вероятностей")
            ax2.grid(True, alpha=0.3)
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = roc_auc_score(y_true, y_prob)
            ax3.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('ROC Curve')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            avg_precision = average_precision_score(y_true, y_prob)
            ax4.plot(recall, precision, color='blue', lw=2,
                    label=f'Precision-Recall (AP = {avg_precision:.2f})')
            ax4.set_xlabel('Recall')
            ax4.set_ylabel('Precision')
            ax4.set_title('Precision-Recall Curve')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"❌ Ошибка построения кривых: {e}")
            return None
    
    def generate_validation_report(self, model, X, y, y_pred, y_prob, model_name):
        """Генерация комплексного отчета по валидации"""
        try:
            report = {
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'dataset_size': len(X),
                'positive_class_ratio': np.mean(y),
                'metrics': {},
                'confusion_matrix': {},
                'feature_importance': {}
            }
            
            # Базовые метрики
            report['metrics'] = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1': f1_score(y, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y, y_prob),
                'average_precision': average_precision_score(y, y_prob)
            }
            
            # Матрица ошибок
            cm = confusion_matrix(y, y_pred)
            report['confusion_matrix'] = {
                'true_negative': int(cm[0, 0]),
                'false_positive': int(cm[0, 1]),
                'false_negative': int(cm[1, 0]),
                'true_positive': int(cm[1, 1])
            }
            
            # Важность признаков (если доступна)
            if hasattr(model, 'feature_importances_'):
                report['feature_importance'] = {
                    'available': True,
                    'top_features': []
                }
            else:
                report['feature_importance'] = {'available': False}
            
            # Сохранение отчета
            self.validation_results[model_name] = report
            
            logger.info(f"📋 Отчет валидации для {model_name}:")
            for metric, value in report['metrics'].items():
                logger.info(f"   {metric}: {value:.4f}")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации отчета: {e}")
            return {}

class ModelMonitor:
    """Мониторинг дрейфа данных и производительности моделей"""
    
    def __init__(self):
        self.performance_history = []
        self.data_drift_metrics = {}
        
    def check_data_drift(self, current_data, reference_data, numerical_columns):
        """Проверка дрейфа данных"""
        try:
            drift_metrics = {}
            
            for col in numerical_columns:
                if col in current_data.columns and col in reference_data.columns:
                    # Проверка различий в распределениях (KS test)
                    from scipy import stats
                    try:
                        statistic, p_value = stats.ks_2samp(
                            reference_data[col].dropna(), 
                            current_data[col].dropna()
                        )
                        
                        drift_metrics[col] = {
                            'ks_statistic': statistic,
                            'p_value': p_value,
                            'drift_detected': p_value < 0.05
                        }
                    except:
                        drift_metrics[col] = {
                            'ks_statistic': None,
                            'p_value': None,
                            'drift_detected': False
                        }
            
            self.data_drift_metrics = drift_metrics
            
            # Логирование дрейфа
            drifted_columns = [col for col, metrics in drift_metrics.items() 
                             if metrics.get('drift_detected', False)]
            if drifted_columns:
                logger.warning(f"⚠️ Обнаружен дрейф в колонках: {drifted_columns}")
            
            return drift_metrics
            
        except Exception as e:
            logger.error(f"❌ Ошибка проверки дрейфа данных: {e}")
            return {}
    
    def log_performance(self, model_name, metrics, dataset_info):
        """Логирование производительности модели"""
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'metrics': metrics,
            'dataset_info': dataset_info
        }
        
        self.performance_history.append(performance_record)
        
        # Сохранение в файл
        try:
            with open('model_performance_history.json', 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения истории производительности: {e}")