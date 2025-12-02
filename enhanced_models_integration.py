# enhanced_models_integration.py
class EnhancedEmploymentPredictor(EmploymentPredictor):
    """Улучшенный предсказатель с продвинутыми моделями"""
    
    def __init__(self):
        super().__init__()
        self.advanced_predictor = AdvancedEmploymentPredictor()
        self.validator = ModelValidator()
        
    def train_enhanced_models(self, df):
        """Обучение улучшенных моделей"""
        logger.info("🚀 Запуск улучшенного обучения...")
        
        # Используем продвинутые методы
        success = self.advanced_predictor.train_advanced_models(df)
        
        if success:
            # Валидация модели
            X, y, _ = self.advanced_predictor.prepare_data(df)
            self.validator.comprehensive_validation(
                self.advanced_predictor.best_model, X, y
            )
            
            logger.info("✅ Улучшенные модели успешно обучены и валидированы")
            return True
        else:
            logger.warning("⚠️ Возврат к базовым моделям")
            return super().train_models(df)
    
    def predict_employment_enhanced(self, student_data):
        """Улучшенное прогнозирование"""
        if self.advanced_predictor.is_trained:
            return self.advanced_predictor.predict_employment_probability(student_data)
        else:
            logger.warning("⚠️ Используется базовая модель")
            return super().predict_employment_probability(student_data)
    
    def get_detailed_feature_analysis(self):
        """Детальный анализ признаков"""
        if self.advanced_predictor.is_trained:
            return self.advanced_predictor.get_feature_importance()
        return None