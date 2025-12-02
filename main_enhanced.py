"""
Обновленный основной пайплайн с улучшенными моделями
"""

import logging
import sys
from pathlib import Path

# Добавляем пути для импорта
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data_loader import RealDataLoader
from enhanced_predictor import EnhancedEmploymentPredictor

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Обновленный основной пайплайн с улучшенными моделями"""
    
    logger.info("🚀 Запуск улучшенного пайплайна ML моделей...")
    
    try:
        # Загрузка данных
        logger.info("📊 Загрузка данных...")
        loader = RealDataLoader()
        graduates_df = loader.load_graduates_data()
        
        if graduates_df is None or len(graduates_df) == 0:
            logger.error("❌ Не удалось загрузить данные выпускников")
            return
        
        logger.info(f"✅ Загружено {len(graduates_df)} записей выпускников")
        
        # Инициализация улучшенного предсказателя
        logger.info("🤖 Инициализация улучшенного предсказателя...")
        enhanced_predictor = EnhancedEmploymentPredictor(use_ensemble=True)
        
        # Обучение улучшенных моделей
        logger.info("🎯 Обучение улучшенных моделей...")
        success = enhanced_predictor.train(graduates_df)
        
        if success:
            logger.info("✅ Обучение завершено успешно!")
            
            # Сохранение моделей
            enhanced_predictor.save_models()
            
            # Анализ важности признаков
            feature_importance = enhanced_predictor.get_feature_importance(10)
            if feature_importance:
                logger.info("🔍 Важные признаки (топ-10):")
                for feature, importance in feature_importance:
                    logger.info(f"   {feature}: {importance:.4f}")
            
            # Демонстрация прогнозирования
            logger.info("🔮 Тестирование прогнозирования...")
            test_data = graduates_df.iloc[:3]  # Тестируем на 3 примерах
            for i in range(len(test_data)):
                student_data = test_data.iloc[[i]]
                probability = enhanced_predictor.predict_employment_probability(student_data)
                actual_employment = student_data['employed'].iloc[0]
                
                logger.info(f"   Студент {i+1}: Прогноз={probability:.1%}, Факт={'Трудоустроен' if actual_employment else 'Не трудоустроен'}")
            
            # Метрики производительности
            metrics = enhanced_predictor.get_model_performance()
            logger.info("📊 Финальные метрики производительности:")
            for model_name, model_metrics in metrics.items():
                logger.info(f"   {model_name}:")
                for metric, value in model_metrics.items():
                    logger.info(f"     {metric}: {value:.4f}")
                    
        else:
            logger.error("❌ Обучение улучшенных моделей не удалось")
            
    except Exception as e:
        logger.error(f"❌ Критическая ошибка в основном пайплайне: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()