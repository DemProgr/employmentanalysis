"""
Демонстрация работы улучшенных моделей
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Добавляем пути
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from enhanced_predictor import EnhancedEmploymentPredictor
from data_provider import RealisticDataProvider

def demo_enhanced_models():
    """Демонстрация улучшенных моделей"""
    print("🚀 ДЕМОНСТРАЦИЯ УЛУЧШЕННЫХ ML МОДЕЛЕЙ")
    print("=" * 60)
    
    try:
        # Генерация демо-данных
        print("📊 Генерация демонстрационных данных...")
        provider = RealisticDataProvider()
        graduates_df = provider.generate_real_graduates(500)  # Уменьшено для скорости
        
        print(f"✅ Сгенерировано {len(graduates_df)} записей")
        
        # Инициализация улучшенного предсказателя
        print("🤖 Инициализация улучшенного предсказателя...")
        enhanced_predictor = EnhancedEmploymentPredictor(use_ensemble=True)
        
        # Обучение
        print("🎯 Обучение улучшенных моделей...")
        success = enhanced_predictor.train(graduates_df)
        
        if success:
            print("✅ Обучение завершено успешно!")
            
            # Демонстрация метрик
            metrics = enhanced_predictor.get_model_performance()
            print("\n📊 МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ:")
            for model_name, model_metrics in metrics.items():
                print(f"\n{model_name}:")
                for metric, value in model_metrics.items():
                    print(f"  {metric}: {value:.4f}")
            
            # Демонстрация важности признаков
            print("\n🔍 ВАЖНОСТЬ ПРИЗНАКОВ (топ-10):")
            feature_importance = enhanced_predictor.get_feature_importance(10)
            if feature_importance:
                for feature, importance in feature_importance:
                    print(f"  {feature}: {importance:.4f}")
            
            # Демонстрация прогнозирования
            print("\n🔮 ДЕМО ПРОГНОЗИРОВАНИЯ:")
            test_student = graduates_df.iloc[[0]]  # Первый студент
            probability = enhanced_predictor.predict_employment_probability(test_student)
            actual_employment = test_student['employed'].iloc[0]
            
            print(f"  Прогнозируемая вероятность: {probability:.1%}")
            print(f"  Фактический статус: {'Трудоустроен' if actual_employment else 'Не трудоустроен'}")
            print(f"  Совпадение: {'✅' if (probability > 0.5) == actual_employment else '❌'}")
            
            # Сохранение моделей
            print("\n💾 Сохранение моделей...")
            enhanced_predictor.save_models()
            
        else:
            print("❌ Обучение не удалось")
            
    except Exception as e:
        print(f"❌ Ошибка демонстрации: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_enhanced_models()