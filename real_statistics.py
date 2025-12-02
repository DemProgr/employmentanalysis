# real_statistics.py
"""
Базовая реализация статистики для проекта
"""

class BelarusEducationStatistics:
    """Базовая статистика образования Беларуси"""
    
    def __init__(self):
        self.faculty_distribution = {
            'ИТ': 0.18, 'Медицина': 0.12, 'Инженерия': 0.22,
            'Экономика': 0.25, 'Педагогика': 0.15, 'Юриспруденция': 0.08
        }
        
        self.university_distribution = {
            'БГУ': 0.152, 'БГУИР': 0.128, 'БНТУ': 0.145, 'БГМУ': 0.083,
            'БГЭУ': 0.091, 'БГПУ': 0.078, 'ГрГУ': 0.062, 'ВГУ': 0.054,
            'ГГТУ': 0.049, 'ПГУ': 0.041
        }
        
        self.regional_distribution = {
            'Минск': 0.45, 'Гродно': 0.12, 'Витебск': 0.11,
            'Гомель': 0.11, 'Могилев': 0.10, 'Брест': 0.11
        }

class RealStatisticsDataProvider:
    """Поставщик данных на основе реальной статистики"""
    
    def __init__(self):
        self.stats = BelarusEducationStatistics()
    
    def calculate_graduate_distribution(self, target_total=100000):
        """Рассчитывает распределение выпускников"""
        distribution = []
        
        for faculty, faculty_share in self.stats.faculty_distribution.items():
            faculty_count = int(target_total * faculty_share)
            
            for uni, uni_share in self.stats.university_distribution.items():
                uni_count = int(faculty_count * uni_share)
                if uni_count == 0:
                    continue
                    
                for region, region_share in self.stats.regional_distribution.items():
                    region_count = int(uni_count * region_share)
                    if region_count == 0:
                        continue
                    
                    distribution.append({
                        'faculty': faculty,
                        'university': uni,
                        'region': region,
                        'count': region_count
                    })
        
        return distribution

# Глобальный экземпляр для импорта
real_stats_provider = RealStatisticsDataProvider()