import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import logging
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import IMAGES_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataVisualizer:
    def __init__(self):
        self.images_dir = IMAGES_DIR
    
    def create_employment_dashboard(self, df):
        """Создание дашборда по трудоустройству"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('📊 Анализ трудоустройства выпускников', fontsize=16, fontweight='bold')
            
            # 1. Уровень трудоустройства по факультетам
            employment_by_faculty = df.groupby('faculty')['employed'].mean().sort_values()
            axes[0, 0].barh(range(len(employment_by_faculty)), employment_by_faculty.values * 100,
                           color=plt.cm.viridis(np.linspace(0, 1, len(employment_by_faculty))))
            axes[0, 0].set_yticks(range(len(employment_by_faculty)))
            axes[0, 0].set_yticklabels(employment_by_faculty.index)
            axes[0, 0].set_xlabel('Доля трудоустроенных (%)')
            axes[0, 0].set_title('Уровень трудоустройства по факультетам')
            axes[0, 0].grid(axis='x', alpha=0.3)
            
            # 2. Распределение зарплат
            employed_df = df[df['employed'] == True]
            if len(employed_df) > 0:
                axes[0, 1].hist(employed_df['salary_byn'], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
                axes[0, 1].axvline(employed_df['salary_byn'].mean(), color='red', linestyle='--', linewidth=2,
                                  label=f'Среднее: {employed_df["salary_byn"].mean():.0f} BYN')
                axes[0, 1].set_xlabel('Зарплата (BYN)')
                axes[0, 1].set_ylabel('Количество выпускников')
                axes[0, 1].set_title('Распределение зарплат')
                axes[0, 1].legend()
                axes[0, 1].grid(alpha=0.3)
            
            # 3. Влияние GPA на трудоустройство
            gpa_bins = pd.cut(df['gpa'], bins=[5, 6, 7, 8, 9, 10])
            gpa_employment = df.groupby(gpa_bins)['employed'].mean()
            axes[1, 0].plot(range(len(gpa_employment)), gpa_employment.values * 100, 'o-', linewidth=2, markersize=8)
            axes[1, 0].set_xticks(range(len(gpa_employment)))
            axes[1, 0].set_xticklabels([f'{interval.left}-{interval.right}' for interval in gpa_employment.index])
            axes[1, 0].set_xlabel('Диапазон GPA')
            axes[1, 0].set_ylabel('Доля трудоустроенных (%)')
            axes[1, 0].set_title('Влияние успеваемости на трудоустройство')
            axes[1, 0].grid(alpha=0.3)
            
            # 4. Влияние стажировок
            internships_impact = df.groupby('internships')['employed'].mean()
            axes[1, 1].plot(internships_impact.index, internships_impact.values * 100, 'o-', linewidth=2, markersize=8, color='green')
            axes[1, 1].set_xlabel('Количество стажировок')
            axes[1, 1].set_ylabel('Доля трудоустроенных (%)')
            axes[1, 1].set_title('Влияние стажировок на трудоустройство')
            axes[1, 1].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.images_dir / 'employment_dashboard.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("✅ Дашборд трудоустройства сохранен")
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания дашборда: {e}")
    
    def create_salary_analysis(self, df):
        """Анализ зарплат"""
        try:
            employed_df = df[df['employed'] == True]
            
            if len(employed_df) == 0:
                logger.warning("❌ Нет данных о трудоустроенных выпускниках")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('💰 Анализ зарплат выпускников', fontsize=16, fontweight='bold')
            
            # 1. Зарплаты по факультетам
            salary_by_faculty = employed_df.groupby('faculty')['salary_byn'].mean().sort_values(ascending=False)
            axes[0, 0].bar(range(len(salary_by_faculty)), salary_by_faculty.values,
                          color=plt.cm.plasma(np.linspace(0, 1, len(salary_by_faculty))))
            axes[0, 0].set_xticks(range(len(salary_by_faculty)))
            axes[0, 0].set_xticklabels(salary_by_faculty.index, rotation=45)
            axes[0, 0].set_ylabel('Средняя зарплата (BYN)')
            axes[0, 0].set_title('Зарплаты по факультетам')
            axes[0, 0].grid(axis='y', alpha=0.3)
            
            # 2. Зарплаты по университетам
            salary_by_university = employed_df.groupby('university')['salary_byn'].mean().sort_values(ascending=False).head(10)
            axes[0, 1].bar(range(len(salary_by_university)), salary_by_university.values,
                          color=plt.cm.Set3(np.linspace(0, 1, len(salary_by_university))))
            axes[0, 1].set_xticks(range(len(salary_by_university)))
            axes[0, 1].set_xticklabels(salary_by_university.index, rotation=45)
            axes[0, 1].set_ylabel('Средняя зарплата (BYN)')
            axes[0, 1].set_title('Топ-10 университетов по зарплатам')
            axes[0, 1].grid(axis='y', alpha=0.3)
            
            # 3. Влияние GPA на зарплату
            axes[1, 0].scatter(employed_df['gpa'], employed_df['salary_byn'], alpha=0.6, s=50)
            axes[1, 0].set_xlabel('GPA')
            axes[1, 0].set_ylabel('Зарплата (BYN)')
            axes[1, 0].set_title('Влияние успеваемости на зарплату')
            axes[1, 0].grid(alpha=0.3)
            
            # 4. Влияние стажировок на зарплату
            internships_salary = employed_df.groupby('internships')['salary_byn'].mean()
            axes[1, 1].plot(internships_salary.index, internships_salary.values, 'o-', linewidth=2, markersize=8, color='red')
            axes[1, 1].set_xlabel('Количество стажировок')
            axes[1, 1].set_ylabel('Средняя зарплата (BYN)')
            axes[1, 1].set_title('Влияние стажировок на зарплату')
            axes[1, 1].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.images_dir / 'salary_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("✅ Анализ зарплат сохранен")
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа зарплат: {e}")
    
    def create_interactive_charts(self, df):
        """Создание интерактивных графиков"""
        try:
            # Интерактивный scatter plot
            employed_df = df[df['employed'] == True]
            
            if len(employed_df) > 0:
                fig = px.scatter(employed_df, x='gpa', y='salary_byn', color='faculty',
                               size='internships', hover_data=['university', 'projects'],
                               title='📊 Взаимосвязь GPA, зарплаты и стажировок')
                fig.write_html(str(self.images_dir / 'interactive_scatter.html'))
            
            # Treemap факультетов и зарплат
            faculty_salary = employed_df.groupby('faculty').agg({
                'salary_byn': 'mean',
                'student_id': 'count'
            }).reset_index()
            faculty_salary.columns = ['faculty', 'avg_salary', 'count']
            
            fig = px.treemap(faculty_salary, path=['faculty'], values='count',
                           color='avg_salary', color_continuous_scale='Viridis',
                           title='🎯 Распределение выпускников и зарплат по факультетам')
            fig.write_html(str(self.images_dir / 'faculty_treemap.html'))
            
            logger.info("✅ Интерактивные графики сохранены")
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания интерактивных графиков: {e}")

# Создание экземпляра для импорта
data_visualizer = DataVisualizer()