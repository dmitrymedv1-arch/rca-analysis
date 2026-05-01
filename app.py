# -*- coding: utf-8 -*-
"""Scientific Data Visualization Dashboard - Streamlit Version"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from collections import Counter, defaultdict
import itertools
import warnings
import io
import zipfile
import json
import os
import sys
import traceback
from datetime import datetime
from tqdm import tqdm
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from wordcloud import WordCloud
from streamlit_option_menu import option_menu
import tempfile
import time
import colorsys  # Для плавных градиентов в хордовых диаграммах

# Настройка страницы Streamlit
st.set_page_config(
    page_title="Scientific Data Visualization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Настройка стиля для научных графиков
def set_scientific_style():
    """Установка научного стиля для графиков"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.2,
        'grid.linestyle': '--',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.title_fontsize': 11,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'legend.framealpha': 0.95,
        'legend.fancybox': False,
        'figure.titlesize': 16,
        'figure.titleweight': 'bold',
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'black',
        'savefig.bbox': 'tight',
    })
    
    # Цветовая палитра для научных графиков
    scientific_palette = [
        '#2E86AB', '#C73E1D', '#F18F01', '#6B8E23', '#8B5FBF',
        '#00A896', '#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0',
        '#118AB2', '#EF476F', '#073B4C', '#7209B7', '#F72585'
    ]
    
    plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=scientific_palette)

set_scientific_style()

# ============================================================================
# КЛАСС ДЛЯ ОБРАБОТКИ И ВИЗУАЛИЗАЦИИ ДАННЫХ
# ============================================================================

class ScientificDataAnalyzer:
    """Класс для анализа и визуализации научных данных"""
    
    def __init__(self):
        self.df = None
        self.df_processed = None
        self.all_figures = {}
        self.plot_data = {}  # Хранит данные для каждого графика
        self.errors = []
        self.warnings = []
        self.progress = 0
        # Настройки визуализации
        self.show_regression_trends = True
        self.top_countries_chord = 20
        self.top_fields_sankey = 10
        
    def log_error(self, error_msg, details=""):
        """Логирование ошибки"""
        self.errors.append({
            'timestamp': datetime.now().isoformat(),
            'message': error_msg,
            'details': details
        })
        st.error(f"❌ ERROR: {error_msg}")
        if details:
            st.error(f"   Details: {details}")
    
    def log_warning(self, warning_msg):
        """Логирование предупреждения"""
        self.warnings.append({
            'timestamp': datetime.now().isoformat(),
            'message': warning_msg
        })
        st.warning(f"⚠️ WARNING: {warning_msg}")
    
    def update_progress(self, value):
        """Обновление прогресса"""
        self.progress = value
    
    def update_visualization_settings(self, show_regression_trends=None, top_countries_chord=None,
                                       top_fields_sankey=None):
        """Обновление настроек визуализации"""
        if show_regression_trends is not None:
            self.show_regression_trends = show_regression_trends
        if top_countries_chord is not None:
            self.top_countries_chord = top_countries_chord
        if top_fields_sankey is not None:
            self.top_fields_sankey = top_fields_sankey
    
    def parse_data(self, data_text):
        """Парсинг данных из текстового ввода с расширенной диагностикой"""
        st.info("🔍 Parsing data...")
        
        try:
            # Разбиваем текст на строки
            lines = data_text.strip().split('\n')
            if len(lines) < 2:
                self.log_error("Not enough data rows", f"Found {len(lines)} lines")
                return None
            
            # Парсим заголовки
            headers = lines[0].split('\t')
            st.info(f"   Found {len(headers)} columns")
            st.info(f"   Headers: {headers}")
            
            # Проверяем необходимые колонки
            required_columns = ['doi', 'Title', 'year', 'count']
            missing_columns = [col for col in required_columns if col not in headers]
            if missing_columns:
                self.log_warning(f"Missing columns: {missing_columns}")
            
            # Парсим данные
            data = []
            for i, line in enumerate(lines[1:]):
                if line.strip():
                    values = line.split('\t')
                    # Заполняем недостающие значения
                    while len(values) < len(headers):
                        values.append('')
                    data.append(values)
            
            # Создаем DataFrame
            self.df = pd.DataFrame(data, columns=headers)
            st.success(f"✅ Successfully parsed {len(self.df)} rows")
            
            # Диагностика данных
            self._diagnose_data()
            
            # Предобработка данных
            self.df_processed = self._preprocess_data(self.df)
            
            return self.df_processed
            
        except Exception as e:
            self.log_error(f"Error parsing data: {str(e)}", traceback.format_exc())
            return None
    
    def _diagnose_data(self):
        """Диагностика качества данных"""
        st.info("🔬 Data Diagnostics:")
        st.write("---")
        
        # Проверка пропущенных значений
        missing_counts = self.df.isnull().sum()
        total_cells = np.prod(self.df.shape)
        missing_percent = (missing_counts.sum() / total_cells) * 100
        
        st.info(f"   Total cells: {total_cells:,}")
        st.info(f"   Missing values: {missing_counts.sum():,} ({missing_percent:.1f}%)")
        
        # Топ колонок с пропусками
        top_missing = missing_counts[missing_counts > 0].sort_values(ascending=False).head(5)
        if len(top_missing) > 0:
            st.info("   Top columns with missing values:")
            for col, count in top_missing.items():
                percent = (count / len(self.df)) * 100
                st.info(f"     - {col}: {count:,} ({percent:.1f}%)")
        
        # Проверка числовых колонок
        numeric_cols = ['author count', 'year', 'Citation counts (CR)', 'Citation counts (OA)',
                       'Annual cit counts (CR)', 'Annual cit counts (OA)', 'references_count', 'count']
        
        for col in numeric_cols:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    non_null = self.df[col].notnull().sum()
                    percent = (non_null / len(self.df)) * 100
                    if percent < 90:
                        self.log_warning(f"Low data coverage in '{col}': {percent:.1f}%")
                except:
                    self.log_warning(f"Cannot convert '{col}' to numeric")
        
        # Статистика по ключевым колонкам
        st.info("📊 Key Statistics:")
        if 'year' in self.df.columns:
            self.df['year'] = pd.to_numeric(self.df['year'], errors='coerce')
            valid_years = self.df['year'].notnull().sum()
            year_range = f"{int(self.df['year'].min())}-{int(self.df['year'].max())}"
            st.info(f"   Year range: {year_range} ({valid_years}/{len(self.df)} valid)")
        
        if 'count' in self.df.columns:
            self.df['count'] = pd.to_numeric(self.df['count'], errors='coerce')
            st.info(f"   Total mentions (count): {self.df['count'].sum():,.0f}")
            st.info(f"   Avg mentions per paper: {self.df['count'].mean():.2f} ± {self.df['count'].std():.2f}")
        
        st.write("---")
    
    def _preprocess_data(self, df):
        """Предобработка данных"""
        st.info("🔄 Preprocessing data...")
        
        df_processed = df.copy()
        
        # Преобразование числовых колонок
        numeric_cols = ['author count', 'year', 'Citation counts (CR)', 'Citation counts (OA)',
                       'Annual cit counts (CR)', 'Annual cit counts (OA)', 'references_count', 'count']
        
        for col in numeric_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Обработка даты
        if 'publication_date' in df_processed.columns:
            df_processed['publication_date'] = pd.to_datetime(df_processed['publication_date'], errors='coerce')
            # Извлекаем год из даты, если year отсутствует
            if 'year' not in df_processed.columns or df_processed['year'].isnull().all():
                df_processed['year'] = df_processed['publication_date'].dt.year
        
        # Обработка списковых колонок
        list_columns = {
            'countries {country 1; ... country last}': 'countries_list',
            'Concepts': 'concepts_list',
            'authors': 'authors_list',
            'affiliations {aff 1; aff 2... aff last}': 'affiliations_list'
        }
        
        for source_col, target_col in list_columns.items():
            if source_col in df_processed.columns:
                df_processed[target_col] = df_processed[source_col].fillna('').apply(
                    lambda x: [item.strip() for item in str(x).split(';') if item.strip()]
                )
        
        # Расчет дополнительных метрик
        current_year = datetime.now().year
        if 'year' in df_processed.columns:
            df_processed['article_age'] = current_year - df_processed['year']
            df_processed['article_age'] = df_processed['article_age'].clip(lower=1)
            
            # Нормированное внимание
            if 'count' in df_processed.columns:
                df_processed['normalized_attention'] = df_processed['count'] / df_processed['article_age']
        
        # Расчет максимальных цитирований между CR и OA (везде используем max_citations)
        if 'Citation counts (CR)' in df_processed.columns and 'Citation counts (OA)' in df_processed.columns:
            df_processed['max_citations'] = df_processed[['Citation counts (CR)', 'Citation counts (OA)']].max(axis=1)
            df_processed['max_annual_citations'] = df_processed[['Annual cit counts (CR)', 'Annual cit counts (OA)']].max(axis=1)
        elif 'Citation counts (CR)' in df_processed.columns:
            df_processed['max_citations'] = df_processed['Citation counts (CR)']
            df_processed['max_annual_citations'] = df_processed['Annual cit counts (CR)']
        elif 'Citation counts (OA)' in df_processed.columns:
            df_processed['max_citations'] = df_processed['Citation counts (OA)']
            df_processed['max_annual_citations'] = df_processed['Annual cit counts (OA)']
        
        # Количество стран и аффилиаций
        if 'countries_list' in df_processed.columns:
            df_processed['num_countries'] = df_processed['countries_list'].apply(len)
        
        if 'affiliations_list' in df_processed.columns:
            df_processed['num_affiliations'] = df_processed['affiliations_list'].apply(len)
        
        st.success("✅ Data preprocessing complete")
        return df_processed
    
    # ============================================================================
    # ФУНКЦИИ ДЛЯ ПОСТРОЕНИЯ ГРАФИКОВ (35 ВИДОВ)
    # ============================================================================
    
    # ==================== ГРАФИК 1: РАСПРЕДЕЛЕНИЕ ВНИМАНИЯ ====================
    def plot_1_distribution_attention(self):
        """1. Распределение внимания (лог-лог, CCDF, Лоренц)"""
        try:
            if 'count' not in self.df_processed.columns:
                return None
            
            counts = self.df_processed['count'].dropna().values
            counts = counts[counts > 0]
            
            if len(counts) < 10:
                self.log_warning("Insufficient data for distribution plot")
                return None
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle('Distribution of Research Attention', fontweight='bold', fontsize=16)
            
            # A: Log-log гистограмма
            axes[0].hist(counts, bins=np.logspace(np.log10(1), np.log10(max(100, counts.max())), 30),
                        edgecolor='black', alpha=0.7, color='#2E86AB')
            axes[0].set_xscale('log')
            axes[0].set_yscale('log')
            axes[0].set_xlabel('Number of Mentions', fontweight='bold')
            axes[0].set_ylabel('Frequency', fontweight='bold')
            axes[0].set_title('A. Log-Log Distribution', fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # Сохраняем данные
            self.plot_data['1_distribution'] = {
                'counts': counts.tolist(),
                'log_bins': np.logspace(np.log10(1), np.log10(max(100, counts.max())), 30).tolist()
            }
            
            # B: CCDF
            sorted_counts = np.sort(counts)
            ccdf = 1 - np.arange(len(sorted_counts)) / len(sorted_counts)
            
            axes[1].loglog(sorted_counts, ccdf, 'o-', markersize=2, linewidth=1.5, color='#C73E1D')
            axes[1].set_xlabel('Number of Mentions', fontweight='bold')
            axes[1].set_ylabel('CCDF (P(X ≥ x))', fontweight='bold')
            axes[1].set_title('B. Complementary CDF', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            # C: Кривая Лоренца
            sorted_counts = np.sort(counts)
            cumulative_counts = np.cumsum(sorted_counts)
            cumulative_percent = cumulative_counts / cumulative_counts[-1]
            population_percent = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
            
            axes[2].plot(population_percent, cumulative_percent, linewidth=2.5,
                        color='#6B8E23', label='Lorenz curve')
            axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5, label='Perfect equality')
            axes[2].fill_between(population_percent, 0, cumulative_percent, alpha=0.2, color='#6B8E23')
            
            # Расчет индекса Джини
            gini = 0
            for i in range(1, len(population_percent)):
                gini += (population_percent[i] - population_percent[i-1]) * \
                       (cumulative_percent[i] + cumulative_percent[i-1])
            gini = 1 - gini
            
            axes[2].text(0.7, 0.3, f'Gini Index = {gini:.3f}',
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
            
            axes[2].set_xlabel('Cumulative Proportion of Papers', fontweight='bold')
            axes[2].set_ylabel('Cumulative Proportion of Mentions', fontweight='bold')
            axes[2].set_title(f'C. Lorenz Curve', fontweight='bold')
            axes[2].legend(loc='upper left')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_1_distribution_attention: {str(e)}")
            return None
    
    # ==================== ГРАФИК 2: ХОРДОВАЯ ДИАГРАММА СТРАН (УЛУЧШЕННАЯ) ====================
    def plot_2_country_chord_diagram(self):
        """2. Круговая хордовая диаграмма коллабораций между странами (улучшенная: толщина хорд пропорциональна весу, текст снаружи)"""
        try:
            if 'countries_list' not in self.df_processed.columns:
                return None
            
            # Собираем данные о коллаборациях
            country_pairs = []
            country_weights = defaultdict(float)
            
            for idx, row in self.df_processed.iterrows():
                if isinstance(row['countries_list'], list) and len(row['countries_list']) >= 2:
                    countries = [c.strip().upper() for c in row['countries_list']]
                    weight = row.get('count', 1)
                    
                    for country in countries:
                        country_weights[country] += weight
                    
                    for i in range(len(countries)):
                        for j in range(i+1, len(countries)):
                            pair = tuple(sorted([countries[i], countries[j]]))
                            country_pairs.append({
                                'country1': countries[i],
                                'country2': countries[j],
                                'weight': weight
                            })
            
            if len(country_weights) < 3:
                self.log_warning("Insufficient data for country chord diagram")
                return None
            
            # Выбираем топ N стран
            top_countries = sorted(country_weights.items(), key=lambda x: x[1], reverse=True)[:self.top_countries_chord]
            top_country_names = [c[0] for c in top_countries]
            
            # Создаем матрицу связей
            n = len(top_country_names)
            country_to_idx = {name: i for i, name in enumerate(top_country_names)}
            adjacency_matrix = np.zeros((n, n))
            
            for pair_data in country_pairs:
                if pair_data['country1'] in country_to_idx and pair_data['country2'] in country_to_idx:
                    i = country_to_idx[pair_data['country1']]
                    j = country_to_idx[pair_data['country2']]
                    adjacency_matrix[i, j] += pair_data['weight']
                    adjacency_matrix[j, i] += pair_data['weight']
            
            # Создаем цветовую схему
            colors = []
            for i in range(n):
                hue = i / n
                rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                color_hex = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
                colors.append(color_hex)
            
            # Расставляем узлы по кругу
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            radius = 1.0
            x_positions = radius * np.cos(angles)
            y_positions = radius * np.sin(angles)
            
            # Позиции для текста снаружи (на увеличенном радиусе)
            text_radius = 1.25
            text_x = text_radius * np.cos(angles)
            text_y = text_radius * np.sin(angles)
            
            # Создаем фигуру
            fig = go.Figure()
            
            # Добавляем внешние метки стран (снаружи от хорд)
            fig.add_trace(go.Scatter(
                x=text_x,
                y=text_y,
                mode='text',
                text=top_country_names,
                textposition='middle center',
                textfont=dict(size=11, color='black', family='Arial, sans-serif', weight='bold'),
                hovertext=[f"<b>{name}</b><br>Total weight: {weight:.1f}" for name, weight in top_countries],
                hoverinfo='text',
                showlegend=False
            ))
            
            # Добавляем узлы (точки на круге)
            fig.add_trace(go.Scatter(
                x=x_positions,
                y=y_positions,
                mode='markers',
                marker=dict(
                    size=25,
                    color=colors,
                    line=dict(color='black', width=1.5),
                    symbol='circle'
                ),
                hovertext=[f"<b>{name}</b><br>Total weight: {weight:.1f}" for name, weight in top_countries],
                hoverinfo='text',
                showlegend=False
            ))
            
            # Добавляем хорды (связи) с толщиной, пропорциональной весу
            max_weight = max(adjacency_matrix.flatten()) if adjacency_matrix.size > 0 else 1
            
            for i in range(n):
                for j in range(i+1, n):
                    weight = adjacency_matrix[i, j]
                    if weight > 0:
                        # Создаем плавную кривую между точками i и j
                        t = np.linspace(0, 1, 100)
                        
                        # Безье кривая между двумя точками на окружности
                        p0 = np.array([x_positions[i], y_positions[i]])
                        p3 = np.array([x_positions[j], y_positions[j]])
                        
                        # Контрольные точки для изгиба наружу
                        mid_angle = (angles[i] + angles[j]) / 2
                        if abs(angles[i] - angles[j]) > np.pi:
                            mid_angle += np.pi
                        control_offset = 0.4 * (1 + weight / max_weight)
                        ctrl_point = np.array([control_offset * np.cos(mid_angle), control_offset * np.sin(mid_angle)])
                        
                        # Кривая Безье 3-го порядка
                        curve_x = (1-t)**3 * p0[0] + 3*(1-t)**2*t * ctrl_point[0] + 3*(1-t)*t**2 * ctrl_point[0] + t**3 * p3[0]
                        curve_y = (1-t)**3 * p0[1] + 3*(1-t)**2*t * ctrl_point[1] + 3*(1-t)*t**2 * ctrl_point[1] + t**3 * p3[1]
                        
                        # Градиентный цвет хорды
                        start_rgb = [int(colors[i][1:3], 16), int(colors[i][3:5], 16), int(colors[i][5:7], 16)]
                        end_rgb = [int(colors[j][1:3], 16), int(colors[j][3:5], 16), int(colors[j][5:7], 16)]
                        
                        # Создаем одну линию с градиентом (используем один цвет для всей хорды)
                        mixed_rgb = [(start_rgb[k] + end_rgb[k]) // 2 for k in range(3)]
                        chord_color = f'rgba({mixed_rgb[0]}, {mixed_rgb[1]}, {mixed_rgb[2]}, 0.8)'
                        
                        # Толщина линии пропорциональна весу (нормирована)
                        line_width = 3 + 15 * weight / max_weight
                        
                        fig.add_trace(go.Scatter(
                            x=curve_x,
                            y=curve_y,
                            mode='lines',
                            line=dict(width=line_width, color=chord_color),
                            hoverinfo='none',
                            showlegend=False
                        ))
            
            # Добавляем внутренний круг
            theta = np.linspace(0, 2*np.pi, 100)
            inner_radius = 0.85
            fig.add_trace(go.Scatter(
                x=inner_radius * np.cos(theta),
                y=inner_radius * np.sin(theta),
                mode='lines',
                line=dict(color='white', width=2),
                fill='toself',
                fillcolor='rgba(240, 240, 240, 0.3)',
                hoverinfo='none',
                showlegend=False
            ))
            
            # Добавляем внешнюю окружность
            fig.add_trace(go.Scatter(
                x=radius * np.cos(theta),
                y=radius * np.sin(theta),
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                hoverinfo='none',
                showlegend=False
            ))
            
            fig.update_layout(
                title=dict(
                    text=f"Country Collaboration Chord Diagram (Top {self.top_countries_chord} Countries)",
                    font=dict(size=16, weight='bold')
                ),
                width=1000,
                height=1000,
                xaxis=dict(
                    visible=False,
                    range=[-1.5, 1.5],
                    scaleanchor="y",
                    scaleratio=1
                ),
                yaxis=dict(
                    visible=False,
                    range=[-1.5, 1.5]
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False,
                hovermode='closest'
            )
            
            # Сохраняем данные
            self.plot_data['2_country_chord'] = {
                'countries': top_country_names,
                'adjacency_matrix': adjacency_matrix.tolist(),
                'total_collaborations': sum(adjacency_matrix.flatten()) / 2,
                'coordinates': {
                    'x': x_positions.tolist(),
                    'y': y_positions.tolist()
                }
            }
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_2_country_chord_diagram: {str(e)}")
            return None
    
    # ==================== ГРАФИК 3: МЕЖДУНАРОДНОСТЬ VS ЦИТИРОВАНИЯ (ЛИНЕЙНАЯ) ====================
    def plot_3_internationality_vs_citations_linear(self):
        """3. Международность vs Цитируемость (линейная шкала)"""
        try:
            required_cols = ['num_countries', 'max_citations', 'author count']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            valid_data = valid_data[valid_data['max_citations'] >= 0]
            if len(valid_data) < 10:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            scatter = ax.scatter(valid_data['num_countries'],
                               valid_data['max_citations'],
                               c=valid_data.get('max_annual_citations', 1),
                               s=valid_data['author count'] * 20,
                               alpha=0.7,
                               cmap='viridis',
                               edgecolors='black',
                               linewidth=0.5)
            
            # Линейная регрессия (если включена)
            if self.show_regression_trends and len(valid_data) > 10:
                x = valid_data['num_countries'].values
                y = valid_data['max_citations'].values
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() > 10:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                    y_line = intercept + slope * x_line
                    ax.plot(x_line, y_line, 'r--', linewidth=2, 
                           label=f'r = {r_value:.3f}, p = {p_value:.3f}')
                    ax.legend(loc='upper left')
            
            ax.set_xlabel('Number of Collaborating Countries', fontweight='bold')
            ax.set_ylabel('Max Citations (max(CR, OA))', fontweight='bold')
            ax.set_title('International Collaboration vs Citation Impact (Linear Scale)',
                        fontweight='bold', fontsize=16)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Annual Citation Rate (max(CR, OA))', fontweight='bold')
            
            # Легенда для размера точек
            sizes = [5, 10, 15]
            labels = ['5 authors', '10 authors', '15 authors']
            legend_elements = []
            for size, label in zip(sizes, labels):
                legend_elements.append(plt.scatter([], [], s=size*20, c='gray',
                                                  alpha=0.7, edgecolors='black',
                                                  label=label))
            
            ax.legend(handles=legend_elements, loc='upper left', title='Team Size')
            ax.grid(True, alpha=0.3)
            
            # Сохраняем данные
            self.plot_data['3_internationality_vs_citations_linear'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_3_internationality_vs_citations_linear: {str(e)}")
            return None
    
    # ==================== ГРАФИК 4: МЕЖДУНАРОДНОСТЬ VS ЦИТИРОВАНИЯ (ЛОГАРИФМИЧЕСКАЯ) ====================
    def plot_4_internationality_vs_citations_log(self):
        """4. Международность vs Цитируемость (логарифмическая шкала для Y)"""
        try:
            required_cols = ['num_countries', 'max_citations', 'author count']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            valid_data = valid_data[valid_data['max_citations'] > 0]
            if len(valid_data) < 10:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            scatter = ax.scatter(valid_data['num_countries'],
                               valid_data['max_citations'],
                               c=valid_data.get('max_annual_citations', 1),
                               s=valid_data['author count'] * 20,
                               alpha=0.7,
                               cmap='viridis',
                               edgecolors='black',
                               linewidth=0.5)
            
            # Экспоненциальная регрессия (если включена)
            if self.show_regression_trends and len(valid_data) > 10:
                x = valid_data['num_countries'].values
                log_y = np.log(valid_data['max_citations'].values)
                mask = np.isfinite(log_y)
                if mask.sum() > 10:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], log_y[mask])
                    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                    y_line = np.exp(intercept + slope * x_line)
                    ax.plot(x_line, y_line, 'r--', linewidth=2,
                           label=f'exponential: y ∝ exp({slope:.3f}x), r = {r_value:.3f}')
                    ax.legend(loc='upper left')
            
            ax.set_xlabel('Number of Collaborating Countries', fontweight='bold')
            ax.set_ylabel('Max Citations (max(CR, OA)) - Log Scale', fontweight='bold')
            ax.set_title('International Collaboration vs Citation Impact (Log Y Scale)',
                        fontweight='bold', fontsize=16)
            
            ax.set_yscale('log')
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Annual Citation Rate (max(CR, OA))', fontweight='bold')
            
            # Легенда для размера точек
            sizes = [5, 10, 15]
            labels = ['5 authors', '10 authors', '15 authors']
            legend_elements = []
            for size, label in zip(sizes, labels):
                legend_elements.append(plt.scatter([], [], s=size*20, c='gray',
                                                  alpha=0.7, edgecolors='black',
                                                  label=label))
            
            ax.legend(handles=legend_elements, loc='upper left', title='Team Size')
            ax.grid(True, alpha=0.3, which='both')
            
            # Сохраняем данные
            self.plot_data['4_internationality_vs_citations_log'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_4_internationality_vs_citations_log: {str(e)}")
            return None
    
    # ==================== ГРАФИК 5: ТЕПЛОВАЯ КАРТА ЖУРНАЛОВ ====================
    def plot_5_journal_year_heatmap(self, top_journals=15):
        """5. Тепловая карта: Журнал vs Год (с использованием max_annual_citations)"""
        try:
            required_cols = ['Full journal Name', 'year', 'max_annual_citations']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            # Выбираем топ журналов
            journal_counts = self.df_processed['Full journal Name'].value_counts()
            top_journals_list = journal_counts.head(top_journals).index.tolist()
            
            heatmap_data = self.df_processed[self.df_processed['Full journal Name'].isin(top_journals_list)].copy()
            if len(heatmap_data) == 0:
                return None
            
            pivot_table = heatmap_data.pivot_table(
                values='max_annual_citations',
                index='year',
                columns='Full journal Name',
                aggfunc='mean',
                fill_value=0
            ).sort_index()
            
            # Фильтруем годы, где все значения = 0
            row_sums = pivot_table.sum(axis=1)
            pivot_table = pivot_table[row_sums > 0]
            
            if len(pivot_table) < 2:
                self.log_warning("Insufficient years with data for heatmap")
                return None
            
            # Сохраняем данные
            self.plot_data['5_journal_year_heatmap'] = {
                'pivot_data': pivot_table.to_dict(),
                'years': pivot_table.index.tolist(),
                'journals': pivot_table.columns.tolist()
            }
            
            fig, ax = plt.subplots(figsize=(16, 10))
            
            im = ax.imshow(pivot_table.values, cmap='Blues', aspect='auto', vmin=0)
            
            ax.set_xticks(np.arange(len(pivot_table.columns)))
            ax.set_yticks(np.arange(len(pivot_table.index)))
            ax.set_xticklabels(pivot_table.columns, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(pivot_table.index.astype(int), fontsize=10)
            
            # Добавляем значения
            for i in range(len(pivot_table.index)):
                for j in range(len(pivot_table.columns)):
                    value = pivot_table.iloc[i, j]
                    if value > 0:
                        ax.text(j, i, f'{value:.1f}',
                              ha="center", va="center",
                              color="white" if value > pivot_table.values.max()/2 else "black",
                              fontsize=8, fontweight='bold')
            
            ax.set_xlabel('Journal', fontweight='bold')
            ax.set_ylabel('Publication Year', fontweight='bold')
            ax.set_title(f'Average Annual Citation Rate by Journal and Year (Top {top_journals} Journals)',
                        fontweight='bold', fontsize=16)
            
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Average Annual Citations (max(CR, OA))', rotation=90, fontsize=12)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_5_journal_year_heatmap: {str(e)}")
            return None
    
    # ==================== ГРАФИК 6: КОЛЛАБОРАЦИИ VS ЦИТИРОВАНИЯ (ЛИНЕЙНАЯ) ====================
    def plot_6_collaboration_vs_citations_linear(self):
        """6. Зависимость цитирований от коллабораций (линейная шкала)"""
        try:
            required_cols = ['author count', 'num_affiliations', 'num_countries', 'max_citations']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            valid_data = valid_data[valid_data['max_citations'] >= 0]
            if len(valid_data) < 10:
                return None
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Collaboration Scale vs Citation Impact (Linear Scale)', 
                        fontweight='bold', fontsize=16)
            
            metrics = [
                ('author count', 'Number of Authors', axes[0]),
                ('num_affiliations', 'Number of Affiliations', axes[1]),
                ('num_countries', 'Number of Countries', axes[2])
            ]
            
            for idx, (metric, label, ax) in enumerate(metrics):
                scatter = ax.scatter(valid_data[metric],
                                   valid_data['max_citations'],
                                   c=valid_data['num_countries'] if metric != 'num_countries' else valid_data['author count'],
                                   s=valid_data['author count'] * 10,
                                   alpha=0.6,
                                   cmap='viridis',
                                   edgecolors='black',
                                   linewidth=0.5)
                
                # Линейная регрессия (если включена)
                if self.show_regression_trends and len(valid_data) > 10:
                    x = valid_data[metric].values
                    y = valid_data['max_citations'].values
                    mask = ~(np.isnan(x) | np.isnan(y))
                    if mask.sum() > 10:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                        x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                        y_line = intercept + slope * x_line
                        ax.plot(x_line, y_line, 'r--', linewidth=2, 
                               label=f'r = {r_value:.3f}, p = {p_value:.3f}')
                        ax.legend(loc='upper left', fontsize=8)
                
                ax.set_xlabel(label, fontweight='bold')
                ax.set_ylabel('Max Citations (max(CR, OA))', fontweight='bold')
                ax.set_title(f'{label} vs Citations (Linear)', fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                if idx < 2:
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar_label = 'Number of Countries' if metric != 'num_countries' else 'Number of Authors'
                    cbar.set_label(cbar_label, fontweight='bold')
            
            # Сохраняем данные
            self.plot_data['6_collaboration_vs_citations_linear'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_6_collaboration_vs_citations_linear: {str(e)}")
            return None
    
    # ==================== ГРАФИК 7: КОЛЛАБОРАЦИИ VS ЦИТИРОВАНИЯ (ЛОГАРИФМИЧЕСКАЯ) ====================
    def plot_7_collaboration_vs_citations_log(self):
        """7. Зависимость цитирований от коллабораций (логарифмическая шкала для Y)"""
        try:
            required_cols = ['author count', 'num_affiliations', 'num_countries', 'max_citations']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            valid_data = valid_data[valid_data['max_citations'] > 0]
            if len(valid_data) < 10:
                return None
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Collaboration Scale vs Citation Impact (Log Y Scale)', 
                        fontweight='bold', fontsize=16)
            
            metrics = [
                ('author count', 'Number of Authors', axes[0]),
                ('num_affiliations', 'Number of Affiliations', axes[1]),
                ('num_countries', 'Number of Countries', axes[2])
            ]
            
            for idx, (metric, label, ax) in enumerate(metrics):
                plot_data = valid_data.copy()
                
                scatter = ax.scatter(plot_data[metric],
                                   plot_data['max_citations'],
                                   c=plot_data['num_countries'] if metric != 'num_countries' else plot_data['author count'],
                                   s=plot_data['author count'] * 10,
                                   alpha=0.6,
                                   cmap='viridis',
                                   edgecolors='black',
                                   linewidth=0.5)
                
                # Экспоненциальная регрессия (если включена)
                if self.show_regression_trends and len(plot_data) > 10:
                    x = plot_data[metric].values
                    log_y = np.log(plot_data['max_citations'].values)
                    mask = np.isfinite(log_y)
                    if mask.sum() > 10:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], log_y[mask])
                        x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                        y_line = np.exp(intercept + slope * x_line)
                        ax.plot(x_line, y_line, 'r--', linewidth=2, 
                               label=f'exponential: y ∝ exp({slope:.3f}x), r = {r_value:.3f}')
                        ax.legend(loc='upper left', fontsize=8)
                
                ax.set_xlabel(label, fontweight='bold')
                ax.set_ylabel('Max Citations (max(CR, OA)) - Log Scale', fontweight='bold')
                ax.set_title(f'{label} vs Citations (Log Y Scale)', fontweight='bold')
                
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3, which='both')
                
                cbar = plt.colorbar(scatter, ax=ax)
                if metric != 'num_countries':
                    cbar.set_label('Number of Countries', fontweight='bold', fontsize=10)
                else:
                    cbar.set_label('Number of Authors', fontweight='bold', fontsize=10)
            
            # Сохраняем данные
            self.plot_data['7_collaboration_vs_citations_log'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_7_collaboration_vs_citations_log: {str(e)}")
            return None
    
    # ==================== ГРАФИК 8: REFERENCES VS CITATIONS (ЛИНЕЙНАЯ) ====================
    def plot_8_references_vs_citations_linear(self):
        """8. Пузырьковая диаграмма: References vs Citations (линейная шкала)"""
        try:
            required_cols = ['references_count', 'max_citations', 'author count', 'num_countries']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            valid_data = valid_data[valid_data['max_citations'] >= 0]
            if len(valid_data) < 10:
                return None
            
            fig, ax = plt.subplots(figsize=(14, 10))
            
            scatter = ax.scatter(valid_data['references_count'],
                               valid_data['max_citations'],
                               s=valid_data['author count'] * 40,
                               c=valid_data['num_countries'],
                               cmap='coolwarm',
                               alpha=0.7,
                               edgecolors='black',
                               linewidth=0.5)
            
            # Линейная регрессия (если включена)
            if self.show_regression_trends and len(valid_data) > 10:
                x = valid_data['references_count'].values
                y = valid_data['max_citations'].values
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() > 10:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                    y_line = intercept + slope * x_line
                    ax.plot(x_line, y_line, 'r--', linewidth=2,
                           label=f'r = {r_value:.3f}, p = {p_value:.3f}')
                    ax.legend(loc='upper left')
            
            ax.set_xlabel('Number of References', fontweight='bold')
            ax.set_ylabel('Max Citations (max(CR, OA)) - Linear Scale', fontweight='bold')
            ax.set_title('Research Breadth vs Impact (Linear Scale)',
                        fontweight='bold', fontsize=16)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Number of Collaborating Countries', fontweight='bold')
            
            # Легенда для размеров
            sizes = [5, 10, 15]
            labels = ['5 authors', '10 authors', '15 authors']
            legend_elements = [plt.scatter([], [], s=size*40, c='gray', alpha=0.7,
                                          edgecolors='black', label=label)
                              for size, label in zip(sizes, labels)]
            
            ax.legend(handles=legend_elements, title='Team Size', loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Сохраняем данные
            self.plot_data['8_references_vs_citations_linear'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_8_references_vs_citations_linear: {str(e)}")
            return None
    
    # ==================== ГРАФИК 9: REFERENCES VS CITATIONS (ЛОГАРИФМИЧЕСКАЯ) ====================
    def plot_9_references_vs_citations_log(self):
        """9. Пузырьковая диаграмма: References vs Citations (логарифмическая шкала для Y)"""
        try:
            required_cols = ['references_count', 'max_citations', 'author count', 'num_countries']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            valid_data = valid_data[valid_data['max_citations'] > 0]
            if len(valid_data) < 10:
                return None
            
            fig, ax = plt.subplots(figsize=(14, 10))
            
            scatter = ax.scatter(valid_data['references_count'],
                               valid_data['max_citations'],
                               s=valid_data['author count'] * 40,
                               c=valid_data['num_countries'],
                               cmap='coolwarm',
                               alpha=0.7,
                               edgecolors='black',
                               linewidth=0.5)
            
            ax.set_xlabel('Number of References', fontweight='bold')
            ax.set_ylabel('Max Citations (max(CR, OA)) - Log Scale', fontweight='bold')
            ax.set_title('Research Breadth vs Impact (Logarithmic Scale)',
                        fontweight='bold', fontsize=16)
            
            ax.set_yscale('log')
            
            # Экспоненциальная регрессия (если включена)
            if self.show_regression_trends and len(valid_data) > 10:
                x = valid_data['references_count'].values
                log_y = np.log(valid_data['max_citations'].values)
                mask = np.isfinite(log_y)
                if mask.sum() > 10:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], log_y[mask])
                    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                    y_line = np.exp(intercept + slope * x_line)
                    ax.plot(x_line, y_line, 'r--', linewidth=2,
                           label=f'exponential: y ∝ exp({slope:.3f}x), r = {r_value:.3f}')
                    ax.legend(loc='upper left')
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Number of Collaborating Countries', fontweight='bold')
            
            # Легенда для размеров
            sizes = [5, 10, 15]
            labels = ['5 authors', '10 authors', '15 authors']
            legend_elements = [plt.scatter([], [], s=size*40, c='gray', alpha=0.7,
                                          edgecolors='black', label=label)
                              for size, label in zip(sizes, labels)]
            
            ax.legend(handles=legend_elements, title='Team Size', loc='upper left')
            ax.grid(True, alpha=0.3, which='both')
            
            # Сохраняем данные
            self.plot_data['9_references_vs_citations_log'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_9_references_vs_citations_log: {str(e)}")
            return None
    
    # ==================== ГРАФИК 10: АНАЛИЗ КОНЦЕПТОВ ====================
    def plot_10_concepts_analysis(self, top_n=30):
        """10. Анализ концептов (30 топ концептов)"""
        try:
            if 'concepts_list' not in self.df_processed.columns:
                return None
            
            # Собираем все концепты
            all_concepts = []
            for concepts in self.df_processed['concepts_list']:
                if isinstance(concepts, list):
                    all_concepts.extend([c.strip() for c in concepts])
            
            if len(all_concepts) == 0:
                return None
            
            concept_counts = pd.Series(all_concepts).value_counts()
            top_concepts = concept_counts.head(top_n)
            
            # Сохраняем данные
            self.plot_data['10_concepts_analysis'] = {
                'top_concepts': top_concepts.to_dict(),
                'total_concepts': len(concept_counts)
            }
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
            
            # График 1: Bar chart
            y_pos = np.arange(len(top_concepts))
            colors = plt.cm.PuBu(np.linspace(0.3, 0.9, len(top_concepts)))
            
            bars = ax1.barh(y_pos, top_concepts.values, color=colors, edgecolor='black')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(top_concepts.index, fontsize=9)
            ax1.set_xlabel('Frequency', fontweight='bold')
            ax1.set_title(f'Top {top_n} Research Concepts', fontweight='bold')
            ax1.invert_yaxis()
            
            # Добавляем значения
            for bar in bars:
                width = bar.get_width()
                ax1.text(width * 1.01, bar.get_y() + bar.get_height()/2,
                        f'{int(width)}', va='center', fontsize=8)
            
            # График 2: Word cloud
            fig_height = fig.get_size_inches()[1]
            wordcloud_height = fig_height * 0.8
            wordcloud_width = wordcloud_height * 1.6
            
            wordcloud = WordCloud(width=int(wordcloud_width*100), 
                                height=int(wordcloud_height*100), 
                                background_color='white',
                                colormap='viridis', max_words=100).generate_from_frequencies(concept_counts.to_dict())
            
            ax2.imshow(wordcloud, interpolation='bilinear')
            ax2.axis('off')
            ax2.set_title('Concept Word Cloud', fontweight='bold')
            
            # Устанавливаем одинаковые пределы для осей Y
            ax1.set_ylim(-0.5, len(top_concepts) - 0.5)
            
            plt.suptitle('Research Concepts Analysis', fontweight='bold', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_10_concepts_analysis: {str(e)}")
            return None
    
    # ==================== ГРАФИК 11: МАТРИЦА СОВМЕСТНОЙ ВСТРЕЧАЕМОСТИ КОНЦЕПТОВ ====================
    def plot_11_concept_cooccurrence(self, top_n=15):
        """11. Матрица совместной встречаемости концептов"""
        try:
            if 'concepts_list' not in self.df_processed.columns:
                return None
            
            # Собираем топ концепты
            all_concepts = []
            for concepts in self.df_processed['concepts_list']:
                if isinstance(concepts, list):
                    all_concepts.extend([c.strip() for c in concepts])
            
            if len(all_concepts) == 0:
                return None
            
            concept_counts = pd.Series(all_concepts).value_counts()
            top_concepts = concept_counts.head(top_n).index.tolist()
            
            # Создаем матрицу совместной встречаемости
            cooccurrence = pd.DataFrame(0, index=top_concepts, columns=top_concepts)
            
            for concepts in self.df_processed['concepts_list']:
                if isinstance(concepts, list):
                    concepts_clean = [c.strip() for c in concepts if c.strip() in top_concepts]
                    for i in range(len(concepts_clean)):
                        for j in range(i+1, len(concepts_clean)):
                            c1, c2 = concepts_clean[i], concepts_clean[j]
                            cooccurrence.loc[c1, c2] += 1
                            cooccurrence.loc[c2, c1] += 1
            
            # Сохраняем данные
            self.plot_data['11_concept_cooccurrence'] = {
                'matrix': cooccurrence.to_dict(),
                'concepts': top_concepts
            }
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            im = ax.imshow(cooccurrence.values, cmap='Blues')
            
            ax.set_xticks(np.arange(len(top_concepts)))
            ax.set_yticks(np.arange(len(top_concepts)))
            ax.set_xticklabels(top_concepts, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(top_concepts, fontsize=9)
            
            # Добавляем значения
            for i in range(len(top_concepts)):
                for j in range(len(top_concepts)):
                    value = cooccurrence.iloc[i, j]
                    if value > 0:
                        ax.text(j, i, str(value),
                               ha="center", va="center",
                               color="white" if value > cooccurrence.values.max()/2 else "black",
                               fontsize=8, fontweight='bold')
            
            ax.set_title(f'Concept Co-occurrence Matrix (Top {top_n} Concepts)',
                        fontweight='bold', fontsize=16)
            
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Co-occurrence Frequency', rotation=90, fontsize=12)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_11_concept_cooccurrence: {str(e)}")
            return None
    
    # ==================== ГРАФИК 12: ВЛИЯНИЕ КЛЮЧЕВЫХ КОНЦЕПТОВ ====================
    def plot_12_concept_influence(self):
        """12. Влияние ключевых концептов (с использованием max_citations)"""
        try:
            if 'concepts_list' not in self.df_processed.columns or 'max_citations' not in self.df_processed.columns:
                return None
            
            # Разворачиваем концепты и связываем с цитированиями
            concept_citations = []
            for idx, row in self.df_processed.iterrows():
                if isinstance(row['concepts_list'], list):
                    for concept in row['concepts_list']:
                        concept_citations.append({
                            'concept': concept.strip(),
                            'max_citations': row.get('max_citations', 0),
                            'max_annual_citations': row.get('max_annual_citations', 0),
                            'year': row.get('year', 0),
                            'count': row.get('count', 0)
                        })
            
            if len(concept_citations) == 0:
                return None
            
            concept_df = pd.DataFrame(concept_citations)
            
            # Агрегируем по концептам
            concept_stats = concept_df.groupby('concept').agg({
                'max_citations': ['sum', 'mean', 'median'],
                'max_annual_citations': 'mean',
                'count': 'size'
            }).round(2)
            
            concept_stats.columns = ['total_citations', 'mean_citations', 'median_citations',
                                   'mean_annual_citations', 'num_papers']
            
            concept_stats = concept_stats[concept_stats['num_papers'] >= 2]
            concept_stats = concept_stats.sort_values('mean_citations', ascending=False).head(20)
            
            # Сохраняем данные
            self.plot_data['12_concept_influence'] = concept_stats.reset_index().to_dict('records')
            
            fig, axes = plt.subplots(1, 2, figsize=(18, 10))
            
            # График 1: Средние цитирования
            y_pos = np.arange(len(concept_stats))
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(concept_stats)))
            
            bars1 = axes[0].barh(y_pos, concept_stats['mean_citations'], color=colors, edgecolor='black')
            axes[0].set_yticks(y_pos)
            axes[0].set_yticklabels(concept_stats.index, fontsize=9)
            axes[0].set_xlabel('Mean Citations per Paper', fontweight='bold')
            axes[0].set_title('Top Concepts by Average Citation Impact', fontweight='bold')
            axes[0].invert_yaxis()
            
            for bar, (idx, row) in zip(bars1, concept_stats.iterrows()):
                width = bar.get_width()
                axes[0].text(width * 1.01, bar.get_y() + bar.get_height()/2,
                           f'n={int(row["num_papers"])}', va='center', fontsize=8)
            
            # График 2: Пузырьковая диаграмма
            scatter = axes[1].scatter(concept_stats['mean_annual_citations'],
                                    concept_stats['mean_citations'],
                                    s=concept_stats['num_papers'] * 15,
                                    c=concept_stats['total_citations'],
                                    cmap='plasma',
                                    alpha=0.7,
                                    edgecolors='black',
                                    linewidth=0.5)
            
            axes[1].set_xlabel('Mean Annual Citations', fontweight='bold')
            axes[1].set_ylabel('Mean Total Citations', fontweight='bold')
            axes[1].set_title('Concept Impact: Annual vs Total Citations', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=axes[1])
            cbar.set_label('Total Citations (Sum)', fontweight='bold')
            
            # Добавляем аннотации для топ-5
            for idx, row in concept_stats.head(5).iterrows():
                short_name = idx[:20] + '...' if len(idx) > 20 else idx
                axes[1].annotate(short_name,
                               xy=(row['mean_annual_citations'], row['mean_citations']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
            
            plt.suptitle('Concept Influence Analysis', fontweight='bold', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_12_concept_influence: {str(e)}")
            return None
    
    # ==================== ГРАФИК 13: ЭВОЛЮЦИЯ ВО ВРЕМЕНИ ====================
    def plot_13_temporal_evolution(self):
        """13. Эволюция публикационной активности и влияния во времени (с использованием max_citations)"""
        try:
            if 'year' not in self.df_processed.columns or 'max_citations' not in self.df_processed.columns:
                return None
            
            valid_data = self.df_processed.dropna(subset=['year', 'max_citations'])
            if len(valid_data) < 10:
                return None
            
            # Группируем по годам
            year_stats = valid_data.groupby('year').agg({
                'max_citations': ['sum', 'mean'],
                'max_annual_citations': 'mean',
                'doi': 'count'
            }).round(2)
            
            year_stats.columns = ['total_citations', 'mean_citations', 'mean_annual_citations', 'num_papers']
            year_stats = year_stats.sort_index()
            
            # Сохраняем данные
            self.plot_data['13_temporal_evolution'] = year_stats.reset_index().to_dict('records')
            
            fig, ax1 = plt.subplots(figsize=(14, 8))
            
            # Столбцы: число публикаций
            ax1.bar(year_stats.index, year_stats['num_papers'], 
                   alpha=0.4, color='steelblue', label='Number of Papers', edgecolor='black')
            ax1.set_xlabel('Publication Year', fontweight='bold')
            ax1.set_ylabel('Number of Papers', fontweight='bold', color='steelblue')
            ax1.tick_params(axis='y', labelcolor='steelblue')
            
            # Линия: суммарные цитирования (правая ось)
            ax2 = ax1.twinx()
            line1 = ax2.plot(year_stats.index, year_stats['total_citations'], 
                           'o-', color='darkorange', linewidth=2.5, markersize=6,
                           label='Total Citations (max)')
            ax2.set_ylabel('Total Citations (max(CR, OA))', fontweight='bold', color='darkorange')
            ax2.tick_params(axis='y', labelcolor='darkorange')
            
            # Линия: средние цитирования (дополнительно)
            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('outward', 60))
            line2 = ax3.plot(year_stats.index, year_stats['mean_citations'], 
                           's-', color='darkgreen', linewidth=2, markersize=5,
                           label='Mean Citations per Paper (max)')
            ax3.set_ylabel('Mean Citations per Paper (max(CR, OA))', fontweight='bold', color='darkgreen')
            ax3.tick_params(axis='y', labelcolor='darkgreen')
            
            # Объединяем легенды
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
            
            ax1.set_title('Temporal Evolution: Publications and Citation Impact',
                        fontweight='bold', fontsize=16)
            ax1.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_13_temporal_evolution: {str(e)}")
            return None
    
    # ==================== ГРАФИК 14: ТЕПЛОВАЯ КАРТА ВО ВРЕМЕНИ ====================
    def plot_14_temporal_heatmap(self):
        """14. Тепловая карта: Год публикации vs Возраст статьи (с использованием max_annual_citations)"""
        try:
            if 'year' not in self.df_processed.columns or 'max_annual_citations' not in self.df_processed.columns:
                return None
            
            valid_data = self.df_processed.dropna(subset=['year', 'max_annual_citations'])
            if len(valid_data) < 10:
                return None
            
            # Создаем данные для тепловой карты
            heatmap_data = []
            current_year = datetime.now().year
            
            for idx, row in valid_data.iterrows():
                pub_year = int(row['year'])
                age = current_year - pub_year
                if age > 0:
                    heatmap_data.append({
                        'pub_year': pub_year,
                        'age': age,
                        'annual_citations': row['max_annual_citations']
                    })
            
            if len(heatmap_data) == 0:
                return None
            
            heatmap_df = pd.DataFrame(heatmap_data)
            
            # Создаем сводную таблицу
            pivot_table = heatmap_df.pivot_table(
                values='annual_citations',
                index='age',
                columns='pub_year',
                aggfunc='mean',
                fill_value=0
            ).sort_index(ascending=False)
            
            # Сохраняем данные
            self.plot_data['14_temporal_heatmap'] = {
                'pivot_data': pivot_table.to_dict(),
                'years': pivot_table.columns.tolist(),
                'ages': pivot_table.index.tolist()
            }
            
            fig, ax = plt.subplots(figsize=(16, 10))
            
            im = ax.imshow(pivot_table.values, cmap='RdYlGn', aspect='auto', vmin=0)
            
            ax.set_xticks(np.arange(len(pivot_table.columns)))
            ax.set_yticks(np.arange(len(pivot_table.index)))
            ax.set_xticklabels(pivot_table.columns.astype(int), rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(pivot_table.index, fontsize=10)
            ax.set_xlabel('Publication Year', fontweight='bold')
            ax.set_ylabel('Article Age (Years)', fontweight='bold')
            ax.set_title('Annual Citation Rate by Publication Year and Article Age',
                        fontweight='bold', fontsize=16)
            
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Mean Annual Citations (max(CR, OA))', rotation=90, fontsize=12)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_14_temporal_heatmap: {str(e)}")
            return None
    
    # ==================== ГРАФИК 15: АНАЛИЗ РАЗМЕРА КОМАНДЫ ====================
    def plot_15_team_size_analysis(self):
        """15. Анализ размера команды (с использованием max_citations)"""
        try:
            if 'author count' not in self.df_processed.columns:
                return None
            
            # Категоризация размера команды
            def categorize_team_size(n):
                if pd.isna(n):
                    return 'Unknown'
                n = int(n)
                if n == 1:
                    return 'Single author'
                elif n == 2:
                    return '2 authors'
                elif n == 3:
                    return '3 authors'
                elif 4 <= n <= 5:
                    return '4-5 authors'
                elif 6 <= n <= 8:
                    return '6-8 authors'
                elif 9 <= n <= 12:
                    return '9-12 authors'
                else:
                    return '13+ authors'
            
            self.df_processed['team_size_group'] = self.df_processed['author count'].apply(categorize_team_size)
            
            # Группируем данные
            group_stats = self.df_processed.groupby('team_size_group').agg({
                'count': ['mean', 'median', 'std', 'size'],
                'max_citations': 'mean',
                'references_count': 'mean'
            }).round(2)
            
            group_stats.columns = ['mean_attention', 'median_attention', 'std_attention',
                                 'num_papers', 'mean_citations', 'mean_references']
            
            # Упорядочиваем по возрастанию числа авторов
            custom_order = ['Single author', '2 authors', '3 authors', '4-5 authors', 
                          '6-8 authors', '9-12 authors', '13+ authors', 'Unknown']
            
            # Фильтруем только существующие категории
            existing_categories = [cat for cat in custom_order if cat in group_stats.index]
            group_stats = group_stats.loc[existing_categories]
            
            # Сохраняем данные
            self.plot_data['15_team_size_analysis'] = group_stats.reset_index().to_dict('records')
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            # График 1: Распределение размеров команд
            team_size_counts = self.df_processed['team_size_group'].value_counts()
            team_size_counts = team_size_counts.reindex(existing_categories, fill_value=0)
            axes[0].bar(team_size_counts.index, team_size_counts.values,
                       alpha=0.7, color='steelblue', edgecolor='black')
            axes[0].set_xlabel('Team Size', fontweight='bold')
            axes[0].set_ylabel('Number of Papers', fontweight='bold')
            axes[0].set_title('Distribution of Team Sizes', fontweight='bold')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # График 2: Среднее внимание по размеру команды
            axes[1].bar(group_stats.index, group_stats['mean_attention'],
                       alpha=0.7, color='darkorange', edgecolor='black')
            axes[1].errorbar(group_stats.index, group_stats['mean_attention'],
                           yerr=group_stats['std_attention'],
                           fmt='none', color='black', capsize=5)
            axes[1].set_xlabel('Team Size', fontweight='bold')
            axes[1].set_ylabel('Mean Attention', fontweight='bold')
            axes[1].set_title('Mean Attention by Team Size', fontweight='bold')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # График 3: Средние цитирования
            axes[2].bar(group_stats.index, group_stats['mean_citations'],
                       alpha=0.7, color='darkgreen', edgecolor='black')
            axes[2].set_xlabel('Team Size', fontweight='bold')
            axes[2].set_ylabel('Mean Citations (max(CR, OA))', fontweight='bold')
            axes[2].set_title('Mean Citations by Team Size', fontweight='bold')
            axes[2].tick_params(axis='x', rotation=45)
            axes[2].grid(True, alpha=0.3, axis='y')
            
            # График 4: Средние ссылки
            axes[3].bar(group_stats.index, group_stats['mean_references'],
                       alpha=0.7, color='darkred', edgecolor='black')
            axes[3].set_xlabel('Team Size', fontweight='bold')
            axes[3].set_ylabel('Mean References', fontweight='bold')
            axes[3].set_title('Mean References by Team Size', fontweight='bold')
            axes[3].tick_params(axis='x', rotation=45)
            axes[3].grid(True, alpha=0.3, axis='y')
            
            plt.suptitle('Team Size Analysis', fontweight='bold', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_15_team_size_analysis: {str(e)}")
            return None
    
    # ==================== ГРАФИК 16: КОРРЕЛЯЦИОННАЯ МАТРИЦА ====================
    def plot_16_correlation_matrix(self):
        """16. Корреляционная матрица с выделением ключевых параметров (с использованием max_citations)"""
        try:
            numeric_cols = ['author count', 'references_count',
                          'max_citations', 'max_annual_citations',
                          'count', 'num_countries', 'num_affiliations',
                          'article_age', 'normalized_attention']
            
            available_cols = [col for col in numeric_cols if col in self.df_processed.columns]
            
            if len(available_cols) < 3:
                return None
            
            correlation_data = self.df_processed[available_cols].dropna()
            
            if len(correlation_data) < 10:
                return None
            
            corr_matrix = correlation_data.corr(method='spearman')
            
            # Переупорядочиваем матрицу: ключевые параметры сначала
            key_params = ['count', 'max_citations', 'max_annual_citations']
            
            # Фильтруем только те, что есть в данных
            existing_key_params = [p for p in key_params if p in corr_matrix.columns]
            other_params = [p for p in corr_matrix.columns if p not in existing_key_params]
            
            # Новый порядок
            new_order = existing_key_params + other_params
            corr_matrix = corr_matrix.reindex(index=new_order, columns=new_order)
            
            # Сохраняем данные
            self.plot_data['16_correlation_matrix'] = {
                'correlation_matrix': corr_matrix.to_dict(),
                'columns': available_cols,
                'method': 'spearman',
                'key_parameters': existing_key_params
            }
            
            fig, ax = plt.subplots(figsize=(14, 12))
            
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Рисуем тепловую карту
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                       cmap='coolwarm', center=0, square=True,
                       linewidths=0.5, cbar_kws={'shrink': 0.8},
                       ax=ax, annot_kws={'fontsize': 9})
            
            # Добавляем выделение для ключевых параметров
            key_param_indices = [i for i, col in enumerate(corr_matrix.columns) if col in existing_key_params]
            for idx in key_param_indices:
                # Выделяем строки
                ax.add_patch(plt.Rectangle((0, idx), len(corr_matrix), 1, 
                                          fill=False, edgecolor='red', linewidth=2, 
                                          alpha=0.7))
                # Выделяем столбцы
                ax.add_patch(plt.Rectangle((idx, 0), 1, len(corr_matrix), 
                                          fill=False, edgecolor='red', linewidth=2, 
                                          alpha=0.7))
            
            # Добавляем легенду для выделения
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='white', edgecolor='red', linewidth=2,
                                   alpha=0.7, label='Key parameters (Count & Citations)')]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            ax.set_title('Correlation Matrix of Research Metrics (Spearman)\nKey Parameters Highlighted in Red', 
                        fontweight='bold', fontsize=16)
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_16_correlation_matrix: {str(e)}")
            return None
    
    # ==================== ГРАФИК 17: СРАВНЕНИЕ CR VS OA ====================
    def plot_17_citation_sources_comparison(self):
        """17. Сравнение CR vs OA цитирований (оставляем как есть для сравнения источников)"""
        try:
            required_cols = ['Citation counts (CR)', 'Citation counts (OA)']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) == 0:
                return None
            
            # Рассчитываем разницу
            valid_data['citation_diff'] = valid_data['Citation counts (OA)'] - valid_data['Citation counts (CR)']
            valid_data['citation_ratio'] = valid_data['Citation counts (OA)'] / valid_data['Citation counts (CR)'].replace(0, 1)
            
            # Сохраняем данные
            self.plot_data['17_citation_sources_comparison'] = {
                'summary': {
                    'mean_diff': float(valid_data['citation_diff'].mean()),
                    'mean_ratio': float(valid_data['citation_ratio'].mean()),
                    'oa_greater': int((valid_data['citation_diff'] > 0).sum()),
                    'cr_greater': int((valid_data['citation_diff'] < 0).sum()),
                    'equal': int((valid_data['citation_diff'] == 0).sum())
                },
                'data_sample': valid_data[required_cols + ['citation_diff', 'citation_ratio']].head(100).to_dict('records')
            }
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # График 1: Scatter plot
            max_val = max(valid_data['Citation counts (CR)'].max(),
                         valid_data['Citation counts (OA)'].max())
            
            ax1.scatter(valid_data['Citation counts (CR)'],
                       valid_data['Citation counts (OA)'],
                       alpha=0.6, c='steelblue', edgecolors='black', linewidth=0.5)
            
            # Линия y=x с регрессией (если включена)
            ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y = x')
            
            # Линейная регрессия (если включена)
            if self.show_regression_trends and len(valid_data) > 10:
                x = valid_data['Citation counts (CR)'].values
                y = valid_data['Citation counts (OA)'].values
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() > 10:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                    y_line = intercept + slope * x_line
                    ax1.plot(x_line, y_line, 'g--', linewidth=2, 
                            label=f'regression: y = {slope:.2f}x + {intercept:.1f}\nr = {r_value:.3f}')
                    ax1.legend(loc='upper left')
            
            ax1.set_xlabel('Citations from Crossref (CR)', fontweight='bold')
            ax1.set_ylabel('Citations from OpenAlex (OA)', fontweight='bold')
            ax1.set_title('Comparison of Citation Counts', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # График 2: Гистограмма разницы
            ax2.hist(valid_data['citation_diff'], bins=30,
                    alpha=0.7, color='darkorange', edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('Difference (OA - CR)', fontweight='bold')
            ax2.set_ylabel('Number of Articles', fontweight='bold')
            ax2.set_title('Distribution of Citation Differences', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Статистика
            stats_text = f"Mean difference: {valid_data['citation_diff'].mean():.1f}\n"
            stats_text += f"OA > CR: {(valid_data['citation_diff'] > 0).sum()} articles\n"
            stats_text += f"CR > OA: {(valid_data['citation_diff'] < 0).sum()} articles"
            
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.suptitle('Comparison of Citation Sources', fontweight='bold', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_17_citation_sources_comparison: {str(e)}")
            return None
    
    # ==================== ГРАФИК 18: ЦИТИРУЕМОСТЬ ПО ДОМЕНАМ ====================
    def plot_18_citation_by_domain(self):
        """18. Цитируемость по доменам науки (с использованием max_annual_citations)"""
        try:
            required_cols = ['Domain', 'max_annual_citations']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) == 0:
                return None
            
            # Агрегируем по доменам
            domain_stats = valid_data.groupby('Domain').agg({
                'max_annual_citations': ['median', 'mean', 'std', 'count'],
                'count': 'mean'
            }).round(2)
            
            domain_stats.columns = ['median_citations', 'mean_citations', 'std_citations',
                                  'num_papers', 'mean_attention']
            domain_stats = domain_stats.sort_values('median_citations', ascending=False)
            
            # Сохраняем данные
            self.plot_data['18_citation_by_domain'] = domain_stats.reset_index().to_dict('records')
            
            # Выбираем топ доменов
            top_domains = domain_stats.head(15).index.tolist()
            filtered_data = valid_data[valid_data['Domain'].isin(top_domains)]
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Boxplot
            box_data = []
            labels = []
            for domain in top_domains:
                data = filtered_data[filtered_data['Domain'] == domain]['max_annual_citations'].values
                if len(data) > 0:
                    box_data.append(data)
                    labels.append(domain)
            
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True, showfliers=False)
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Добавляем средние значения
            means = [np.mean(group) for group in box_data]
            for i, mean in enumerate(means):
                ax.scatter(i+1, mean, color='red', s=80, zorder=3,
                          edgecolors='black', marker='D')
            
            ax.set_xlabel('Research Domain', fontweight='bold')
            ax.set_ylabel('Annual Citation Rate (max(CR, OA))', fontweight='bold')
            ax.set_title('Citation Impact Distribution Across Research Domains', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_18_citation_by_domain: {str(e)}")
            return None
    
    # ==================== ГРАФИК 19: НАКОПИТЕЛЬНАЯ КРИВАЯ ВЛИЯНИЯ ====================
    def plot_19_cumulative_influence(self):
        """19. Накопительная кривая влияния"""
        try:
            if 'count' not in self.df_processed.columns:
                return None
            
            valid_data = self.df_processed.dropna(subset=['count'])
            if len(valid_data) == 0:
                return None
            
            # Сортируем по локальным цитированиям
            sorted_counts = valid_data['count'].sort_values(ascending=False).reset_index(drop=True)
            
            # Вычисляем кумулятивные суммы
            total_citations = sorted_counts.sum()
            cumulative_citations = sorted_counts.cumsum()
            cumulative_percentage = cumulative_citations / total_citations * 100
            article_percentage = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100
            
            # Сохраняем данные
            self.plot_data['19_cumulative_influence'] = {
                'sorted_counts': sorted_counts.tolist(),
                'cumulative_percentage': cumulative_percentage.tolist(),
                'article_percentage': article_percentage.tolist(),
                'total_citations': float(total_citations),
                'gini_coefficient': self._calculate_gini(sorted_counts.values)
            }
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # График 1: Накопительная кривая
            ax1.plot(article_percentage, cumulative_percentage,
                    linewidth=2.5, color='darkgreen')
            ax1.fill_between(article_percentage, 0, cumulative_percentage,
                            alpha=0.3, color='lightgreen')
            
            # Линия 20/80
            twenty_percent_idx = int(len(sorted_counts) * 0.2)
            twenty_percent_cites = cumulative_percentage.iloc[twenty_percent_idx]
            
            ax1.axvline(x=20, color='red', linestyle='--', linewidth=1.5,
                       label=f'Top 20%: {twenty_percent_cites:.1f}% of citations')
            ax1.axhline(y=80, color='blue', linestyle='--', linewidth=1.5,
                       label='80% of citations')
            
            ax1.set_xlabel('Percentage of Articles', fontweight='bold')
            ax1.set_ylabel('Percentage of Total Mentions', fontweight='bold')
            ax1.set_title('Cumulative Influence Curve (Lorenz Curve)', fontweight='bold')
            ax1.legend(loc='lower right')
            ax1.grid(True, alpha=0.3)
            
            # График 2: Распределение
            log_bins = np.logspace(0, np.log10(sorted_counts.max() + 1), 20)
            ax2.hist(sorted_counts, bins=log_bins, alpha=0.7,
                    color='steelblue', edgecolor='black')
            ax2.set_xscale('log')
            ax2.set_xlabel('Number of Local Citations (log scale)', fontweight='bold')
            ax2.set_ylabel('Number of Articles', fontweight='bold')
            ax2.set_title('Distribution of Local Citation Counts', fontweight='bold')
            ax2.grid(True, alpha=0.3, which='both')
            
            # Статистика
            gini = self._calculate_gini(sorted_counts.values)
            stats_text = f"Total articles: {len(sorted_counts):,}\n"
            stats_text += f"Total mentions: {total_citations:,}\n"
            stats_text += f"Mean: {sorted_counts.mean():.2f}\n"
            stats_text += f"Median: {sorted_counts.median():.1f}\n"
            stats_text += f"Gini coefficient: {gini:.3f}"
            
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.suptitle('Analysis of Local Influence Within Dataset', fontweight='bold', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_19_cumulative_influence: {str(e)}")
            return None
    
    def _calculate_gini(self, x):
        """Расчет коэффициента Джини"""
        x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(x, dtype=float)
        if cumx[-1] == 0:
            return 0
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    
    # ==================== ГРАФИК 20: ССЫЛКИ VS ВНИМАНИЕ ====================
    def plot_20_references_vs_attention(self):
        """20. Объем ссылок vs внимание (с использованием max_citations)"""
        try:
            required_cols = ['references_count', 'count', 'max_annual_citations']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) < 10:
                return None
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # График 1: References vs Attention
            scatter1 = ax1.scatter(valid_data['references_count'],
                                 valid_data['count'],
                                 c=valid_data['max_annual_citations'],
                                 cmap='viridis', alpha=0.6, s=30,
                                 edgecolors='black', linewidth=0.5)
            
            # Линейная регрессия (если включена)
            if self.show_regression_trends and len(valid_data) > 10:
                x = valid_data['references_count'].values
                y = valid_data['count'].values
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = intercept + slope * x_line
                ax1.plot(x_line, y_line, 'r--', linewidth=2,
                        label=f'r = {r_value:.3f}, p = {p_value:.3f}')
                ax1.legend()
            
            ax1.set_xlabel('Number of References', fontweight='bold')
            ax1.set_ylabel('Local Mentions (count)', fontweight='bold')
            ax1.set_title('References vs Local Attention', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            cbar1 = plt.colorbar(scatter1, ax=ax1)
            cbar1.set_label('Annual Citations (max(CR, OA))', fontweight='bold')
            
            # График 2: References vs Citations
            scatter2 = ax2.scatter(valid_data['references_count'],
                                 valid_data['max_annual_citations'],
                                 c=valid_data['count'],
                                 cmap='plasma', alpha=0.6, s=30,
                                 edgecolors='black', linewidth=0.5)
            
            # Линейная регрессия (если включена)
            if self.show_regression_trends and len(valid_data) > 10:
                x = valid_data['references_count'].values
                y = valid_data['max_annual_citations'].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = intercept + slope * x_line
                ax2.plot(x_line, y_line, 'r--', linewidth=2,
                        label=f'r = {r_value:.3f}, p = {p_value:.3f}')
                ax2.legend()
            
            ax2.set_xlabel('Number of References', fontweight='bold')
            ax2.set_ylabel('Annual Citations (max(CR, OA))', fontweight='bold')
            ax2.set_title('References vs Citation Impact', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            cbar2 = plt.colorbar(scatter2, ax=ax2)
            cbar2.set_label('Local Mentions', fontweight='bold')
            
            plt.suptitle('Impact of Reference Count on Research Metrics', fontweight='bold', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_20_references_vs_attention: {str(e)}")
            return None
    
    # ==================== ГРАФИК 21: ВЛИЯНИЕ ЖУРНАЛОВ ====================
    def plot_21_journal_impact(self):
        """21. Влияние журналов (с использованием max_citations)"""
        try:
            required_cols = ['Full journal Name', 'count', 'max_annual_citations', 'max_citations']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) == 0:
                return None
            
            # Агрегируем по журналам
            journal_stats = valid_data.groupby('Full journal Name').agg({
                'count': ['mean', 'median', 'std', 'size'],
                'max_annual_citations': 'mean',
                'references_count': 'mean',
                'max_citations': 'mean'
            }).round(2)
            
            journal_stats.columns = ['mean_attention', 'median_attention', 'std_attention',
                                   'num_papers', 'mean_annual_citations', 'mean_references', 'mean_citations']
            
            # Фильтруем журналы с достаточным количеством статей
            journal_stats = journal_stats[journal_stats['num_papers'] >= 3]
            journal_stats = journal_stats.sort_values('mean_attention', ascending=False)
            
            # Сохраняем данные
            self.plot_data['21_journal_impact'] = journal_stats.reset_index().to_dict('records')
            
            # Выбираем топ журналов
            top_journals = journal_stats.head(15)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # График 1: Среднее внимание
            y_pos = np.arange(len(top_journals))
            colors1 = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_journals)))
            
            bars1 = ax1.barh(y_pos, top_journals['mean_attention'],
                            color=colors1, edgecolor='black', alpha=0.8)
            
            ax1.set_yticks(y_pos)
            journal_names = [name[:25] + '...' if len(name) > 25 else name
                            for name in top_journals.index]
            ax1.set_yticklabels(journal_names, fontsize=9)
            ax1.set_xlabel('Mean Attention per Paper', fontweight='bold')
            ax1.set_title('Top Journals by Attention', fontweight='bold')
            ax1.invert_yaxis()
            
            # Добавляем значения
            for bar, (_, row) in zip(bars1, top_journals.iterrows()):
                width = bar.get_width()
                info_text = f"n={int(row['num_papers'])}"
                ax1.text(width * 1.01, bar.get_y() + bar.get_height()/2,
                        info_text, va='center', fontsize=8)
            
            # График 2: Пузырьковая диаграмма
            scatter = ax2.scatter(top_journals['mean_annual_citations'],
                                top_journals['mean_attention'],
                                s=top_journals['num_papers'] * 10,
                                c=top_journals['mean_references'],
                                cmap='coolwarm', alpha=0.7,
                                edgecolors='black', linewidth=0.5)
            
            ax2.set_xlabel('Mean Annual Citations (max(CR, OA))', fontweight='bold')
            ax2.set_ylabel('Mean Attention', fontweight='bold')
            ax2.set_title('Journal Impact: Citations vs Attention', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Mean References', fontweight='bold')
            
            # Добавляем аннотации
            for idx, row in top_journals.head(5).iterrows():
                short_name = idx[:15] + '...' if len(idx) > 15 else idx
                ax2.annotate(short_name,
                            xy=(row['mean_annual_citations'], row['mean_attention']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, alpha=0.8)
            
            plt.suptitle('Journal Impact Analysis', fontweight='bold', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_21_journal_impact: {str(e)}")
            return None
    
    # ==================== ГРАФИК 22: ИЕРАРХИЧЕСКАЯ ДИАГРАММА САНКИ ====================
    def plot_22_hierarchical_sankey(self):
        """22. Иерархическая диаграмма Санки: Domain → Field → Subfield → Topic (с ограничением по полям, с использованием max_citations)"""
        try:
            required_cols = ['Domain', 'Field', 'Subfield', 'Topic', 'max_citations']
            available_cols = [col for col in required_cols if col in self.df_processed.columns]
            
            if len(available_cols) < 3:
                return None
            
            valid_data = self.df_processed.dropna(subset=available_cols)
            if len(valid_data) < 10:
                return None
            
            # Ограничиваем количество полей (Field) для читаемости
            # Сначала определяем топ поля по суммарным цитированиям
            field_citations = valid_data.groupby('Field')['max_citations'].sum().sort_values(ascending=False)
            top_fields = field_citations.head(self.top_fields_sankey).index.tolist()
            
            # Фильтруем данные только для топ полей
            filtered_data = valid_data[valid_data['Field'].isin(top_fields)]
            
            if len(filtered_data) < 5:
                self.log_warning(f"Insufficient data after filtering to top {self.top_fields_sankey} fields")
                return None
            
            # Создаем иерархические связи
            links = []
            nodes = []
            node_indices = {}
            
            def add_node(name):
                if name not in node_indices:
                    node_indices[name] = len(nodes)
                    nodes.append(name)
                return node_indices[name]
            
            # Агрегируем веса (суммарные цитирования)
            hierarchy_data = filtered_data.groupby(['Domain', 'Field', 'Subfield', 'Topic']).agg({
                'max_citations': 'sum',
                'count': 'size'
            }).reset_index()
            
            for _, row in hierarchy_data.iterrows():
                domain = str(row['Domain']) if pd.notna(row['Domain']) else 'Unknown'
                field = str(row['Field']) if pd.notna(row['Field']) else 'Unknown'
                subfield = str(row['Subfield']) if pd.notna(row['Subfield']) else 'Unknown'
                topic = str(row['Topic']) if pd.notna(row['Topic']) else 'Unknown'
                weight = row['max_citations']
                
                if weight <= 0:
                    continue
                
                # Добавляем связи
                domain_idx = add_node(domain)
                field_idx = add_node(field)
                subfield_idx = add_node(subfield)
                topic_idx = add_node(topic)
                
                links.append({'source': domain_idx, 'target': field_idx, 'value': weight})
                links.append({'source': field_idx, 'target': subfield_idx, 'value': weight})
                links.append({'source': subfield_idx, 'target': topic_idx, 'value': weight})
            
            # Сохраняем данные
            self.plot_data['22_hierarchical_sankey'] = {
                'nodes': nodes,
                'links': links,
                'total_weight': sum([l['value'] for l in links]),
                'top_fields_used': top_fields,
                'fields_limit': self.top_fields_sankey
            }
            
            # Создаем диаграмму Санки с plotly
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    label=nodes,
                    color="blue",
                    hovertemplate='%{label}<br>Value: %{value}<extra></extra>'
                ),
                link=dict(
                    source=[l['source'] for l in links],
                    target=[l['target'] for l in links],
                    value=[l['value'] for l in links]
                )
            )])
            
            fig.update_layout(
                title_text=f"Hierarchical Knowledge Structure: Domain → Field → Subfield → Topic<br>(Top {self.top_fields_sankey} Fields by Citations)",
                font_size=12,
                width=1200,
                height=800,
                font=dict(
                    family="Arial, sans-serif",
                    size=12,
                    color="black"
                )
            )
            
            # Дополнительные настройки для текста узлов
            fig.update_traces(
                textfont=dict(
                    family="Arial, sans-serif",
                    size=11,
                    color="black"
                ),
                selector=dict(type='sankey')
            )
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_22_hierarchical_sankey: {str(e)}")
            return None
    
    # ==================== ГРАФИК 23: МНОГОМЕРНОЕ ШКАЛИРОВАНИЕ ====================
    def plot_23_multidimensional_scaling(self):
        """23. Многомерное шкалирование важных предикторов (с использованием max_citations)"""
        try:
            # Выбираем ключевые предикторы
            predictors = ['author count', 'references_count', 'num_countries',
                         'max_annual_citations', 'article_age', 'normalized_attention']
            
            available_predictors = [p for p in predictors if p in self.df_processed.columns]
            
            if len(available_predictors) < 3:
                return None
            
            # Готовим данные
            analysis_data = self.df_processed[available_predictors + ['count']].dropna()
            
            if len(analysis_data) < 20:
                return None
            
            # Стандартизация
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(analysis_data[available_predictors])
            
            # PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            
            # Сохраняем данные
            self.plot_data['23_mds_analysis'] = {
                'explained_variance': pca.explained_variance_ratio_.tolist(),
                'components': pca.components_.tolist(),
                'pca_coordinates': pca_result.tolist(),
                'predictors': available_predictors
            }
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # График 1: PCA scatter plot
            scatter = ax1.scatter(pca_result[:, 0], pca_result[:, 1],
                                 c=analysis_data['count'], cmap='viridis',
                                 alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
            
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontweight='bold')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontweight='bold')
            ax1.set_title('PCA: Multidimensional Scaling of Predictors', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Local Mentions (count)', fontweight='bold')
            
            # График 2: Важность признаков
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            
            for i, predictor in enumerate(available_predictors):
                ax2.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                         color='red', alpha=0.5, head_width=0.05)
                ax2.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15,
                        predictor, color='red', fontsize=10, fontweight='bold')
            
            # Окружность корреляций
            circle = plt.Circle((0, 0), 1, fill=False, color='blue', alpha=0.3)
            ax2.add_artist(circle)
            
            ax2.set_xlabel('PC1', fontweight='bold')
            ax2.set_ylabel('PC2', fontweight='bold')
            ax2.set_title('Predictor Loadings on Principal Components', fontweight='bold')
            ax2.set_xlim(-1.2, 1.2)
            ax2.set_ylim(-1.2, 1.2)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            plt.suptitle('Multidimensional Analysis of Research Predictors', fontweight='bold', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_23_multidimensional_scaling: {str(e)}")
            return None
    
    # ==================== ГРАФИК 24: ВЗВЕШЕННАЯ СЕТЬ КОНЦЕПТОВ ====================
    def plot_24_concept_network_weighted(self):
        """24. Сеть концептов с весами по влиянию (с использованием max_citations)"""
        try:
            if 'concepts_list' not in self.df_processed.columns or 'max_citations' not in self.df_processed.columns:
                return None
            
            # Собираем топ концепты по встречаемости
            all_concepts = []
            for concepts in self.df_processed['concepts_list']:
                if isinstance(concepts, list):
                    all_concepts.extend([c.strip() for c in concepts])
            
            if len(all_concepts) == 0:
                return None
            
            concept_counts = pd.Series(all_concepts).value_counts()
            top_concepts = concept_counts.head(25).index.tolist()
            
            # Создаем взвешенный граф
            G = nx.Graph()
            
            # Добавляем узлы с весами по цитированиям
            for concept in top_concepts:
                # Находим все статьи с этим концептом
                concept_papers = []
                for idx, row in self.df_processed.iterrows():
                    if isinstance(row['concepts_list'], list) and concept in row['concepts_list']:
                        concept_papers.append(row.get('max_citations', 0))
                
                total_citations = sum(concept_papers)
                G.add_node(concept, citations=total_citations, papers=len(concept_papers))
            
            # Добавляем ребра с весами по совместной встречаемости
            for idx, row in self.df_processed.iterrows():
                if isinstance(row['concepts_list'], list):
                    concepts_in_paper = [c.strip() for c in row['concepts_list'] if c.strip() in top_concepts]
                    weight = row.get('max_citations', 1)
                    
                    for i in range(len(concepts_in_paper)):
                        for j in range(i+1, len(concepts_in_paper)):
                            c1, c2 = concepts_in_paper[i], concepts_in_paper[j]
                            if G.has_edge(c1, c2):
                                G[c1][c2]['weight'] += weight
                            else:
                                G.add_edge(c1, c2, weight=weight)
            
            # Сохраняем данные
            self.plot_data['24_concept_network_weighted'] = {
                'nodes': [{'concept': node, 'citations': G.nodes[node]['citations'],
                          'papers': G.nodes[node]['papers']} for node in G.nodes()],
                'edges': [{'concept1': u, 'concept2': v, 'weight': G[u][v]['weight']} 
                         for u, v in G.edges()]
            }
            
            # Визуализация
            fig, ax = plt.subplots(figsize=(16, 12))
            
            pos = nx.spring_layout(G, k=2, seed=42)
            
            # Размер узлов по цитированиям
            node_sizes = [G.nodes[n]['citations'] * 0.2 + 500 for n in G.nodes()]
            node_colors = [G.nodes[n]['papers'] for n in G.nodes()]
            
            nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                          node_color=node_colors, cmap='RdYlGn',
                                          alpha=0.8, edgecolors='black', linewidths=1.5, ax=ax)
            
            # Ребра с толщиной по весу
            if G.edges():
                edge_weights = [G[u][v]['weight'] * 0.01 for u, v in G.edges()]
                edges = nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5,
                                              edge_color='gray', style='solid', ax=ax)
            
            # Подписи
            nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
            
            ax.set_title('Concept Network with Citation Impact', fontweight='bold', fontsize=16)
            ax.axis('off')
            
            # Цветовая шкала
            sm = plt.cm.ScalarMappable(cmap='RdYlGn',
                                      norm=plt.Normalize(vmin=min(node_colors),
                                                       vmax=max(node_colors)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Number of Papers', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_24_concept_network_weighted: {str(e)}")
            return None
    
    # ==================== НОВЫЙ ГРАФИК 25: ВРЕМЕННАЯ АКТИВНОСТЬ ТЕРМИНОВ (VIOLIN PLOT) ====================
    def plot_25_term_temporal_density(self, hierarchy_level='Topic', top_percent=15, sort_by='first_year'):
        """
        25. Временная активность терминов с выбором уровня иерархии.
        hierarchy_level: 'Topic', 'Subfield', 'Field', 'Domain', 'Concepts'
        top_percent: процент терминов для отображения (по умолчанию 15%)
        sort_by: 'first_year', 'peak_year', 'total_attention'
        """
        try:
            # Определяем колонку для анализа
            level_col = None
            if hierarchy_level == 'Topic' and 'Topic' in self.df_processed.columns:
                level_col = 'Topic'
            elif hierarchy_level == 'Subfield' and 'Subfield' in self.df_processed.columns:
                level_col = 'Subfield'
            elif hierarchy_level == 'Field' and 'Field' in self.df_processed.columns:
                level_col = 'Field'
            elif hierarchy_level == 'Domain' and 'Domain' in self.df_processed.columns:
                level_col = 'Domain'
            elif hierarchy_level == 'Concepts' and 'concepts_list' in self.df_processed.columns:
                # Для Concepts нужно развернуть список
                level_col = 'Concepts'
            else:
                self.log_warning(f"Column '{hierarchy_level}' not found in data")
                return None
            
            if level_col is None:
                return None
            
            current_year = datetime.now().year
            
            # Собираем данные по терминам
            term_data = defaultdict(lambda: {'years': [], 'counts': [], 'total_attention': 0, 'papers': []})
            
            if hierarchy_level == 'Concepts':
                # Разворачиваем концепты из списка
                for idx, row in self.df_processed.iterrows():
                    year = row.get('year')
                    if pd.isna(year):
                        continue
                    year = int(year)
                    attention = row.get('count', 1)
                    
                    if isinstance(row['concepts_list'], list):
                        for concept in row['concepts_list']:
                            concept_clean = concept.strip()
                            if concept_clean:
                                term_data[concept_clean]['years'].append(year)
                                term_data[concept_clean]['counts'].append(attention)
                                term_data[concept_clean]['total_attention'] += attention
                                term_data[concept_clean]['papers'].append(idx)
            else:
                # Обычная иерархия
                for idx, row in self.df_processed.iterrows():
                    term = row.get(level_col)
                    year = row.get('year')
                    if pd.isna(term) or pd.isna(year):
                        continue
                    term = str(term).strip()
                    year = int(year)
                    attention = row.get('count', 1)
                    
                    term_data[term]['years'].append(year)
                    term_data[term]['counts'].append(attention)
                    term_data[term]['total_attention'] += attention
                    term_data[term]['papers'].append(idx)
            
            if len(term_data) < 3:
                self.log_warning(f"Insufficient terms ({len(term_data)}) for temporal density plot")
                return None
            
            # Вычисляем метрики для каждого термина
            term_metrics = {}
            for term, data in term_data.items():
                years = np.array(data['years'])
                if len(years) == 0:
                    continue
                
                first_year = years.min()
                last_year = years.max()
                
                # Находим год с максимальной плотностью (по количеству статей ИЛИ по сумме attention)
                year_counts = defaultdict(float)
                for y, att in zip(data['years'], data['counts']):
                    year_counts[y] += att
                
                peak_year = max(year_counts.items(), key=lambda x: x[1])[0]
                
                # Средняя плотность по годам
                all_years = range(first_year, last_year + 1)
                densities = []
                for y in all_years:
                    densities.append(year_counts.get(y, 0))
                
                mean_density = np.mean(densities) if densities else 0
                max_density = max(densities) if densities else 0
                
                term_metrics[term] = {
                    'first_year': first_year,
                    'last_year': last_year,
                    'peak_year': peak_year,
                    'total_attention': data['total_attention'],
                    'num_papers': len(data['papers']),
                    'years': data['years'],
                    'counts': data['counts'],
                    'mean_density': mean_density,
                    'max_density': max_density,
                    'activity_span': last_year - first_year
                }
            
            if len(term_metrics) < 3:
                self.log_warning("Insufficient valid term metrics")
                return None
            
            # Отбираем топ терминов по выбранному критерию
            if sort_by == 'total_attention':
                sorted_terms = sorted(term_metrics.items(), key=lambda x: x[1]['total_attention'], reverse=True)
            elif sort_by == 'activity_span':
                sorted_terms = sorted(term_metrics.items(), key=lambda x: x[1]['activity_span'], reverse=True)
            elif sort_by == 'peak_density':
                sorted_terms = sorted(term_metrics.items(), key=lambda x: x[1]['max_density'], reverse=True)
            else:  # first_year
                sorted_terms = sorted(term_metrics.items(), key=lambda x: x[1]['first_year'])
            
            # Берем топ процентов
            top_n = max(3, int(len(sorted_terms) * top_percent / 100))
            top_terms = sorted_terms[:top_n]
            
            # Сохраняем данные
            self.plot_data['25_term_temporal_density'] = {
                'hierarchy_level': hierarchy_level,
                'top_percent': top_percent,
                'sort_by': sort_by,
                'terms': [{'term': term, **metrics} for term, metrics in top_terms]
            }
            
            # Создаем фигуру
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # Подготавливаем данные для violin plot
            violin_data = []
            positions = []
            term_labels = []
            first_years = []
            peak_years = []
            last_years = []
            
            for i, (term, metrics) in enumerate(top_terms):
                # Создаем распределение лет с весами по attention
                weighted_years = []
                for year, att in zip(metrics['years'], metrics['counts']):
                    weighted_years.extend([year] * int(np.ceil(att)))
                
                if len(weighted_years) > 0:
                    violin_data.append(weighted_years)
                    positions.append(i + 1)
                    short_term = term[:30] + '...' if len(term) > 30 else term
                    term_labels.append(short_term)
                    first_years.append(metrics['first_year'])
                    peak_years.append(metrics['peak_year'])
                    last_years.append(metrics['last_year'])
            
            if len(violin_data) == 0:
                return None
            
            # Создаем violin plot
            parts = ax.violinplot(violin_data, positions=positions, widths=0.7,
                                 showmeans=False, showmedians=True, showextrema=False)
            
            # Настройка цветов violins
            colors_violin = plt.cm.viridis(np.linspace(0.2, 0.8, len(violin_data)))
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors_violin[i])
                pc.set_alpha(0.6)
                pc.set_edgecolor('black')
                pc.set_linewidth(1)
            
            # Добавляем медианы
            for i, data in enumerate(violin_data):
                median = np.median(data)
                ax.scatter(i+1, median, color='black', s=60, zorder=5,
                          marker='s', label='Median' if i == 0 else "")
            
            # Добавляем маркеры для first, peak, last
            for i, (term, metrics) in enumerate(top_terms):
                # First year (зеленый треугольник вверх)
                ax.scatter(i+1, metrics['first_year'], color='green', s=100, zorder=6,
                          marker='^', edgecolors='black', linewidth=1.5,
                          label='First appearance' if i == 0 else "")
                
                # Peak year (красный ромб)
                ax.scatter(i+1, metrics['peak_year'], color='red', s=100, zorder=6,
                          marker='D', edgecolors='black', linewidth=1.5,
                          label='Peak density' if i == 0 else "")
                
                # Last year (синий треугольник вниз)
                ax.scatter(i+1, metrics['last_year'], color='blue', s=100, zorder=6,
                          marker='v', edgecolors='black', linewidth=1.5,
                          label='Last appearance' if i == 0 else "")
            
            # Соединяем first_year линией (показывает волну появления новых тем)
            first_years_sorted = [first_years[i] for i in range(len(first_years))]
            ax.plot(positions, first_years_sorted, 'g--', linewidth=2, alpha=0.7,
                   label='Emergence wave')
            
            # Настройка осей
            ax.set_xticks(positions)
            ax.set_xticklabels(term_labels, rotation=45, ha='right', fontsize=9)
            ax.set_xlabel(f'{hierarchy_level} (Top {top_percent}% by {sort_by.replace("_", " ")})', fontweight='bold')
            ax.set_ylabel('Publication Year', fontweight='bold')
            ax.set_title(f'Temporal Activity of {hierarchy_level}s: Distribution, First/Peak/Last Appearance',
                        fontweight='bold', fontsize=16)
            
            # Инвертируем ось Y (чтобы свежие годы были сверху)
            ax.invert_yaxis()
            
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='upper left', fontsize=9, ncol=2)
            
            # Добавляем аннотацию с информацией о топ-терминах
            info_text = f"Total {hierarchy_level}s analyzed: {len(term_metrics)}\n"
            info_text += f"Showing top {top_n} of {len(sorted_terms)} ({top_percent}%)\n"
            info_text += f"Sorted by: {sort_by.replace('_', ' ')}"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_25_term_temporal_density: {str(e)}")
            return None

    # ==================== ГРАФИК 26: SUNBURST CHART ====================
    def plot_26_sunburst_chart(self):
        """26. Sunburst диаграмма: Domain → Field → Topic с цветом по среднему count"""
        try:
            required_cols = ['Domain', 'Field', 'Topic', 'count']
            available_cols = [col for col in required_cols if col in self.df_processed.columns]
            
            if len(available_cols) < 2:
                return None
            
            valid_data = self.df_processed.dropna(subset=available_cols)
            if len(valid_data) < 10:
                return None
            
            # Агрегируем данные для sunburst
            hierarchy_data = valid_data.groupby(['Domain', 'Field', 'Topic']).agg({
                'count': ['sum', 'mean', 'size']
            }).reset_index()
            
            hierarchy_data.columns = ['Domain', 'Field', 'Topic', 'total_attention', 'mean_attention', 'num_papers']
            
            # Убираем пустые значения
            hierarchy_data = hierarchy_data[hierarchy_data['Domain'].notna()]
            hierarchy_data = hierarchy_data[hierarchy_data['Field'].notna()]
            hierarchy_data = hierarchy_data[hierarchy_data['Topic'].notna()]
            
            if len(hierarchy_data) == 0:
                return None
            
            # Создаем sunburst диаграмму
            fig = px.sunburst(
                hierarchy_data,
                path=['Domain', 'Field', 'Topic'],
                values='total_attention',
                color='mean_attention',
                color_continuous_scale='RdBu',
                title="Sunburst Chart: Knowledge Structure (Domain → Field → Topic)<br>Color = Mean Attention, Size = Total Attention",
                hover_data={'num_papers': True, 'mean_attention': ':.2f'}
            )
            
            fig.update_layout(
                width=1000,
                height=800,
                font=dict(size=11)
            )
            
            # Сохраняем данные
            self.plot_data['26_sunburst_chart'] = hierarchy_data.head(100).to_dict('records')
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_26_sunburst_chart: {str(e)}")
            return None
    
    # ==================== ГРАФИК 27: 3D BUBBLE CHART ====================
    def plot_27_3d_bubble_chart(self):
        """27. 3D пузырьковая диаграмма: count vs max_citations vs references_count"""
        try:
            required_cols = ['count', 'max_citations', 'references_count', 'author count', 'num_countries']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            valid_data = valid_data[valid_data['max_citations'] > 0]
            
            if len(valid_data) < 10:
                return None
            
            # Берем топ-200 для читаемости
            if len(valid_data) > 200:
                valid_data = valid_data.nlargest(200, 'count')
            
            fig = go.Figure(data=[go.Scatter3d(
                x=valid_data['count'],
                y=valid_data['max_citations'],
                z=valid_data['references_count'],
                mode='markers',
                marker=dict(
                    size=valid_data['author count'] * 3,
                    color=valid_data['num_countries'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Number of Countries"),
                    opacity=0.7,
                    line=dict(width=0.5, color='black')
                ),
                text=valid_data['Title'],
                hoverinfo='text',
                hovertemplate='<b>%{text}</b><br>' +
                              'Count: %{x}<br>' +
                              'Max Citations: %{y}<br>' +
                              'References: %{z}<br>' +
                              'Authors: %{marker.size:.0f}<br>' +
                              'Countries: %{marker.color}<extra></extra>'
            )])
            
            fig.update_layout(
                title="3D Bubble Chart: Attention vs Citations vs References",
                scene=dict(
                    xaxis_title="Local Mentions (count)",
                    yaxis_title="Max Citations (max(CR, OA))",
                    zaxis_title="Number of References",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                width=1000,
                height=800,
                margin=dict(l=0, r=0, b=0, t=50)
            )
            
            # Сохраняем данные
            self.plot_data['27_3d_bubble_chart'] = valid_data[required_cols].head(100).to_dict('records')
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_27_3d_bubble_chart: {str(e)}")
            return None
    
    # ==================== ГРАФИК 28: СЕТЬ АВТОРЫ-ЖУРНАЛЫ-ТЕМЫ ====================
    def plot_28_network_authors_journals_topics(self):
        """28. Сетевой граф: Авторы ↔ Журналы ↔ Темы"""
        try:
            if 'authors_list' not in self.df_processed.columns or 'Full journal Name' not in self.df_processed.columns:
                return None
            
            # Ограничиваем топ-10 авторов по count для читаемости
            # Сначала собираем суммарный count по авторам
            author_attention = defaultdict(float)
            for idx, row in self.df_processed.iterrows():
                if isinstance(row['authors_list'], list):
                    weight = row.get('count', 1)
                    for author in row['authors_list']:
                        if author.strip():
                            author_attention[author.strip()] += weight
            
            top_authors = [a for a, _ in sorted(author_attention.items(), key=lambda x: x[1], reverse=True)[:15]]
            
            # Топ-10 журналов
            top_journals = self.df_processed['Full journal Name'].value_counts().head(10).index.tolist()
            
            # Топ-10 концептов
            all_concepts = []
            for concepts in self.df_processed['concepts_list']:
                if isinstance(concepts, list):
                    all_concepts.extend([c.strip() for c in concepts])
            concept_counts = pd.Series(all_concepts).value_counts()
            top_concepts = concept_counts.head(10).index.tolist()
            
            # Создаем граф
            G = nx.Graph()
            
            # Добавляем узлы
            for author in top_authors:
                G.add_node(author, type='author', attention=author_attention[author])
            for journal in top_journals:
                G.add_node(journal, type='journal')
            for concept in top_concepts:
                G.add_node(concept, type='concept')
            
            # Добавляем ребра (автор-журнал)
            for idx, row in self.df_processed.iterrows():
                if isinstance(row['authors_list'], list) and row['Full journal Name'] in top_journals:
                    weight = row.get('count', 1)
                    for author in row['authors_list']:
                        if author in top_authors:
                            if G.has_edge(author, row['Full journal Name']):
                                G[author][row['Full journal Name']]['weight'] += weight
                            else:
                                G.add_edge(author, row['Full journal Name'], weight=weight)
            
            # Добавляем ребра (журнал-концепт)
            for idx, row in self.df_processed.iterrows():
                if row['Full journal Name'] in top_journals and isinstance(row['concepts_list'], list):
                    weight = row.get('count', 1)
                    for concept in row['concepts_list']:
                        if concept.strip() in top_concepts:
                            if G.has_edge(row['Full journal Name'], concept.strip()):
                                G[row['Full journal Name']][concept.strip()]['weight'] += weight
                            else:
                                G.add_edge(row['Full journal Name'], concept.strip(), weight=weight)
            
            if len(G.nodes()) == 0:
                return None
            
            # Позиционирование узлов
            pos = nx.spring_layout(G, k=1.5, seed=42)
            
            # Разделяем узлы по типам
            author_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'author']
            journal_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'journal']
            concept_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'concept']
            
            # Создаем фигуру plotly
            fig = go.Figure()
            
            # Добавляем ребра
            for u, v, data in G.edges(data=True):
                weight = data.get('weight', 1)
                line_width = 1 + weight / 10
                
                fig.add_trace(go.Scatter(
                    x=[pos[u][0], pos[v][0]],
                    y=[pos[u][1], pos[v][1]],
                    mode='lines',
                    line=dict(width=line_width, color='lightgray'),
                    hoverinfo='none',
                    showlegend=False
                ))
            
            # Добавляем узлы-авторы
            if author_nodes:
                x_author = [pos[n][0] for n in author_nodes]
                y_author = [pos[n][1] for n in author_nodes]
                sizes = [G.nodes[n].get('attention', 1) * 10 + 20 for n in author_nodes]
                
                fig.add_trace(go.Scatter(
                    x=x_author,
                    y=y_author,
                    mode='markers+text',
                    marker=dict(size=sizes, color='#2E86AB', line=dict(width=1, color='black')),
                    text=[n[:15] + '...' if len(n) > 15 else n for n in author_nodes],
                    textposition='top center',
                    textfont=dict(size=9),
                    name='Authors',
                    hovertemplate='<b>%{text}</b><br>Attention: %{marker.size}<extra></extra>'
                ))
            
            # Добавляем узлы-журналы
            if journal_nodes:
                x_journal = [pos[n][0] for n in journal_nodes]
                y_journal = [pos[n][1] for n in journal_nodes]
                
                fig.add_trace(go.Scatter(
                    x=x_journal,
                    y=y_journal,
                    mode='markers+text',
                    marker=dict(size=25, color='#C73E1D', line=dict(width=1, color='black'), symbol='square'),
                    text=[n[:20] + '...' if len(n) > 20 else n for n in journal_nodes],
                    textposition='top center',
                    textfont=dict(size=8),
                    name='Journals',
                    hovertemplate='<b>%{text}</b><extra></extra>'
                ))
            
            # Добавляем узлы-концепты
            if concept_nodes:
                x_concept = [pos[n][0] for n in concept_nodes]
                y_concept = [pos[n][1] for n in concept_nodes]
                
                fig.add_trace(go.Scatter(
                    x=x_concept,
                    y=y_concept,
                    mode='markers+text',
                    marker=dict(size=20, color='#6B8E23', line=dict(width=1, color='black'), symbol='diamond'),
                    text=[n[:15] + '...' if len(n) > 15 else n for n in concept_nodes],
                    textposition='top center',
                    textfont=dict(size=8),
                    name='Concepts',
                    hovertemplate='<b>%{text}</b><extra></extra>'
                ))
            
            fig.update_layout(
                title="Network Graph: Authors ↔ Journals ↔ Topics",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor='white',
                width=1200,
                height=900,
                hovermode='closest',
                legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
            )
            
            # Сохраняем данные
            self.plot_data['28_network_authors_journals_topics'] = {
                'authors': author_nodes,
                'journals': journal_nodes,
                'concepts': concept_nodes,
                'edges_count': G.number_of_edges()
            }
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_28_network_authors_journals_topics: {str(e)}")
            return None
    
    # ==================== ГРАФИК 29: АНИМИРОВАННАЯ ПУЗЫРЬКОВАЯ ДИАГРАММА ====================
    def plot_29_animated_bubble_chart(self):
        """29. Анимированная пузырьковая диаграмма по годам: count vs max_citations"""
        try:
            required_cols = ['year', 'count', 'max_citations', 'author count', 'num_countries']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            valid_data = valid_data[valid_data['max_citations'] > 0]
            valid_data = valid_data[valid_data['year'].notna()]
            
            if len(valid_data) < 10:
                return None
            
            # Преобразуем год в int
            valid_data['year'] = valid_data['year'].astype(int)
            
            # Создаем анимированную диаграмму
            fig = px.scatter(
                valid_data,
                x='count',
                y='max_citations',
                size='author count',
                color='num_countries',
                hover_name='Title',
                animation_frame='year',
                animation_group='doi',
                size_max=50,
                color_continuous_scale='Viridis',
                title="Animated Bubble Chart: Attention vs Citations Over Time",
                labels={
                    'count': 'Local Mentions (count)',
                    'max_citations': 'Max Citations (max(CR, OA))',
                    'author count': 'Number of Authors',
                    'num_countries': 'Number of Countries'
                }
            )
            
            fig.update_layout(
                xaxis=dict(title="Local Mentions (count)", type="log"),
                yaxis=dict(title="Max Citations (max(CR, OA))", type="log"),
                height=700,
                width=1000,
                hovermode='closest'
            )
            
            # Сохраняем данные
            self.plot_data['29_animated_bubble_chart'] = valid_data[required_cols].to_dict('records')
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_29_animated_bubble_chart: {str(e)}")
            return None
    
    # ==================== ГРАФИК 30: ХОРДОВАЯ ДИАГРАММА ТЕМ ====================
    def plot_30_topic_chord_diagram(self):
        """30. Хордовая диаграмма совместной встречаемости тем (Topic)"""
        try:
            if 'Topic' not in self.df_processed.columns:
                return None
            
            # Собираем данные о совместной встречаемости Topic
            topic_pairs = []
            topic_weights = defaultdict(float)
            
            for idx, row in self.df_processed.iterrows():
                topic = row['Topic']
                if pd.notna(topic) and str(topic).strip():
                    topic_str = str(topic).strip()
                    weight = row.get('count', 1)
                    topic_weights[topic_str] += weight
            
            if len(topic_weights) < 3:
                self.log_warning("Insufficient data for topic chord diagram")
                return None
            
            # Выбираем топ N тем
            top_topics = sorted(topic_weights.items(), key=lambda x: x[1], reverse=True)[:15]
            top_topic_names = [t[0] for t in top_topics]
            
            # Создаем матрицу связей (совместная встречаемость тем в разных статьях? 
            # Topic уникален для статьи, поэтому связи строим через общие концепты или поля)
            # Альтернатива: используем Subfield или Field для связей
            
            if 'Subfield' in self.df_processed.columns:
                # Строим связи через общий Subfield
                subfield_to_topics = defaultdict(list)
                for idx, row in self.df_processed.iterrows():
                    topic = row['Topic']
                    subfield = row['Subfield']
                    if pd.notna(topic) and pd.notna(subfield):
                        subfield_to_topics[str(subfield)].append(str(topic))
                
                n = len(top_topic_names)
                topic_to_idx = {name: i for i, name in enumerate(top_topic_names)}
                adjacency_matrix = np.zeros((n, n))
                
                for subfield, topics in subfield_to_topics.items():
                    unique_topics = list(set(topics))
                    for i in range(len(unique_topics)):
                        for j in range(i+1, len(unique_topics)):
                            if unique_topics[i] in topic_to_idx and unique_topics[j] in topic_to_idx:
                                i_idx = topic_to_idx[unique_topics[i]]
                                j_idx = topic_to_idx[unique_topics[j]]
                                adjacency_matrix[i_idx, j_idx] += 1
                                adjacency_matrix[j_idx, i_idx] += 1
                
                # Создаем цветовую схему
                colors = []
                for i in range(n):
                    hue = i / n
                    rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                    color_hex = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
                    colors.append(color_hex)
                
                # Расставляем узлы по кругу
                angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
                radius = 1.0
                x_positions = radius * np.cos(angles)
                y_positions = radius * np.sin(angles)
                
                # Позиции для текста снаружи
                text_radius = 1.25
                text_x = text_radius * np.cos(angles)
                text_y = text_radius * np.sin(angles)
                
                # Создаем фигуру
                fig = go.Figure()
                
                # Добавляем внешние метки
                fig.add_trace(go.Scatter(
                    x=text_x,
                    y=text_y,
                    mode='text',
                    text=top_topic_names,
                    textposition='middle center',
                    textfont=dict(size=10, color='black', family='Arial, sans-serif'),
                    hovertext=[f"<b>{name}</b><br>Total weight: {weight:.1f}" for name, weight in top_topics],
                    hoverinfo='text',
                    showlegend=False
                ))
                
                # Добавляем узлы
                fig.add_trace(go.Scatter(
                    x=x_positions,
                    y=y_positions,
                    mode='markers',
                    marker=dict(size=20, color=colors, line=dict(color='black', width=1.5)),
                    hovertext=[f"<b>{name}</b><br>Total weight: {weight:.1f}" for name, weight in top_topics],
                    hoverinfo='text',
                    showlegend=False
                ))
                
                # Добавляем хорды
                max_weight = max(adjacency_matrix.flatten()) if adjacency_matrix.size > 0 else 1
                
                for i in range(n):
                    for j in range(i+1, n):
                        weight = adjacency_matrix[i, j]
                        if weight > 0:
                            t = np.linspace(0, 1, 100)
                            p0 = np.array([x_positions[i], y_positions[i]])
                            p3 = np.array([x_positions[j], y_positions[j]])
                            
                            mid_angle = (angles[i] + angles[j]) / 2
                            if abs(angles[i] - angles[j]) > np.pi:
                                mid_angle += np.pi
                            control_offset = 0.4 * (1 + weight / max_weight)
                            ctrl_point = np.array([control_offset * np.cos(mid_angle), control_offset * np.sin(mid_angle)])
                            
                            curve_x = (1-t)**3 * p0[0] + 3*(1-t)**2*t * ctrl_point[0] + 3*(1-t)*t**2 * ctrl_point[0] + t**3 * p3[0]
                            curve_y = (1-t)**3 * p0[1] + 3*(1-t)**2*t * ctrl_point[1] + 3*(1-t)*t**2 * ctrl_point[1] + t**3 * p3[1]
                            
                            start_rgb = [int(colors[i][1:3], 16), int(colors[i][3:5], 16), int(colors[i][5:7], 16)]
                            end_rgb = [int(colors[j][1:3], 16), int(colors[j][3:5], 16), int(colors[j][5:7], 16)]
                            mixed_rgb = [(start_rgb[k] + end_rgb[k]) // 2 for k in range(3)]
                            chord_color = f'rgba({mixed_rgb[0]}, {mixed_rgb[1]}, {mixed_rgb[2]}, 0.8)'
                            
                            line_width = 2 + 10 * weight / max_weight
                            
                            fig.add_trace(go.Scatter(
                                x=curve_x,
                                y=curve_y,
                                mode='lines',
                                line=dict(width=line_width, color=chord_color),
                                hoverinfo='none',
                                showlegend=False
                            ))
                
                # Добавляем внутренний круг
                theta = np.linspace(0, 2*np.pi, 100)
                inner_radius = 0.85
                fig.add_trace(go.Scatter(
                    x=inner_radius * np.cos(theta),
                    y=inner_radius * np.sin(theta),
                    mode='lines',
                    line=dict(color='white', width=2),
                    fill='toself',
                    fillcolor='rgba(240, 240, 240, 0.3)',
                    hoverinfo='none',
                    showlegend=False
                ))
                
                fig.update_layout(
                    title=f"Topic Co-occurrence Chord Diagram (Top {len(top_topics)} Topics)",
                    width=1000,
                    height=1000,
                    xaxis=dict(visible=False, range=[-1.5, 1.5], scaleanchor="y", scaleratio=1),
                    yaxis=dict(visible=False, range=[-1.5, 1.5]),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    showlegend=False,
                    hovermode='closest'
                )
                
                # Сохраняем данные
                self.plot_data['30_topic_chord_diagram'] = {
                    'topics': top_topic_names,
                    'adjacency_matrix': adjacency_matrix.tolist()
                }
                
                return fig
            else:
                self.log_warning("Subfield column not found for topic chord diagram")
                return None
            
        except Exception as e:
            self.log_error(f"Error in plot_30_topic_chord_diagram: {str(e)}")
            return None
    
    # ==================== НОВЫЙ ГРАФИК 31: АКТИВНЫЕ ЭЛЕМЕНТЫ ЗА ПОСЛЕДНИЕ 5 ЛЕТ ====================
    def plot_31_active_elements_last_5_years(self, hierarchy_level='Topic', top_n=15):
        """
        31. Наиболее активные элементы иерархии за последние 5 лет.
        Показывает total_attention и количество статей для терминов, появившихся в последние 5 лет.
        """
        try:
            current_year = datetime.now().year
            start_year = current_year - 5
            
            # Определяем колонку для анализа
            level_col = None
            if hierarchy_level == 'Topic' and 'Topic' in self.df_processed.columns:
                level_col = 'Topic'
            elif hierarchy_level == 'Subfield' and 'Subfield' in self.df_processed.columns:
                level_col = 'Subfield'
            elif hierarchy_level == 'Field' and 'Field' in self.df_processed.columns:
                level_col = 'Field'
            elif hierarchy_level == 'Domain' and 'Domain' in self.df_processed.columns:
                level_col = 'Domain'
            elif hierarchy_level == 'Concepts' and 'concepts_list' in self.df_processed.columns:
                level_col = 'Concepts'
            else:
                self.log_warning(f"Column '{hierarchy_level}' not found in data")
                return None
            
            if level_col is None:
                return None
            
            # Фильтруем данные за последние 5 лет
            recent_data = self.df_processed[self.df_processed['year'] >= start_year].copy()
            
            if len(recent_data) < 5:
                self.log_warning(f"Insufficient recent data (only {len(recent_data)} papers in last 5 years)")
                return None
            
            # Собираем данные по терминам
            term_stats = defaultdict(lambda: {'total_attention': 0, 'num_papers': 0, 'avg_attention': 0})
            
            if hierarchy_level == 'Concepts':
                for idx, row in recent_data.iterrows():
                    attention = row.get('count', 1)
                    if isinstance(row['concepts_list'], list):
                        for concept in row['concepts_list']:
                            concept_clean = concept.strip()
                            if concept_clean:
                                term_stats[concept_clean]['total_attention'] += attention
                                term_stats[concept_clean]['num_papers'] += 1
            else:
                for idx, row in recent_data.iterrows():
                    term = row.get(level_col)
                    if pd.isna(term):
                        continue
                    term = str(term).strip()
                    attention = row.get('count', 1)
                    term_stats[term]['total_attention'] += attention
                    term_stats[term]['num_papers'] += 1
            
            if len(term_stats) == 0:
                self.log_warning("No terms found in recent data")
                return None
            
            # Вычисляем среднее внимание
            for term in term_stats:
                term_stats[term]['avg_attention'] = term_stats[term]['total_attention'] / term_stats[term]['num_papers']
            
            # Сортируем и берем топ N
            sorted_terms = sorted(term_stats.items(), key=lambda x: x[1]['total_attention'], reverse=True)
            top_terms = sorted_terms[:top_n]
            
            # Сохраняем данные
            self.plot_data['31_active_elements_last_5_years'] = {
                'hierarchy_level': hierarchy_level,
                'period': f"{start_year}-{current_year}",
                'terms': [{'term': term, **stats} for term, stats in top_terms]
            }
            
            # Создаем фигуру
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            terms = [t[0][:35] + '...' if len(t[0]) > 35 else t[0] for t in top_terms]
            total_att = [t[1]['total_attention'] for t in top_terms]
            num_papers = [t[1]['num_papers'] for t in top_terms]
            avg_att = [t[1]['avg_attention'] for t in top_terms]
            
            y_pos = np.arange(len(terms))
            
            # График 1: Total Attention (столбцы)
            bars1 = ax1.barh(y_pos, total_att, color='steelblue', edgecolor='black', alpha=0.8)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(terms, fontsize=9)
            ax1.set_xlabel('Total Attention (last 5 years)', fontweight='bold')
            ax1.set_title(f'Most Active {hierarchy_level}s by Total Attention\n({start_year}-{current_year})',
                         fontweight='bold')
            ax1.invert_yaxis()
            
            # Добавляем значения и количество статей
            for i, (bar, papers) in enumerate(zip(bars1, num_papers)):
                width = bar.get_width()
                ax1.text(width * 1.01, bar.get_y() + bar.get_height()/2,
                        f'n={papers}', va='center', fontsize=8, fontweight='bold')
            
            ax1.grid(True, alpha=0.3, axis='x')
            
            # График 2: Bubble chart (Total Attention vs Avg Attention)
            scatter = ax2.scatter(total_att, avg_att, s=np.array(num_papers) * 15,
                                 c=total_att, cmap='plasma', alpha=0.7,
                                 edgecolors='black', linewidth=1.5)
            
            ax2.set_xlabel('Total Attention', fontweight='bold')
            ax2.set_ylabel('Average Attention per Paper', fontweight='bold')
            ax2.set_title('Attention Distribution: Total vs Average', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Добавляем аннотации для топ-5
            for i in range(min(5, len(top_terms))):
                term_short = terms[i][:20] + '...' if len(terms[i]) > 20 else terms[i]
                ax2.annotate(term_short, xy=(total_att[i], avg_att[i]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, alpha=0.8)
            
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Total Attention', fontweight='bold')
            
            plt.suptitle(f'Most Active {hierarchy_level}s in the Last 5 Years ({start_year}-{current_year})',
                        fontweight='bold', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_31_active_elements_last_5_years: {str(e)}")
            return None
    
    # ==================== НОВЫЙ ГРАФИК 32: АНИМИРОВАННАЯ ТЕПЛОВАЯ КАРТА ЖУРНАЛОВ ====================
    def plot_32_animated_journal_heatmap(self, top_journals=15):
        """32. Анимированная тепловая карта журналов по годам"""
        try:
            required_cols = ['Full journal Name', 'year', 'max_annual_citations']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            # Выбираем топ журналов
            journal_counts = self.df_processed['Full journal Name'].value_counts()
            top_journals_list = journal_counts.head(top_journals).index.tolist()
            
            heatmap_data = self.df_processed[self.df_processed['Full journal Name'].isin(top_journals_list)].copy()
            if len(heatmap_data) == 0:
                return None
            
            # Создаем данные для анимации
            years = sorted(heatmap_data['year'].unique())
            
            # Создаем фреймы данных для каждого года
            frames = []
            for year in years:
                year_data = heatmap_data[heatmap_data['year'] == year]
                pivot = year_data.pivot_table(
                    values='max_annual_citations',
                    index='Full journal Name',
                    columns='year',
                    aggfunc='mean',
                    fill_value=0
                )
                frames.append(pivot)
            
            # Создаем анимированную тепловую карту
            fig = px.imshow(
                frames,
                animation_frame=0,
                labels=dict(x="Year", y="Journal", color="Annual Citations"),
                title=f"Animated Journal Heatmap: Annual Citation Rate Over Time (Top {top_journals} Journals)",
                color_continuous_scale="Blues"
            )
            
            # Настройка осей
            fig.update_layout(
                height=600,
                width=1000,
                xaxis_title="Publication Year",
                yaxis_title="Journal"
            )
            
            # Сохраняем данные
            self.plot_data['32_animated_journal_heatmap'] = {
                'top_journals': top_journals_list,
                'years': years,
                'frames_count': len(frames)
            }
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_32_animated_journal_heatmap: {str(e)}")
            return None
    
    # ==================== НОВЫЙ ГРАФИК 33: АНИМИРОВАННАЯ ХОРДОВАЯ ДИАГРАММА СТРАН ====================
    def plot_33_animated_country_chord(self, periods=4):
        """33. Анимированная хордовая диаграмма коллабораций стран по периодам"""
        try:
            if 'countries_list' not in self.df_processed.columns:
                return None
            
            current_year = datetime.now().year
            year_min = int(self.df_processed['year'].min())
            year_max = int(self.df_processed['year'].max())
            
            # Создаем периоды
            period_length = max(1, (year_max - year_min) // periods)
            periods_list = []
            for i in range(periods):
                start = year_min + i * period_length
                end = start + period_length - 1 if i < periods - 1 else year_max
                periods_list.append((start, end, f"{start}-{end}"))
            
            # Создаем фигуру с субплотами
            fig, axes = plt.subplots(2, 2, figsize=(16, 16))
            axes = axes.flatten()
            
            for idx, (start, end, label) in enumerate(periods_list):
                if idx >= len(axes):
                    break
                
                # Фильтруем данные за период
                period_data = self.df_processed[(self.df_processed['year'] >= start) & 
                                                (self.df_processed['year'] <= end)]
                
                if len(period_data) < 10:
                    axes[idx].text(0.5, 0.5, f"Insufficient data\nfor {label}",
                                  ha='center', va='center', transform=axes[idx].transAxes)
                    axes[idx].axis('off')
                    continue
                
                # Собираем коллаборации
                country_pairs = []
                country_weights = defaultdict(float)
                
                for _, row in period_data.iterrows():
                    if isinstance(row['countries_list'], list) and len(row['countries_list']) >= 2:
                        countries = [c.strip().upper() for c in row['countries_list']]
                        weight = row.get('count', 1)
                        
                        for country in countries:
                            country_weights[country] += weight
                        
                        for i in range(len(countries)):
                            for j in range(i+1, len(countries)):
                                pair = tuple(sorted([countries[i], countries[j]]))
                                country_pairs.append((pair, weight))
                
                if len(country_weights) < 3:
                    axes[idx].text(0.5, 0.5, f"Few collaborations\nin {label}",
                                  ha='center', va='center', transform=axes[idx].transAxes)
                    axes[idx].axis('off')
                    continue
                
                # Выбираем топ стран
                top_countries = sorted(country_weights.items(), key=lambda x: x[1], reverse=True)[:10]
                top_names = [c[0] for c in top_countries]
                
                # Создаем матрицу связей
                n = len(top_names)
                country_to_idx = {name: i for i, name in enumerate(top_names)}
                adj_matrix = np.zeros((n, n))
                
                for (c1, c2), weight in country_pairs:
                    if c1 in country_to_idx and c2 in country_to_idx:
                        i, j = country_to_idx[c1], country_to_idx[c2]
                        adj_matrix[i, j] += weight
                        adj_matrix[j, i] += weight
                
                # Визуализация как тепловой карты (для простоты анимации)
                im = axes[idx].imshow(adj_matrix, cmap='Blues', vmin=0)
                axes[idx].set_xticks(range(n))
                axes[idx].set_yticks(range(n))
                axes[idx].set_xticklabels(top_names, rotation=45, ha='right', fontsize=8)
                axes[idx].set_yticklabels(top_names, fontsize=8)
                axes[idx].set_title(f"Country Collaborations\n{label}", fontweight='bold')
                
                # Добавляем значения
                for i in range(n):
                    for j in range(n):
                        if adj_matrix[i, j] > 0:
                            axes[idx].text(j, i, f'{adj_matrix[i, j]:.0f}',
                                         ha='center', va='center', fontsize=7)
            
            plt.suptitle('Animated Country Collaboration Heatmaps by Period', fontweight='bold', fontsize=16)
            plt.tight_layout()
            
            # Сохраняем данные
            self.plot_data['33_animated_country_chord'] = {
                'periods': periods_list,
                'total_periods': periods
            }
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_33_animated_country_chord: {str(e)}")
            return None
    
    # ==================== НОВЫЙ ГРАФИК 34: АНИМИРОВАННАЯ КАРТА МИРА ====================
    def plot_34_animated_worldmap(self):
        """34. Анимированная карта мира с пузырьками по странам (внимание по годам)"""
        try:
            if 'countries_list' not in self.df_processed.columns:
                return None
            
            # Собираем данные по странам и годам
            country_year_data = defaultdict(lambda: defaultdict(float))
            
            for idx, row in self.df_processed.iterrows():
                year = row.get('year')
                if pd.isna(year):
                    continue
                year = int(year)
                attention = row.get('count', 1)
                
                if isinstance(row['countries_list'], list):
                    for country in row['countries_list']:
                        country_clean = country.strip().upper()
                        if country_clean:
                            country_year_data[country_clean][year] += attention
            
            if len(country_year_data) == 0:
                return None
            
            # Подготавливаем данные для plotly
            years = sorted(set(y for country in country_year_data.values() for y in country.keys()))
            
            # Создаем DataFrame для всех лет
            data_rows = []
            for country, year_dict in country_year_data.items():
                for year in years:
                    attention = year_dict.get(year, 0)
                    if attention > 0:
                        data_rows.append({'country': country, 'year': year, 'attention': attention})
            
            df_map = pd.DataFrame(data_rows)
            
            if len(df_map) == 0:
                return None
            
            # Создаем анимированную карту мира
            fig = px.scatter_geo(
                df_map,
                locations='country',
                locationmode='country names',
                size='attention',
                hover_name='country',
                animation_frame='year',
                projection='natural earth',
                title='Animated World Map: Research Attention by Country Over Time',
                size_max=50,
                color='attention',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                height=700,
                width=1200,
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    coastlinecolor='gray',
                    showland=True,
                    landcolor='rgb(240, 240, 240)',
                    showocean=True,
                    oceancolor='rgb(200, 220, 240)'
                )
            )
            
            # Сохраняем данные
            self.plot_data['34_animated_worldmap'] = {
                'countries': list(country_year_data.keys()),
                'years': years,
                'total_attention': sum(sum(d.values()) for d in country_year_data.values())
            }
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_34_animated_worldmap: {str(e)}")
            return None
    
    # ==================== НОВЫЙ ГРАФИК 35: АНИМИРОВАННАЯ ГИСТОГРАММА ТОП-10 ТЕМ ====================
    def plot_35_animated_top10_topics(self):
        """35. Анимированная гистограмма топ-10 тем по годам"""
        try:
            if 'Topic' not in self.df_processed.columns:
                return None
            
            # Собираем данные по темам и годам
            topic_year_data = defaultdict(lambda: defaultdict(float))
            
            for idx, row in self.df_processed.iterrows():
                topic = row.get('Topic')
                year = row.get('year')
                if pd.isna(topic) or pd.isna(year):
                    continue
                topic = str(topic).strip()
                year = int(year)
                attention = row.get('count', 1)
                
                topic_year_data[topic][year] += attention
            
            if len(topic_year_data) == 0:
                return None
            
            # Определяем топ-10 тем по суммарному вниманию
            total_by_topic = {topic: sum(year_dict.values()) for topic, year_dict in topic_year_data.items()}
            top_topics = sorted(total_by_topic.items(), key=lambda x: x[1], reverse=True)[:10]
            top_topic_names = [t[0] for t in top_topics]
            
            # Подготавливаем данные для анимации
            years = sorted(set(y for topic in top_topic_names for y in topic_year_data[topic].keys()))
            
            # Создаем фреймы для каждого года
            frames_data = []
            for year in sorted(years):
                year_data = []
                for topic in top_topic_names:
                    attention = topic_year_data[topic].get(year, 0)
                    year_data.append({'topic': topic[:30] + '...' if len(topic) > 30 else topic, 
                                     'attention': attention, 'year': year})
                frames_data.extend(year_data)
            
            df_topics = pd.DataFrame(frames_data)
            
            if len(df_topics) == 0:
                return None
            
            # Создаем анимированную горизонтальную гистограмму
            fig = px.bar(
                df_topics,
                x='attention',
                y='topic',
                animation_frame='year',
                orientation='h',
                title='Top 10 Research Topics Evolution Over Time',
                labels={'attention': 'Attention (Mentions)', 'topic': 'Research Topic'},
                color='attention',
                color_continuous_scale='Viridis',
                range_x=[0, df_topics['attention'].max() * 1.1]
            )
            
            fig.update_layout(
                height=600,
                width=1000,
                yaxis=dict(categoryorder='total ascending'),
                xaxis_title="Attention (Mentions)",
                yaxis_title="Research Topic"
            )
            
            # Сохраняем данные
            self.plot_data['35_animated_top10_topics'] = {
                'top_topics': top_topic_names,
                'years': years,
                'total_attention': sum(total_by_topic.values())
            }
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_35_animated_top10_topics: {str(e)}")
            return None
    
    def generate_all_plots(self, selected_plots=None):
        """Генерация всех графиков с прогресс-баром (35 графиков)"""
        self.all_figures = {}
        self.plot_data = {}
        self.errors = []
        self.warnings = []
        
        # Обновленный список всех функций графиков (35 графиков)
        plot_functions = [
            ("1_distribution", "1. Distribution of Attention", self.plot_1_distribution_attention),
            ("2_country_chord", "2. Country Collaboration Chord Diagram", self.plot_2_country_chord_diagram),
            ("3_internationality_linear", "3. Internationality vs Citations (Linear)", self.plot_3_internationality_vs_citations_linear),
            ("4_internationality_log", "4. Internationality vs Citations (Log Y Scale)", self.plot_4_internationality_vs_citations_log),
            ("5_journal_heatmap", "5. Journal-Year Heatmap", lambda: self.plot_5_journal_year_heatmap(15)),
            ("6_collab_linear", "6. Collaboration vs Citations (Linear)", self.plot_6_collaboration_vs_citations_linear),
            ("7_collab_log", "7. Collaboration vs Citations (Log Y Scale)", self.plot_7_collaboration_vs_citations_log),
            ("8_references_linear", "8. References vs Citations (Linear)", self.plot_8_references_vs_citations_linear),
            ("9_references_log", "9. References vs Citations (Log Y Scale)", self.plot_9_references_vs_citations_log),
            ("10_concepts", "10. Concepts Analysis", lambda: self.plot_10_concepts_analysis(30)),
            ("11_concept_cooccurrence", "11. Concept Co-occurrence", lambda: self.plot_11_concept_cooccurrence(15)),
            ("12_concept_influence", "12. Concept Influence Analysis", self.plot_12_concept_influence),
            ("13_temporal_evolution", "13. Temporal Evolution", self.plot_13_temporal_evolution),
            ("14_temporal_heatmap", "14. Temporal Heatmap", self.plot_14_temporal_heatmap),
            ("15_team_size", "15. Team Size Analysis", self.plot_15_team_size_analysis),
            ("16_correlation", "16. Correlation Matrix", self.plot_16_correlation_matrix),
            ("17_cr_vs_oa", "17. CR vs OA Comparison", self.plot_17_citation_sources_comparison),
            ("18_domain_citations", "18. Citations by Domain", self.plot_18_citation_by_domain),
            ("19_cumulative_influence", "19. Cumulative Influence", self.plot_19_cumulative_influence),
            ("20_references_attention", "20. References vs Attention", self.plot_20_references_vs_attention),
            ("21_journal_impact", "21. Journal Impact", self.plot_21_journal_impact),
            ("22_hierarchical_sankey", "22. Hierarchical Sankey Diagram", self.plot_22_hierarchical_sankey),
            ("23_mds", "23. Multidimensional Scaling", self.plot_23_multidimensional_scaling),
            ("24_concept_network", "24. Weighted Concept Network", self.plot_24_concept_network_weighted),
            ("25_term_temporal", "25. Term Temporal Density (Violin Plot)", lambda: self.plot_25_term_temporal_density('Topic', 15, 'first_year')),
            ("26_sunburst", "26. Sunburst Chart", self.plot_26_sunburst_chart),
            ("27_3d_bubble", "27. 3D Bubble Chart", self.plot_27_3d_bubble_chart),
            ("28_network", "28. Network: Authors-Journals-Topics", self.plot_28_network_authors_journals_topics),
            ("29_animated_bubble", "29. Animated Bubble Chart", self.plot_29_animated_bubble_chart),
            ("30_topic_chord", "30. Topic Chord Diagram", self.plot_30_topic_chord_diagram),
            ("31_active_elements_5y", "31. Active Elements (Last 5 Years)", lambda: self.plot_31_active_elements_last_5_years('Topic', 15)),
            ("32_animated_journal_heatmap", "32. Animated Journal Heatmap", lambda: self.plot_32_animated_journal_heatmap(15)),
            ("33_animated_country_chord", "33. Animated Country Chord Diagram", lambda: self.plot_33_animated_country_chord(4)),
            ("34_animated_worldmap", "34. Animated World Map", self.plot_34_animated_worldmap),
            ("35_animated_top10_topics", "35. Animated Top-10 Topics", self.plot_35_animated_top10_topics)
        ]
        
        # Если выбраны определенные графики
        if selected_plots:
            plot_functions = [pf for pf in plot_functions if pf[0] in selected_plots]
        
        successful_plots = 0
        progress_bar = st.progress(0)
        
        for i, (name, description, func) in enumerate(plot_functions):
            try:
                progress = (i / len(plot_functions))
                progress_bar.progress(progress)
                
                st.info(f"[{i+1}/{len(plot_functions)}] {description}...")
                fig = func()
                if fig is not None:
                    self.all_figures[name] = fig
                    successful_plots += 1
                    st.success(f"   ✅ Success")
                else:
                    st.warning(f"   ⚠️  Skipped (insufficient data)")
            except Exception as e:
                error_msg = str(e)[:100]
                self.log_error(f"Error generating {description}: {error_msg}")
                st.error(f"   ❌ Error: {error_msg}...")
        
        progress_bar.progress(1.0)
        return self.all_figures
    
    def create_excel_report(self):
        """Создает Excel файл с данными для всех графиков"""
        if not self.plot_data:
            st.warning("⚠️ No plot data available for Excel report")
            return None
        
        # Создаем Excel writer
        excel_buffer = io.BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # 1. Основная статистика
            if self.df_processed is not None:
                basic_stats = pd.DataFrame({
                    'Metric': [
                        'Total papers', 'Year range', 'Total mentions',
                        'Mean mentions per paper', 'Median mentions',
                        'Total authors', 'Unique countries', 'Unique journals',
                        'Max citations (max(CR, OA))', 'Max annual citations (max(annual CR, annual OA))'
                    ],
                    'Value': [
                        len(self.df_processed),
                        f"{int(self.df_processed['year'].min())}-{int(self.df_processed['year'].max())}",
                        int(self.df_processed['count'].sum()) if 'count' in self.df_processed.columns else 'N/A',
                        f"{self.df_processed['count'].mean():.2f}" if 'count' in self.df_processed.columns else 'N/A',
                        f"{self.df_processed['count'].median():.1f}" if 'count' in self.df_processed.columns else 'N/A',
                        'N/A',
                        self.df_processed['num_countries'].nunique() if 'num_countries' in self.df_processed.columns else 'N/A',
                        self.df_processed['Full journal Name'].nunique() if 'Full journal Name' in self.df_processed.columns else 'N/A',
                        f"{self.df_processed['max_citations'].mean():.1f}" if 'max_citations' in self.df_processed.columns else 'N/A',
                        f"{self.df_processed['max_annual_citations'].mean():.1f}" if 'max_annual_citations' in self.df_processed.columns else 'N/A'
                    ]
                })
                basic_stats.to_excel(writer, sheet_name='Basic_Statistics', index=False)
            
            # 2. Данные для каждого графика
            for plot_name, data in self.plot_data.items():
                sheet_name = f"Plot_{plot_name}"
                if len(sheet_name) > 31:
                    sheet_name = sheet_name[:31]
                
                try:
                    if isinstance(data, dict):
                        if all(isinstance(v, dict) for v in data.values()):
                            for sub_name, sub_data in data.items():
                                sub_sheet_name = f"{plot_name[:26]}_{sub_name}"[:31]
                                if isinstance(sub_data, dict):
                                    df = pd.DataFrame([sub_data])
                                else:
                                    df = pd.DataFrame(sub_data)
                                df.to_excel(writer, sheet_name=sub_sheet_name, index=False)
                        else:
                            df = pd.DataFrame([data])
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                    elif isinstance(data, list):
                        df = pd.DataFrame(data)
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                    else:
                        df = pd.DataFrame({'Value': [str(data)]})
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                except Exception as e:
                    st.warning(f"   ⚠️  Could not save data for {plot_name}: {str(e)[:50]}")
            
            # 3. Ошибки и предупреждения
            if self.errors:
                errors_df = pd.DataFrame(self.errors)
                errors_df.to_excel(writer, sheet_name='Errors', index=False)
            
            if self.warnings:
                warnings_df = pd.DataFrame(self.warnings)
                warnings_df.to_excel(writer, sheet_name='Warnings', index=False)
            
            # 4. Лист с объяснением терминов и формул
            self._add_terminology_sheet(writer)
        
        excel_buffer.seek(0)
        return excel_buffer
    
    def _add_terminology_sheet(self, writer):
        """Добавляет лист с объяснением терминов и формул"""
        terminology_data = {
            'Term': [
                'PC1 (Principal Component 1)',
                'PC2 (Principal Component 2)',
                'normalized_attention',
                'attention (count)',
                'total mentions',
                'article_age',
                'max_citations',
                'max_annual_citations',
                'Annual cit counts (CR)',
                'Annual cit counts (OA)',
                'num_countries',
                'num_affiliations',
                'Gini coefficient',
                'CCDF (Complementary Cumulative Distribution Function)',
                'Lorenz curve',
                'Spearman correlation',
                'PCA (Principal Component Analysis)'
            ],
            'Description': [
                'First principal component from PCA analysis - linear combination of predictors explaining maximum variance',
                'Second principal component from PCA analysis - orthogonal to PC1, explaining second most variance',
                'Attention normalized by article age: normalized_attention = count / article_age',
                'Local mentions count within the dataset (column "count" in original data)',
                'Sum of all attention (count) values across the dataset',
                'Age of article in years: article_age = current_year - publication_year',
                'Maximum citations between CR and OA: max(Citation counts (CR), Citation counts (OA))',
                'Maximum annual citations between CR and OA: max(Annual cit counts (CR), Annual cit counts (OA))',
                'Annual citation rate from Crossref: Citation counts (CR) / article_age',
                'Annual citation rate from OpenAlex: Citation counts (OA) / article_age',
                'Number of collaborating countries extracted from "countries {country 1; ... country last}"',
                'Number of affiliations extracted from "affiliations {aff 1; aff 2... aff last}"',
                'Measure of inequality (0 = perfect equality, 1 = maximum inequality) calculated from Lorenz curve',
                'Probability that a variable X is greater than or equal to x: P(X ≥ x) = 1 - CDF(x)',
                'Graphical representation of distribution inequality, plots cumulative % of population vs cumulative % of variable',
                'Non-parametric rank correlation coefficient measuring monotonic relationship',
                'Dimensionality reduction technique transforming correlated variables into uncorrelated principal components'
            ],
            'Formula/Calculation': [
                'PC1 = w₁₁*x₁ + w₁₂*x₂ + ... + w₁ₙ*xₙ where w are eigenvectors of covariance matrix',
                'PC2 = w₂₁*x₁ + w₂₂*x₂ + ... + w₂ₙ*xₙ orthogonal to PC1',
                'normalized_attention = count / max(1, current_year - year)',
                'Directly from "count" column in input data',
                'total_mentions = Σ count_i for all papers i',
                'article_age = 2024 - year (assuming current year 2024)',
                'max_citations = max(CR_citations, OA_citations)',
                'max_annual_citations = max(CR_annual, OA_annual)',
                'Annual_cit_CR = Citation counts (CR) / max(1, article_age)',
                'Annual_cit_OA = Citation counts (OA) / max(1, article_age)',
                'num_countries = len(split("countries {country 1; ... country last}", ";"))',
                'num_affiliations = len(split("affiliations {aff 1; aff 2... aff last}", ";"))',
                'G = (A / (A+B)) where A is area between Lorenz curve and equality line, B is area under Lorenz curve',
                'CCDF(x) = 1 - (number of observations ≤ x) / (total observations)',
                'Plot of (cumulative % of papers, cumulative % of mentions)',
                'ρ = 1 - (6Σd²)/(n(n²-1)) where d is difference between ranks',
                'PCA = eigendecomposition of covariance matrix XᵀX/(n-1)'
            ],
            'Data Source': [
                'Calculated from standardized predictor variables',
                'Calculated from standardized predictor variables',
                'Calculated: original "count" divided by calculated "article_age"',
                'Original data column "count"',
                'Calculated: sum of all "count" values',
                'Calculated: current year minus publication year',
                'Calculated: max of CR and OA citation counts',
                'Calculated: max of CR and OA annual citation rates',
                'Calculated: original "Citation counts (CR)" divided by article_age',
                'Calculated: original "Citation counts (OA)" divided by article_age',
                'Calculated from original "countries {country 1; ... country last}" column',
                'Calculated from original "affiliations {aff 1; aff 2... aff last}" column',
                'Calculated from sorted attention values',
                'Calculated from sorted attention values',
                'Calculated from sorted attention values',
                'Calculated using pandas corr(method="spearman")',
                'Calculated using sklearn.decomposition.PCA'
            ]
        }
        
        terminology_df = pd.DataFrame(terminology_data)
        terminology_df.to_excel(writer, sheet_name='Terminology_Formulas', index=False)
    
    def save_all_to_zip(self, include_excel=True):
        """Сохраняет все графики и отчеты в ZIP архив"""
        if not self.all_figures:
            st.error("❌ No plots to save!")
            return None
        
        # Создаем ZIP архив
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 1. Сохраняем графики
            for i, (name, fig) in enumerate(self.all_figures.items()):
                img_buffer = io.BytesIO()
                # Проверяем тип графика (plotly или matplotlib)
                if hasattr(fig, 'update_layout'):
                    # plotly фигура
                    import plotly.io as pio
                    pio.write_image(fig, img_buffer, format='png', width=1200, height=800, scale=2)
                else:
                    # matplotlib фигура
                    fig.savefig(img_buffer, format='png', dpi=300,
                              bbox_inches='tight', facecolor='white',
                              edgecolor='black')
                img_buffer.seek(0)
                
                filename = f"plot_{i+1:02d}_{name}.png"
                zip_file.writestr(filename, img_buffer.read())
                if not hasattr(fig, 'update_layout'):
                    plt.close(fig)
            
            # 2. Сохраняем Excel отчет
            if include_excel:
                excel_buffer = self.create_excel_report()
                if excel_buffer:
                    zip_file.writestr("plot_data.xlsx", excel_buffer.read())
            
            # 3. Сохраняем метаданные
            metadata = {
                'generated_date': datetime.now().isoformat(),
                'total_plots': len(self.all_figures),
                'visualization_settings': {
                    'show_regression_trends': self.show_regression_trends,
                    'top_countries_chord': self.top_countries_chord,
                    'top_fields_sankey': self.top_fields_sankey
                },
                'dataset_statistics': {
                    'total_rows': len(self.df_processed) if self.df_processed is not None else 0,
                    'year_range': f"{int(self.df_processed['year'].min())}-{int(self.df_processed['year'].max())}" if self.df_processed is not None and 'year' in self.df_processed.columns else 'N/A',
                    'total_mentions': int(self.df_processed['count'].sum()) if self.df_processed is not None and 'count' in self.df_processed.columns else 0
                },
                'errors_count': len(self.errors),
                'warnings_count': len(self.warnings)
            }
            
            zip_file.writestr('metadata.json', json.dumps(metadata, indent=2))
        
        zip_buffer.seek(0)
        return zip_buffer

# ============================================================================
# ФУНКЦИИ ДЛЯ STREAMLIT ИНТЕРФЕЙСА
# ============================================================================

def main():
    """Основная функция Streamlit приложения"""
    
    # Заголовок
    st.title("📊 Scientific Data Visualization Dashboard")
    st.markdown("---")
    
    # Определить all_plots здесь, в начале функции
    ALL_PLOTS = [
        ("1_distribution", "1. Distribution of Attention"),
        ("2_country_chord", "2. Country Collaboration Chord Diagram"),
        ("3_internationality_linear", "3. Internationality vs Citations (Linear)"),
        ("4_internationality_log", "4. Internationality vs Citations (Log Y Scale)"),
        ("5_journal_heatmap", "5. Journal-Year Heatmap"),
        ("6_collab_linear", "6. Collaboration vs Citations (Linear)"),
        ("7_collab_log", "7. Collaboration vs Citations (Log Y Scale)"),
        ("8_references_linear", "8. References vs Citations (Linear)"),
        ("9_references_log", "9. References vs Citations (Log Y Scale)"),
        ("10_concepts", "10. Concepts Analysis"),
        ("11_concept_cooccurrence", "11. Concept Co-occurrence"),
        ("12_concept_influence", "12. Concept Influence Analysis"),
        ("13_temporal_evolution", "13. Temporal Evolution"),
        ("14_temporal_heatmap", "14. Temporal Heatmap"),
        ("15_team_size", "15. Team Size Analysis"),
        ("16_correlation", "16. Correlation Matrix"),
        ("17_cr_vs_oa", "17. CR vs OA Comparison"),
        ("18_domain_citations", "18. Citations by Domain"),
        ("19_cumulative_influence", "19. Cumulative Influence"),
        ("20_references_attention", "20. References vs Attention"),
        ("21_journal_impact", "21. Journal Impact"),
        ("22_hierarchical_sankey", "22. Hierarchical Sankey Diagram"),
        ("23_mds", "23. Multidimensional Scaling"),
        ("24_concept_network", "24. Weighted Concept Network"),
        ("25_term_temporal", "25. Term Temporal Density (Violin Plot)"),
        ("26_sunburst", "26. Sunburst Chart"),
        ("27_3d_bubble", "27. 3D Bubble Chart"),
        ("28_network", "28. Network: Authors-Journals-Topics"),
        ("29_animated_bubble", "29. Animated Bubble Chart"),
        ("30_topic_chord", "30. Topic Chord Diagram"),
        ("31_active_elements_5y", "31. Active Elements (Last 5 Years)"),
        ("32_animated_journal_heatmap", "32. Animated Journal Heatmap"),
        ("33_animated_country_chord", "33. Animated Country Chord Diagram"),
        ("34_animated_worldmap", "34. Animated World Map"),
        ("35_animated_top10_topics", "35. Animated Top-10 Topics")
    ]
    
    # Инициализация состояния сессии
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ScientificDataAnalyzer()
    
    if 'plots_generated' not in st.session_state:
        st.session_state.plots_generated = False
    
    if 'selected_plots' not in st.session_state:
        st.session_state.selected_plots = [plot[0] for plot in ALL_PLOTS]
    
    # Боковая панель
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Меню навигации
        selected = option_menu(
            menu_title="Меню",
            options=["📋 Загрузка данных", "📊 Визуализация", "📥 Скачивание"],
            icons=["upload", "bar-chart", "download"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "#f0f2f6"},
                "icon": {"color": "orange", "font-size": "18px"},
                "nav-link": {"font-size": "14px", "text-align": "left", "margin": "2px"},
                "nav-link-selected": {"background-color": "#2E86AB"},
            }
        )
        
        st.markdown("---")
        
        # Настройки визуализации (применяются в реальном времени при генерации)
        st.subheader("🎨 Настройки визуализации")
        
        show_regression = st.checkbox(
            "📈 Показывать регрессионные тренды",
            value=st.session_state.analyzer.show_regression_trends if hasattr(st.session_state.analyzer, 'show_regression_trends') else True,
            help="Отображать линии регрессии на графиках (где применимо)"
        )
        
        top_countries = st.slider(
            "🌍 Количество стран в хордовой диаграмме",
            min_value=10,
            max_value=50,
            value=st.session_state.analyzer.top_countries_chord,
            step=5,
            help="Выберите количество топ стран для отображения в хордовой диаграмме коллабораций"
        )
        
        top_fields = st.slider(
            "📚 Количество полей (Field) в Sankey диаграмме",
            min_value=5,
            max_value=20,
            value=st.session_state.analyzer.top_fields_sankey,
            step=1,
            help="Ограничьте количество полей для читаемости Sankey диаграммы"
        )
        
        # Применяем настройки к анализатору
        st.session_state.analyzer.update_visualization_settings(
            show_regression_trends=show_regression,
            top_countries_chord=top_countries,
            top_fields_sankey=top_fields
        )
        
        st.markdown("---")
        
        # Настройки для графика временной плотности
        st.subheader("📈 Настройки Temporal Density Plot")
        
        hierarchy_level = st.selectbox(
            "Уровень иерархии",
            options=["Topic", "Subfield", "Field", "Domain", "Concepts"],
            index=0,
            help="Выберите уровень детализации для анализа временной активности"
        )
        
        top_percent = st.slider(
            "Процент топ терминов для отображения",
            min_value=5,
            max_value=30,
            value=15,
            step=5,
            help="Выберите процент наиболее активных терминов для отображения"
        )
        
        sort_by = st.selectbox(
            "Сортировка терминов",
            options=["first_year", "total_attention", "activity_span", "peak_density"],
            index=0,
            help="Критерий сортировки для выбора топ терминов"
        )
        
        # Настройки для графика активных элементов за 5 лет
        st.subheader("📈 Настройки Active Elements (5 Years)")
        
        active_hierarchy_level = st.selectbox(
            "Уровень для анализа активности за 5 лет",
            options=["Topic", "Subfield", "Field", "Domain", "Concepts"],
            index=0,
            key="active_hierarchy",
            help="Выберите уровень иерархии для анализа наиболее активных элементов за последние 5 лет"
        )
        
        top_n_active = st.slider(
            "Количество топ элементов для отображения",
            min_value=5,
            max_value=30,
            value=15,
            step=5,
            key="top_n_active",
            help="Выберите количество наиболее активных элементов для отображения"
        )
        
        st.markdown("---")
        
        # Пример данных
        st.subheader("📚 Пример данных")
        sample_data = """doi	publication_date	Title	authors	ORCID ID 1; ORCID ID 2... ORCID ID last	author count	affiliations {aff 1; aff 2... aff last}	countries {country 1; ... country last}	Full journal Name	year	Volume	Pages (or article number)	Citation counts (CR)	Citation counts (OA)	Annual cit counts (CR)	Annual cit counts (OA)	references_count	count	Topic	Subfield	Field	Domain	Concepts
10.1021/acs.chemrev.6b00284	2016-11-09	Strategies for Carbon and Sulfur Tolerant Solid Oxide Fuel Cell Materials, Incorporating Lessons from Heterogeneous Catalysis	Paul Boldrin; Enrique Ruiz-Trejo; Joshua Mermelstein; José Miguel Bermúdez Menéndez; Tomás Ramı́rez Reina; Nigel P. Brandon	https://orcid.org/0000-0003-0058-6876; https://orcid.org/0000-0001-5560-5750; https://orcid.org/0000-0001-7211-2958; https://orcid.org/0000-0001-9693-5107; https://orcid.org/0000-0003-2230-8666	6	University of Surrey; Imperial College London; Boeing (United States)	US; GB	Chemical Reviews	2016	116	13633-13684	289	296	26.27	26.91	465	5	Advancements in Solid Oxide Fuel Cells	Chemistry	Carbon fibers	Catalysis	Sulfur; Chemistry; Carbon fibers; Catalysis; Oxide; Solid oxide fuel cell; Fuel cells; Nanotechnology; Environmental chemistry; Chemical engineering; Organic chemistry; Materials science; Engineering; Composite number; Physical chemistry; Composite material; Anode; Electrode
10.1126/science.aab3987	2015-07-23	Readily processed protonic ceramic fuel cells with high performance at low temperatures	Chuancheng Duan; Jianhua Tong; Meng Shang; Stefan Nikodemski; Michael Sanders; Sandrine Ricote; Ali Almansoori; Ryan O'Hayre	https://orcid.org/0000-0002-1826-1415; https://orcid.org/0000-0002-0684-1658; https://orcid.org/0000-0001-6366-5219; https://orcid.org/0000-0001-7565-0284; https://orcid.org/0000-0002-0789-5105; https://orcid.org/0000-0003-3762-3052	8	American Petroleum Institute; Colorado School of Mines	US	Science	2015	349	1321-1326	1325	1352	110.42	112.67	91	5	Advancements in Solid Oxide Fuel Cells	Oxide	Fuel cells	Materials science	Ceramic; Oxide; Fuel cells; Materials science; Methane; Electrolyte; Chemical engineering; Cathode; Ion; Solid oxide fuel cell; Chemistry; Composite material; Electrode; Metallurgy; Organic chemistry; Engineering; Physical chemistry"""
        
        if st.button("📋 Загрузить пример данных", use_container_width=True):
            st.session_state.sample_data_loaded = sample_data
            st.rerun()
        
        st.markdown("---")
        st.info("""
        **Инструкция:**
        1. Вставьте данные в формате TSV
        2. Нажмите "Загрузить данные"
        3. Выберите графики для генерации
        4. Скачайте результаты
        """)
    
    # Основное содержимое
    if selected == "📋 Загрузка данных":
        st.header("📋 Загрузка данных")
        
        # Поле для ввода данных
        data_input = st.text_area(
            "Вставьте данные в формате TSV (табуляция между колонками)",
            value=st.session_state.get('sample_data_loaded', ''),
            height=300,
            help="Скопируйте и вставьте данные из Excel/Google Sheets. Первая строка должна содержать заголовки колонок."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Загрузить данные", type="primary", use_container_width=True):
                if data_input.strip():
                    with st.spinner("Обработка данных..."):
                        st.session_state.analyzer.parse_data(data_input)
                        st.success("✅ Данные успешно загружены!")
                        st.session_state.plots_generated = False
                else:
                    st.error("❌ Пожалуйста, введите данные")
        
        with col2:
            if st.button("🗑️ Очистить", use_container_width=True):
                st.session_state.sample_data_loaded = ''
                st.rerun()
        
        # Показать информацию о данных
        if st.session_state.analyzer.df_processed is not None:
            st.subheader("📊 Информация о данных")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Статей", len(st.session_state.analyzer.df_processed))
            with col2:
                if 'year' in st.session_state.analyzer.df_processed.columns:
                    year_min = int(st.session_state.analyzer.df_processed['year'].min())
                    year_max = int(st.session_state.analyzer.df_processed['year'].max())
                    st.metric("Годы", f"{year_min}-{year_max}")
            with col3:
                if 'count' in st.session_state.analyzer.df_processed.columns:
                    total_mentions = int(st.session_state.analyzer.df_processed['count'].sum())
                    st.metric("Всего упоминаний", f"{total_mentions:,}")
            with col4:
                if 'max_citations' in st.session_state.analyzer.df_processed.columns:
                    mean_max_cit = st.session_state.analyzer.df_processed['max_citations'].mean()
                    st.metric("Ср. макс. цитирований", f"{mean_max_cit:.1f}")
            
            # Показать таблицу
            with st.expander("👁️ Просмотреть данные"):
                st.dataframe(st.session_state.analyzer.df_processed.head(10))
    
    elif selected == "📊 Визуализация":
        st.header("📊 Визуализация данных")
        
        if st.session_state.analyzer.df_processed is None:
            st.warning("⚠️ Сначала загрузите данные в разделе 'Загрузка данных'")
            return
        
        # Дополнительные настройки для временных графиков
        st.subheader("🎛️ Дополнительные настройки для временных графиков")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            hierarchy_level_viz = st.selectbox(
                "Уровень иерархии (для Temporal Density & Active Elements)",
                options=["Topic", "Subfield", "Field", "Domain", "Concepts"],
                index=0,
                key="viz_hierarchy"
            )
        
        with col2:
            top_percent_viz = st.slider(
                "Топ % терминов (Temporal Density)",
                min_value=5, max_value=30, value=15, step=5, key="viz_top_percent"
            )
        
        with col3:
            sort_by_viz = st.selectbox(
                "Сортировка (Temporal Density)",
                options=["first_year", "total_attention", "activity_span", "peak_density"],
                index=0, key="viz_sort"
            )
        
        st.markdown("---")
        
        # Выбор графиков
        st.subheader("🎯 Выберите графики для генерации")
        
        # Выбор всех графиков
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Выбрать все", use_container_width=True):
                st.session_state.selected_plots = [plot[0] for plot in ALL_PLOTS]
                st.rerun()
        with col2:
            if st.button("❌ Очистить выбор", use_container_width=True):
                st.session_state.selected_plots = []
                st.rerun()
        
        # Чекбоксы для выбора графиков
        st.markdown("### Доступные графики")
        
        # Группируем графики по категориям
        categories = {
            "📈 Основные распределения": ["1_distribution", "19_cumulative_influence"],
            "🌍 Международное сотрудничество": ["2_country_chord", "3_internationality_linear", "4_internationality_log", "6_collab_linear", "7_collab_log"],
            "📚 Журналы и публикации": ["5_journal_heatmap", "21_journal_impact"],
            "🔗 Ссылки и цитирования": ["8_references_linear", "9_references_log", "17_cr_vs_oa", "20_references_attention"],
            "🏷️ Концепты и темы": ["10_concepts", "11_concept_cooccurrence", "12_concept_influence", "24_concept_network", "30_topic_chord"],
            "⏰ Временной анализ": ["13_temporal_evolution", "14_temporal_heatmap", "25_term_temporal", "31_active_elements_5y"],
            "👥 Команды и организации": ["15_team_size", "28_network"],
            "📊 Анализ метрик": ["16_correlation", "18_domain_citations", "23_mds"],
            "🏛️ Иерархическая структура": ["22_hierarchical_sankey", "26_sunburst"],
            "🎯 Многомерные графики": ["27_3d_bubble"],
            "🎬 Анимированные графики": ["29_animated_bubble", "32_animated_journal_heatmap", "33_animated_country_chord", "34_animated_worldmap", "35_animated_top10_topics"]
        }
        
        for category, plot_ids in categories.items():
            with st.expander(category):
                for plot_id in plot_ids:
                    plot_name = next(name for pid, name in ALL_PLOTS if pid == plot_id)
                    if st.checkbox(plot_name, 
                                 value=plot_id in st.session_state.selected_plots,
                                 key=f"checkbox_{plot_id}"):
                        if plot_id not in st.session_state.selected_plots:
                            st.session_state.selected_plots.append(plot_id)
                    else:
                        if plot_id in st.session_state.selected_plots:
                            st.session_state.selected_plots.remove(plot_id)
        
        # Кнопки генерации
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Сгенерировать выбранные графики", type="primary", use_container_width=True):
                if not st.session_state.selected_plots:
                    st.error("❌ Выберите хотя бы один график")
                else:
                    # Обновляем настройки для временных графиков
                    st.session_state.analyzer.plot_25_term_temporal_density = lambda: st.session_state.analyzer.plot_25_term_temporal_density(
                        hierarchy_level_viz, top_percent_viz, sort_by_viz
                    )
                    st.session_state.analyzer.plot_31_active_elements_last_5_years = lambda: st.session_state.analyzer.plot_31_active_elements_last_5_years(
                        hierarchy_level_viz, top_n_active
                    )
                    
                    with st.spinner("Генерация графиков..."):
                        st.session_state.analyzer.generate_all_plots(st.session_state.selected_plots)
                        st.session_state.plots_generated = True
                        st.success(f"✅ Сгенерировано {len(st.session_state.analyzer.all_figures)} графиков!")
        
        with col2:
            if st.button("🎯 Сгенерировать все графики", use_container_width=True):
                st.session_state.selected_plots = [plot[0] for plot in ALL_PLOTS]
                
                # Обновляем настройки для временных графиков
                st.session_state.analyzer.plot_25_term_temporal_density = lambda: st.session_state.analyzer.plot_25_term_temporal_density(
                    hierarchy_level_viz, top_percent_viz, sort_by_viz
                )
                st.session_state.analyzer.plot_31_active_elements_last_5_years = lambda: st.session_state.analyzer.plot_31_active_elements_last_5_years(
                    hierarchy_level_viz, top_n_active
                )
                
                with st.spinner("Генерация всех графиков..."):
                    st.session_state.analyzer.generate_all_plots()
                    st.session_state.plots_generated = True
                    st.success(f"✅ Сгенерировано {len(st.session_state.analyzer.all_figures)} графиков!")
        
        # Показать сгенерированные графики
        if st.session_state.plots_generated and st.session_state.analyzer.all_figures:
            st.markdown("---")
            st.subheader("📈 Результаты визуализации")
            
            # Навигация по графикам
            plot_names = list(st.session_state.analyzer.all_figures.keys())
            
            if len(plot_names) > 0:
                # Селектор для выбора графика
                selected_plot = st.selectbox(
                    "Выберите график для просмотра",
                    options=plot_names,
                    format_func=lambda x: next(name for pid, name in ALL_PLOTS if pid == x)
                )
                
                # Показать выбранный график
                if selected_plot in st.session_state.analyzer.all_figures:
                    fig = st.session_state.analyzer.all_figures[selected_plot]
                    
                    # Проверяем тип графика (plotly или matplotlib)
                    if hasattr(fig, 'update_layout'):
                        # Это plotly фигура
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Кнопка для скачивания анимации как HTML (для анимированных графиков)
                        if selected_plot in ["29_animated_bubble", "32_animated_journal_heatmap", 
                                            "33_animated_country_chord", "34_animated_worldmap", 
                                            "35_animated_top10_topics"]:
                            import plotly.io as pio
                            html_str = pio.to_html(fig, include_plotlyjs='cdn', full_html=True)
                            st.download_button(
                                label="📥 Скачать анимацию как HTML",
                                data=html_str,
                                file_name=f"{selected_plot}_animation.html",
                                mime="text/html",
                                key=f"html_download_{selected_plot}"
                            )
                    else:
                        # Это matplotlib фигура
                        st.pyplot(fig)
                    
                    # Информация о графике
                    plot_name = next(name for pid, name in ALL_PLOTS if pid == selected_plot)
                    st.info(f"**{plot_name}**")
                    
                    # Кнопки для навигации
                    col1, col2, col3 = st.columns(3)
                    current_index = plot_names.index(selected_plot)
                    
                    with col1:
                        if current_index > 0:
                            if st.button("◀️ Предыдущий"):
                                st.session_state.current_plot_index = current_index - 1
                                st.rerun()
                    
                    with col2:
                        st.write(f"График {current_index + 1} из {len(plot_names)}")
                    
                    with col3:
                        if current_index < len(plot_names) - 1:
                            if st.button("Следующий ▶️"):
                                st.session_state.current_plot_index = current_index + 1
                                st.rerun()
    
    elif selected == "📥 Скачивание":
        st.header("📥 Скачивание результатов")
        
        if not st.session_state.plots_generated or not st.session_state.analyzer.all_figures:
            st.warning("⚠️ Сначала сгенерируйте графики в разделе 'Визуализация'")
            return
        
        st.success(f"✅ Доступно для скачивания: {len(st.session_state.analyzer.all_figures)} графиков")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Скачать отдельные графики
            st.subheader("📸 Отдельные графики")

            plot_options = {}
            for pid in st.session_state.analyzer.all_figures.keys():
                for plot_id, name in ALL_PLOTS:
                    if plot_id == pid:
                        plot_options[name] = pid
                        break
            
            selected_plot_name = st.selectbox("Выберите график", options=list(plot_options.keys()))
            
            if selected_plot_name:
                plot_id = plot_options[selected_plot_name]
                fig = st.session_state.analyzer.all_figures[plot_id]
                
                # Сохранить график в буфер
                if hasattr(fig, 'update_layout'):
                    # plotly фигура
                    import plotly.io as pio
                    
                    # Для анимированных графиков предлагаем HTML
                    if plot_id in ["29_animated_bubble", "32_animated_journal_heatmap", 
                                   "33_animated_country_chord", "34_animated_worldmap", 
                                   "35_animated_top10_topics"]:
                        html_str = pio.to_html(fig, include_plotlyjs='cdn', full_html=True)
                        st.download_button(
                            label=f"📥 Скачать {selected_plot_name} как HTML",
                            data=html_str,
                            file_name=f"plot_{plot_id}_animation.html",
                            mime="text/html",
                            use_container_width=True
                        )
                    
                    # Также предлагаем PNG
                    img_buffer = io.BytesIO()
                    pio.write_image(fig, img_buffer, format='png', width=1200, height=800, scale=2)
                    img_buffer.seek(0)
                    st.download_button(
                        label=f"📥 Скачать {selected_plot_name} как PNG",
                        data=img_buffer,
                        file_name=f"plot_{plot_id}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                else:
                    # matplotlib фигура
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                    img_buffer.seek(0)
                    st.download_button(
                        label=f"📥 Скачать {selected_plot_name}",
                        data=img_buffer,
                        file_name=f"plot_{plot_id}.png",
                        mime="image/png",
                        use_container_width=True
                    )
        
        with col2:
            # Скачать все в ZIP
            st.subheader("📦 Все результаты")
            
            if st.button("📥 Скачать ZIP архив", type="primary", use_container_width=True):
                with st.spinner("Создание ZIP архива..."):
                    zip_buffer = st.session_state.analyzer.save_all_to_zip(include_excel=True)
                    
                    if zip_buffer:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"scientific_analysis_{timestamp}.zip"
                        
                        st.download_button(
                            label="⬇️ Скачать ZIP",
                            data=zip_buffer,
                            file_name=filename,
                            mime="application/zip",
                            use_container_width=True
                        )
        
        # Статистика
        st.markdown("---")
        st.subheader("📊 Статистика")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Всего графиков", len(st.session_state.analyzer.all_figures))
        with col2:
            st.metric("Ошибок", len(st.session_state.analyzer.errors))
        with col3:
            st.metric("Предупреждений", len(st.session_state.analyzer.warnings))
        
        # Показать ошибки и предупреждения
        if st.session_state.analyzer.errors:
            with st.expander("❌ Ошибки"):
                for error in st.session_state.analyzer.errors:
                    st.error(f"{error['timestamp']}: {error['message']}")
        
        if st.session_state.analyzer.warnings:
            with st.expander("⚠️ Предупреждения"):
                for warning in st.session_state.analyzer.warnings:
                    st.warning(f"{warning['timestamp']}: {warning['message']}")

# Запуск приложения
if __name__ == "__main__":
    main()
