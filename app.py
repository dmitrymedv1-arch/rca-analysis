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
        
        # Расчет максимальных цитирований между CR и OA
        if 'Citation counts (CR)' in df_processed.columns and 'Citation counts (OA)' in df_processed.columns:
            df_processed['max_citations'] = df_processed[['Citation counts (CR)', 'Citation counts (OA)']].max(axis=1)
            df_processed['max_annual_citations'] = df_processed[['Annual cit counts (CR)', 'Annual cit counts (OA)']].max(axis=1)
        
        # Количество стран и аффилиаций
        if 'countries_list' in df_processed.columns:
            df_processed['num_countries'] = df_processed['countries_list'].apply(len)
        
        if 'affiliations_list' in df_processed.columns:
            df_processed['num_affiliations'] = df_processed['affiliations_list'].apply(len)
        
        st.success("✅ Data preprocessing complete")
        return df_processed
    
    # ============================================================================
    # ФУНКЦИИ ДЛЯ ПОСТРОЕНИЯ ГРАФИКОВ (23 ВИДА)
    # ============================================================================
    
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
    
    def plot_2_country_collaboration_network(self):
        """2. Сеть коллабораций между странами"""
        try:
            if 'countries_list' not in self.df_processed.columns:
                return None
            
            # Создаем граф
            G = nx.Graph()
            country_pairs = []
            
            for idx, row in self.df_processed.iterrows():
                if isinstance(row['countries_list'], list) and len(row['countries_list']) >= 2:
                    countries = [c.strip().upper() for c in row['countries_list']]
                    weight = row.get('count', 1)
                    
                    # Добавляем узлы
                    for country in countries:
                        if not G.has_node(country):
                            G.add_node(country, weight=0, papers=0)
                        G.nodes[country]['weight'] += weight
                        G.nodes[country]['papers'] += 1
                    
                    # Добавляем ребра между всеми парами
                    for i in range(len(countries)):
                        for j in range(i+1, len(countries)):
                            if G.has_edge(countries[i], countries[j]):
                                G[countries[i]][countries[j]]['weight'] += weight
                                G[countries[i]][countries[j]]['papers'] += 1
                            else:
                                G.add_edge(countries[i], countries[j], weight=weight, papers=1)
                            
                            country_pairs.append({
                                'country1': countries[i],
                                'country2': countries[j],
                                'weight': weight,
                                'paper_id': idx
                            })
            
            if len(G.nodes()) < 3:
                self.log_warning("Insufficient data for country network")
                return None
            
            # Сохраняем данные
            self.plot_data['2_country_network'] = {
                'nodes': [{'country': node, 'weight': G.nodes[node]['weight'], 
                          'papers': G.nodes[node]['papers']} for node in G.nodes()],
                'edges': [{'country1': u, 'country2': v, 'weight': G[u][v]['weight'],
                          'papers': G[u][v]['papers']} for u, v in G.edges()]
            }
            
            # Визуализация
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Фильтруем слабые связи
            min_weight = np.percentile([d['weight'] for u, v, d in G.edges(data=True)], 50)
            edges_to_keep = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] >= min_weight]
            H = G.edge_subgraph([u for u, v in edges_to_keep] + [v for u, v in edges_to_keep])
            
            if len(H.nodes()) == 0:
                H = G
            
            # Позиционирование
            pos = nx.spring_layout(H, k=2, seed=42)
            
            # Размер узлов по весу
            node_sizes = [H.nodes[n]['weight'] * 0.5 + 500 for n in H.nodes()]
            node_colors = [H.nodes[n]['papers'] for n in H.nodes()]
            
            # Рисуем граф с градиентной цветовой схемой (от светлого к темному)
            nodes = nx.draw_networkx_nodes(H, pos, node_size=node_sizes,
                                          node_color=node_colors, cmap='Blues',
                                          alpha=0.8, edgecolors='black', linewidths=1.5, ax=ax)
            
            # Ребра
            if H.edges():
                edge_weights = [H[u][v]['weight'] * 0.3 for u, v in H.edges()]
                edges = nx.draw_networkx_edges(H, pos, width=edge_weights, alpha=0.7,
                                              edge_color='gray', style='solid', ax=ax)
            
            # Подписи
            nx.draw_networkx_labels(H, pos, font_size=10, font_weight='bold', ax=ax)
            
            ax.set_title('Country Collaboration Network', fontweight='bold', fontsize=16, pad=20)
            ax.axis('off')
            
            # Цветовая шкала
            sm = plt.cm.ScalarMappable(cmap='Blues',
                                      norm=plt.Normalize(vmin=min(node_colors), 
                                                       vmax=max(node_colors)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Number of Papers', fontweight='bold', fontsize=10)
            
            # Статистика
            stats_text = f"Nodes: {len(G.nodes())} | Edges: {len(G.edges())}"
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_2_country_collaboration_network: {str(e)}")
            return None
    
    def plot_3_internationality_vs_citations(self):
        """3. Международность vs Цитируемость"""
        try:
            required_cols = ['num_countries', 'Citation counts (CR)', 'author count']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) < 10:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            scatter = ax.scatter(valid_data['num_countries'],
                               valid_data['Citation counts (CR)'],
                               c=valid_data.get('Annual cit counts (CR)', 1),
                               s=valid_data['author count'] * 20,
                               alpha=0.7,
                               cmap='viridis',
                               edgecolors='black',
                               linewidth=0.5)
            
            ax.set_xlabel('Number of Collaborating Countries', fontweight='bold')
            ax.set_ylabel('Total Citations (CR)', fontweight='bold')
            ax.set_title('International Collaboration vs Citation Impact',
                        fontweight='bold', fontsize=16)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Annual Citation Rate (CR)', fontweight='bold')
            
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
            self.plot_data['3_internationality_vs_citations'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_3_internationality_vs_citations: {str(e)}")
            return None
            
    def plot_4_journal_year_heatmap(self, top_journals=15):
        """4. Тепловая карта: Журнал vs Год"""
        try:
            required_cols = ['Full journal Name', 'year', 'Annual cit counts (CR)']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            # Выбираем топ журналов
            journal_counts = self.df_processed['Full journal Name'].value_counts()
            top_journals_list = journal_counts.head(top_journals).index.tolist()
            
            heatmap_data = self.df_processed[self.df_processed['Full journal Name'].isin(top_journals_list)].copy()
            if len(heatmap_data) == 0:
                return None
            
            pivot_table = heatmap_data.pivot_table(
                values='Annual cit counts (CR)',
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
            self.plot_data['4_journal_year_heatmap'] = {
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
            cbar.ax.set_ylabel('Average Annual Citations (CR)', rotation=90, fontsize=12)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_4_journal_year_heatmap: {str(e)}")
            return None
    
    def plot_5_collaboration_vs_citations_linear(self):
        """5. Зависимость цитирований от коллабораций (Линейная шкала)"""
        try:
            required_cols = ['author count', 'num_affiliations', 'num_countries', 'max_citations']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
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
                
                # Линейная регрессия
                if len(valid_data) > 10:
                    x = valid_data[metric].values
                    y = valid_data['max_citations'].values
                    mask = ~(np.isnan(x) | np.isnan(y))
                    if mask.sum() > 10:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                        x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                        y_line = intercept + slope * x_line
                        ax.plot(x_line, y_line, 'r--', linewidth=2, 
                               label=f'r = {r_value:.3f}, p = {p_value:.3f}')
                
                ax.set_xlabel(label, fontweight='bold')
                ax.set_ylabel('Maximum Citations (max(CR, OA))', fontweight='bold')
                ax.set_title(f'{label} vs Citations', fontweight='bold')
                ax.legend(loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                if idx < 2:
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar_label = 'Number of Countries' if metric != 'num_countries' else 'Number of Authors'
                    cbar.set_label(cbar_label, fontweight='bold')
            
            # Сохраняем данные
            self.plot_data['5_collaboration_vs_citations_linear'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_5_collaboration_vs_citations_linear: {str(e)}")
            return None
    
    def plot_6_collaboration_vs_citations_log(self):
        """6. Зависимость цитирований от коллабораций (логарифмическая шкала только для Y)"""
        try:
            required_cols = ['author count', 'num_affiliations', 'num_countries', 'max_citations']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
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
                # Фильтруем данные > 0 для логарифмической шкалы по Y
                plot_data = valid_data[valid_data['max_citations'] > 0].copy()
                if len(plot_data) < 10:
                    continue
                
                # Создаем scatter plot
                scatter = ax.scatter(plot_data[metric],
                                   plot_data['max_citations'],
                                   c=plot_data['num_countries'] if metric != 'num_countries' else plot_data['author count'],
                                   s=plot_data['author count'] * 10,
                                   alpha=0.6,
                                   cmap='viridis',
                                   edgecolors='black',
                                   linewidth=0.5)
                
                # Экспоненциальная регрессия (log Y)
                if len(plot_data) > 10:
                    x = plot_data[metric].values
                    log_y = np.log(plot_data['max_citations'].values)
                    
                    # Убираем бесконечные значения
                    mask = np.isfinite(log_y)
                    if mask.sum() > 10:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], log_y[mask])
                        x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                        y_line = np.exp(intercept + slope * x_line)
                        ax.plot(x_line, y_line, 'r--', linewidth=2, 
                               label=f'exponential: y ∝ exp({slope:.3f}x), r = {r_value:.3f}')
                
                ax.set_xlabel(label, fontweight='bold')
                ax.set_ylabel('Maximum Citations (max(CR, OA)) - Log Scale', fontweight='bold')
                ax.set_title(f'{label} vs Citations (Log Y Scale)', fontweight='bold')
                
                # Устанавливаем логарифмическую шкалу только для оси Y
                ax.set_yscale('log')
                # Ось X остается линейной
                
                ax.legend(loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.3, which='both')
                
                # Добавляем цветовую шкалу
                cbar = plt.colorbar(scatter, ax=ax)
                if metric != 'num_countries':
                    cbar.set_label('Number of Countries', fontweight='bold', fontsize=10)
                else:
                    cbar.set_label('Number of Authors', fontweight='bold', fontsize=10)
                
                # Добавляем легенду для размера пузырьков
                from matplotlib.lines import Line2D
                legend_elements = []
                
                # Определяем размеры пузырьков для легенды
                size_values = [2, 5, 10, 15]  # количество авторов
                for n_authors in size_values:
                    if n_authors <= plot_data['author count'].max():
                        marker_size = n_authors * 10  # соответствие размеру в scatter
                        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                                     markerfacecolor='gray', 
                                                     markersize=np.sqrt(marker_size),
                                                     label=f'{n_authors} authors'))
                
                if legend_elements:
                    ax.legend(handles=legend_elements, title='Bubble size = Team size',
                             loc='lower right', fontsize=8, title_fontsize=9)
            
            # Сохраняем данные
            self.plot_data['6_collaboration_vs_citations_log'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_6_collaboration_vs_citations_log: {str(e)}")
            return None
    
    def plot_6_1_bubble_chart(self):
        """6.1 Пузырьковая диаграмма: References vs Citations (линейная шкала)"""
        try:
            required_cols = ['references_count', 'Citation counts (CR)', 'author count', 'num_countries']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) < 10:
                return None
            
            fig, ax = plt.subplots(figsize=(14, 10))
            
            scatter = ax.scatter(valid_data['references_count'],
                               valid_data['Citation counts (CR)'],
                               s=valid_data['author count'] * 40,
                               c=valid_data['num_countries'],
                               cmap='coolwarm',
                               alpha=0.7,
                               edgecolors='black',
                               linewidth=0.5)
            
            ax.set_xlabel('Number of References', fontweight='bold')
            ax.set_ylabel('Total Citations (CR) - Linear Scale', fontweight='bold')
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
            self.plot_data['6_1_bubble_chart'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_6_1_bubble_chart: {str(e)}")
            return None
    
    def plot_6_2_bubble_chart(self):
        """6.2 Пузырьковая диаграмма: References vs Citations (логарифмическая шкала)"""
        try:
            required_cols = ['references_count', 'Citation counts (CR)', 'author count', 'num_countries']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) < 10:
                return None
            
            fig, ax = plt.subplots(figsize=(14, 10))
            
            scatter = ax.scatter(valid_data['references_count'],
                               valid_data['Citation counts (CR)'],
                               s=valid_data['author count'] * 40,
                               c=valid_data['num_countries'],
                               cmap='coolwarm',
                               alpha=0.7,
                               edgecolors='black',
                               linewidth=0.5)
            
            ax.set_xlabel('Number of References', fontweight='bold')
            ax.set_ylabel('Total Citations (CR) - Log Scale', fontweight='bold')
            ax.set_title('Research Breadth vs Impact (Logarithmic Scale)',
                        fontweight='bold', fontsize=16)
            
            # Устанавливаем логарифмическую шкалу для оси Y
            ax.set_yscale('log')
            
            # Убедимся, что значения больше 0 для логарифмической шкалы
            min_citation = valid_data['Citation counts (CR)'].min()
            if min_citation <= 0:
                valid_log_data = valid_data[valid_data['Citation counts (CR)'] > 0].copy()
                if len(valid_log_data) > 0:
                    scatter = ax.scatter(valid_log_data['references_count'],
                                       valid_log_data['Citation counts (CR)'],
                                       s=valid_log_data['author count'] * 40,
                                       c=valid_log_data['num_countries'],
                                       cmap='coolwarm',
                                       alpha=0.7,
                                       edgecolors='black',
                                       linewidth=0.5)
            
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
            self.plot_data['6_2_bubble_chart'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_6_2_bubble_chart: {str(e)}")
            return None
    
    def plot_7_concepts_analysis(self, top_n=30):
        """7. Анализ концептов (30 топ концептов)"""
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
            self.plot_data['7_concepts_analysis'] = {
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
            self.log_error(f"Error in plot_7_concepts_analysis: {str(e)}")
            return None
    
    def plot_8_concept_cooccurrence(self, top_n=15):
        """8. Матрица совместной встречаемости концептов"""
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
            self.plot_data['8_concept_cooccurrence'] = {
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
            self.log_error(f"Error in plot_8_concept_cooccurrence: {str(e)}")
            return None
    
    def plot_9_concept_influence(self):
        """9. Влияние ключевых концептов"""
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
            self.plot_data['9_concept_influence'] = concept_stats.reset_index().to_dict('records')
            
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
            self.log_error(f"Error in plot_9_concept_influence: {str(e)}")
            return None
    
    def plot_10_temporal_evolution(self):
        """10. Эволюция публикационной активности и влияния во времени"""
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
            self.plot_data['10_temporal_evolution'] = year_stats.reset_index().to_dict('records')
            
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
                           label='Total Citations')
            ax2.set_ylabel('Total Citations', fontweight='bold', color='darkorange')
            ax2.tick_params(axis='y', labelcolor='darkorange')
            
            # Линия: средние цитирования (дополнительно)
            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('outward', 60))
            line2 = ax3.plot(year_stats.index, year_stats['mean_citations'], 
                           's-', color='darkgreen', linewidth=2, markersize=5,
                           label='Mean Citations per Paper')
            ax3.set_ylabel('Mean Citations per Paper', fontweight='bold', color='darkgreen')
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
            self.log_error(f"Error in plot_10_temporal_evolution: {str(e)}")
            return None
    
    def plot_11_temporal_heatmap(self):
        """11. Тепловая карта: Год публикации vs Возраст статьи"""
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
            self.plot_data['11_temporal_heatmap'] = {
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
            self.log_error(f"Error in plot_11_temporal_heatmap: {str(e)}")
            return None
    
    def plot_11_team_size_analysis(self):
        """11. Анализ размера команды (оригинальный)"""
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
                'Citation counts (CR)': 'mean',
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
            self.plot_data['11_team_size_analysis'] = group_stats.reset_index().to_dict('records')
            
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
            axes[2].set_ylabel('Mean Citations (CR)', fontweight='bold')
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
            self.log_error(f"Error in plot_11_team_size_analysis: {str(e)}")
            return None
    
    def plot_12_correlation_matrix(self):
        """12. Корреляционная матрица с выделением ключевых параметров"""
        try:
            numeric_cols = ['author count', 'references_count',
                          'Citation counts (CR)', 'Citation counts (OA)',
                          'Annual cit counts (CR)', 'Annual cit counts (OA)',
                          'count', 'num_countries', 'num_affiliations',
                          'article_age', 'normalized_attention', 'max_citations', 'max_annual_citations']
            
            available_cols = [col for col in numeric_cols if col in self.df_processed.columns]
            
            if len(available_cols) < 3:
                return None
            
            correlation_data = self.df_processed[available_cols].dropna()
            
            if len(correlation_data) < 10:
                return None
            
            corr_matrix = correlation_data.corr(method='spearman')
            
            # Переупорядочиваем матрицу: ключевые параметры сначала
            key_params = ['count', 'max_citations', 'max_annual_citations',
                         'Annual cit counts (CR)', 'Annual cit counts (OA)',
                         'Citation counts (CR)', 'Citation counts (OA)']
            
            # Фильтруем только те, что есть в данных
            existing_key_params = [p for p in key_params if p in corr_matrix.columns]
            other_params = [p for p in corr_matrix.columns if p not in existing_key_params]
            
            # Новый порядок
            new_order = existing_key_params + other_params
            corr_matrix = corr_matrix.reindex(index=new_order, columns=new_order)
            
            # Сохраняем данные
            self.plot_data['12_correlation_matrix'] = {
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
            self.log_error(f"Error in plot_12_correlation_matrix: {str(e)}")
            return None
    
    def plot_13_cr_vs_oa_comparison(self):
        """13. Сравнение CR vs OA цитирований"""
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
            self.plot_data['13_cr_vs_oa_comparison'] = {
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
            ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y = x')
            
            ax1.set_xlabel('Citations from Crossref (CR)', fontweight='bold')
            ax1.set_ylabel('Citations from OpenAlex (OA)', fontweight='bold')
            ax1.set_title('Comparison of Citation Counts', fontweight='bold')
            ax1.legend()
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
            self.log_error(f"Error in plot_13_cr_vs_oa_comparison: {str(e)}")
            return None
    
    def plot_14_citation_by_domain(self):
        """14. Цитируемость по доменам науки"""
        try:
            required_cols = ['Domain', 'Annual cit counts (CR)']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) == 0:
                return None
            
            # Агрегируем по доменам
            domain_stats = valid_data.groupby('Domain').agg({
                'Annual cit counts (CR)': ['median', 'mean', 'std', 'count'],
                'count': 'mean'
            }).round(2)
            
            domain_stats.columns = ['median_citations', 'mean_citations', 'std_citations',
                                  'num_papers', 'mean_attention']
            domain_stats = domain_stats.sort_values('median_citations', ascending=False)
            
            # Сохраняем данные
            self.plot_data['14_citation_by_domain'] = domain_stats.reset_index().to_dict('records')
            
            # Выбираем топ доменов
            top_domains = domain_stats.head(15).index.tolist()
            filtered_data = valid_data[valid_data['Domain'].isin(top_domains)]
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Boxplot
            box_data = []
            labels = []
            for domain in top_domains:
                data = filtered_data[filtered_data['Domain'] == domain]['Annual cit counts (CR)'].values
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
            ax.set_ylabel('Annual Citation Rate (CR)', fontweight='bold')
            ax.set_title('Citation Impact Distribution Across Research Domains', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_14_citation_by_domain: {str(e)}")
            return None
    
    def plot_15_cumulative_influence(self):
        """15. Накопительная кривая влияния"""
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
            self.plot_data['15_cumulative_influence'] = {
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
            self.log_error(f"Error in plot_15_cumulative_influence: {str(e)}")
            return None
    
    def _calculate_gini(self, x):
        """Расчет коэффициента Джини"""
        x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(x, dtype=float)
        if cumx[-1] == 0:
            return 0
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    
    def plot_16_references_vs_impact(self):
        """16. Объем ссылок vs влияние"""
        try:
            required_cols = ['references_count', 'count', 'Annual cit counts (CR)']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) < 10:
                return None
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # График 1: References vs Attention
            scatter1 = ax1.scatter(valid_data['references_count'],
                                 valid_data['count'],
                                 c=valid_data['Annual cit counts (CR)'],
                                 cmap='viridis', alpha=0.6, s=30,
                                 edgecolors='black', linewidth=0.5)
            
            # Линейная регрессия
            if len(valid_data) > 10:
                x = valid_data['references_count'].values
                y = valid_data['count'].values
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = intercept + slope * x_line
                ax1.plot(x_line, y_line, 'r--', linewidth=2,
                        label=f'r = {r_value:.3f}, p = {p_value:.3f}')
            
            ax1.set_xlabel('Number of References', fontweight='bold')
            ax1.set_ylabel('Local Mentions (count)', fontweight='bold')
            ax1.set_title('References vs Local Attention', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            cbar1 = plt.colorbar(scatter1, ax=ax1)
            cbar1.set_label('Annual Citations (CR)', fontweight='bold')
            
            # График 2: References vs Citations
            scatter2 = ax2.scatter(valid_data['references_count'],
                                 valid_data['Annual cit counts (CR)'],
                                 c=valid_data['count'],
                                 cmap='plasma', alpha=0.6, s=30,
                                 edgecolors='black', linewidth=0.5)
            
            ax2.set_xlabel('Number of References', fontweight='bold')
            ax2.set_ylabel('Annual Citations (CR)', fontweight='bold')
            ax2.set_title('References vs Citation Impact', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            cbar2 = plt.colorbar(scatter2, ax=ax2)
            cbar2.set_label('Local Mentions', fontweight='bold')
            
            plt.suptitle('Impact of Reference Count on Research Metrics', fontweight='bold', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_16_references_vs_impact: {str(e)}")
            return None
    
    def plot_17_journal_impact(self):
        """17. Влияние журналов"""
        try:
            required_cols = ['Full journal Name', 'count', 'Annual cit counts (CR)']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) == 0:
                return None
            
            # Агрегируем по журналам
            journal_stats = valid_data.groupby('Full journal Name').agg({
                'count': ['mean', 'median', 'std', 'size'],
                'Annual cit counts (CR)': 'mean',
                'references_count': 'mean'
            }).round(2)
            
            journal_stats.columns = ['mean_attention', 'median_attention', 'std_attention',
                                   'num_papers', 'mean_citations', 'mean_references']
            
            # Фильтруем журналы с достаточным количеством статей
            journal_stats = journal_stats[journal_stats['num_papers'] >= 3]
            journal_stats = journal_stats.sort_values('mean_attention', ascending=False)
            
            # Сохраняем данные
            self.plot_data['17_journal_impact'] = journal_stats.reset_index().to_dict('records')
            
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
            scatter = ax2.scatter(top_journals['mean_citations'],
                                top_journals['mean_attention'],
                                s=top_journals['num_papers'] * 10,
                                c=top_journals['mean_references'],
                                cmap='coolwarm', alpha=0.7,
                                edgecolors='black', linewidth=0.5)
            
            ax2.set_xlabel('Mean Annual Citations (CR)', fontweight='bold')
            ax2.set_ylabel('Mean Attention', fontweight='bold')
            ax2.set_title('Journal Impact: Citations vs Attention', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Mean References', fontweight='bold')
            
            # Добавляем аннотации
            for idx, row in top_journals.head(5).iterrows():
                short_name = idx[:15] + '...' if len(idx) > 15 else idx
                ax2.annotate(short_name,
                            xy=(row['mean_citations'], row['mean_attention']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, alpha=0.8)
            
            plt.suptitle('Journal Impact Analysis', fontweight='bold', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_17_journal_impact: {str(e)}")
            return None
    
    def plot_18_18_1_affiliation_network(self):
        """18.1 Сеть аффилиаций (топ 20)"""
        try:
            return self._plot_affiliation_network_impl(20, "1")
        except Exception as e:
            self.log_error(f"Error in plot_18_18_1_affiliation_network: {str(e)}")
            return None
    
    def plot_18_18_2_affiliation_network(self):
        """18.2 Сеть аффилиаций (топ 30)"""
        try:
            return self._plot_affiliation_network_impl(30, "2")
        except Exception as e:
            self.log_error(f"Error in plot_18_18_2_affiliation_network: {str(e)}")
            return None
    
    def plot_18_18_3_affiliation_network(self):
        """18.3 Сеть аффилиаций (топ 50)"""
        try:
            return self._plot_affiliation_network_impl(50, "3")
        except Exception as e:
            self.log_error(f"Error in plot_18_18_3_affiliation_network: {str(e)}")
            return None
    
    def _plot_affiliation_network_impl(self, top_n, suffix):
        """Общая реализация сети аффилиаций"""
        try:
            if 'affiliations_list' not in self.df_processed.columns:
                return None
            
            # Создаем граф
            G = nx.Graph()
            
            for idx, row in self.df_processed.iterrows():
                if isinstance(row['affiliations_list'], list) and len(row['affiliations_list']) >= 2:
                    affs = [a.strip() for a in row['affiliations_list']]
                    weight = row.get('count', 1)
                    
                    for aff in affs:
                        if not G.has_node(aff):
                            G.add_node(aff, weight=0, papers=0)
                        G.nodes[aff]['weight'] += weight
                        G.nodes[aff]['papers'] += 1
                    
                    for i in range(len(affs)):
                        for j in range(i+1, len(affs)):
                            if G.has_edge(affs[i], affs[j]):
                                G[affs[i]][affs[j]]['weight'] += weight
                                G[affs[i]][affs[j]]['papers'] += 1
                            else:
                                G.add_edge(affs[i], affs[j], weight=weight, papers=1)
            
            if len(G.nodes()) < 3:
                self.log_warning("Insufficient data for affiliation network")
                return None
            
            # Выбираем топ аффилиации
            degree_dict = dict(G.degree(weight='weight'))
            top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_node_names = [node[0] for node in top_nodes]
            
            H = G.subgraph(top_node_names)
            
            # Сохраняем данные
            self.plot_data[f'18_18_{suffix}_affiliation_network'] = {
                'nodes': [{'affiliation': node, 'weight': H.nodes[node]['weight'],
                          'papers': H.nodes[node]['papers']} for node in H.nodes()],
                'edges': [{'aff1': u, 'aff2': v, 'weight': H[u][v]['weight'],
                          'papers': H[u][v]['papers']} for u, v in H.edges()]
            }
            
            fig, ax = plt.subplots(figsize=(16, 12))
            
            pos = nx.spring_layout(H, k=3, seed=42)
            
            node_sizes = [H.nodes[n]['weight'] * 0.3 + 300 for n in H.nodes()]
            node_colors = [H.nodes[n]['papers'] for n in H.nodes()]
            
            nodes = nx.draw_networkx_nodes(H, pos, node_size=node_sizes,
                                          node_color=node_colors, cmap='Blues',
                                          alpha=0.8, edgecolors='black', linewidths=1.5, ax=ax)
            
            if H.edges():
                edge_weights = [H[u][v]['weight'] * 0.05 for u, v in H.edges()]
                edges = nx.draw_networkx_edges(H, pos, width=edge_weights, alpha=0.4,
                                              edge_color='gray', style='solid', ax=ax)
            
            # Подписи с переносом строк
            labels = {}
            for node in H.nodes():
                words = node.split()
                if len(node) > 30:
                    mid = len(words) // 2
                    labels[node] = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                else:
                    labels[node] = node
            
            nx.draw_networkx_labels(H, pos, labels=labels, font_size=8, font_weight='bold', ax=ax)
            
            ax.set_title(f'Top {top_n} Affiliation Collaboration Network (Version {suffix})', 
                        fontweight='bold', fontsize=16)
            ax.axis('off')
            
            # Цветовая шкала
            sm = plt.cm.ScalarMappable(cmap='Blues',
                                      norm=plt.Normalize(vmin=min(node_colors),
                                                       vmax=max(node_colors)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Number of Papers', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in _plot_affiliation_network_impl: {str(e)}")
            return None
    
    def plot_19_hierarchical_sankey(self):
        """19. Иерархическая диаграмма Санки: Domain → Field → Subfield → Topic"""
        try:
            required_cols = ['Domain', 'Field', 'Subfield', 'Topic', 'max_citations']
            available_cols = [col for col in required_cols if col in self.df_processed.columns]
            
            if len(available_cols) < 3:
                return None
            
            valid_data = self.df_processed.dropna(subset=available_cols)
            if len(valid_data) < 10:
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
            hierarchy_data = valid_data.groupby(['Domain', 'Field', 'Subfield', 'Topic']).agg({
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
            self.plot_data['19_hierarchical_sankey'] = {
                'nodes': nodes,
                'links': links,
                'total_weight': sum([l['value'] for l in links])
            }
            
            # Создаем диаграмму Санки с plotly (лучше для интерактивности)
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=nodes,
                    color="blue"
                ),
                link=dict(
                    source=[l['source'] for l in links],
                    target=[l['target'] for l in links],
                    value=[l['value'] for l in links]
                )
            )])
            
            fig.update_layout(
                title_text="Hierarchical Knowledge Structure: Domain → Field → Subfield → Topic",
                font_size=10,
                width=1200,
                height=800
            )
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_19_hierarchical_sankey: {str(e)}")
            return None
    
    def plot_20_multidimensional_scaling(self):
        """20. Многомерное шкалирование важных предикторов"""
        try:
            # Выбираем ключевые предикторы
            predictors = ['author count', 'references_count', 'num_countries',
                         'Annual cit counts (CR)', 'article_age', 'normalized_attention']
            
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
            self.plot_data['20_mds_analysis'] = {
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
            self.log_error(f"Error in plot_20_multidimensional_scaling: {str(e)}")
            return None
    
    def plot_21_concept_network_weighted(self):
        """21. Сеть концептов с весами по влиянию"""
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
            self.plot_data['21_concept_network_weighted'] = {
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
            self.log_error(f"Error in plot_21_concept_network_weighted: {str(e)}")
            return None
    
    def generate_all_plots(self, selected_plots=None):
        """Генерация всех графиков с прогресс-баром"""
        self.all_figures = {}
        self.plot_data = {}
        self.errors = []
        self.warnings = []
        
        # Обновленный список всех функций графиков (23 графика)
        plot_functions = [
            ("1_distribution", "1. Distribution of Attention", self.plot_1_distribution_attention),
            ("2_country_network", "2. Country Collaboration Network", self.plot_2_country_collaboration_network),
            ("3_internationality", "3. Internationality vs Citations", self.plot_3_internationality_vs_citations),
            ("4_journal_heatmap", "4. Journal-Year Heatmap", lambda: self.plot_4_journal_year_heatmap(15)),
            ("5_collab_linear", "5. Collaboration vs Citations (Linear)", self.plot_5_collaboration_vs_citations_linear),
            ("6_collab_log", "6. Collaboration vs Citations (Log Y Scale)", self.plot_6_collaboration_vs_citations_log),
            ("6_1_bubble_chart", "6.1 References vs Impact (Linear)", self.plot_6_1_bubble_chart),
            ("6_2_bubble_chart", "6.2 References vs Impact (Log)", self.plot_6_2_bubble_chart),
            ("7_concepts", "7. Concepts Analysis", lambda: self.plot_7_concepts_analysis(30)),
            ("8_concept_cooccurrence", "8. Concept Co-occurrence", lambda: self.plot_8_concept_cooccurrence(15)),
            ("9_concept_influence", "9. Concept Influence Analysis", self.plot_9_concept_influence),
            ("10_temporal_evolution", "10. Temporal Evolution", self.plot_10_temporal_evolution),
            ("11_temporal_heatmap", "11. Temporal Heatmap", self.plot_11_temporal_heatmap),
            ("11_team_size", "12. Team Size Analysis", self.plot_11_team_size_analysis),
            ("12_correlation", "13. Correlation Matrix", self.plot_12_correlation_matrix),
            ("13_cr_vs_oa", "14. CR vs OA Comparison", self.plot_13_cr_vs_oa_comparison),
            ("14_domain_citations", "15. Citations by Domain", self.plot_14_citation_by_domain),
            ("15_cumulative_influence", "16. Cumulative Influence", self.plot_15_cumulative_influence),
            ("16_references_impact", "17. References vs Impact", self.plot_16_references_vs_impact),
            ("17_journal_impact", "18. Journal Impact", self.plot_17_journal_impact),
            ("18_18_1_affiliation_network", "19.1 Affiliation Network (Top 20)", self.plot_18_18_1_affiliation_network),
            ("18_18_2_affiliation_network", "19.2 Affiliation Network (Top 30)", self.plot_18_18_2_affiliation_network),
            ("18_18_3_affiliation_network", "19.3 Affiliation Network (Top 50)", self.plot_18_18_3_affiliation_network),
            ("19_hierarchical_sankey", "20. Hierarchical Sankey Diagram", self.plot_19_hierarchical_sankey),
            ("20_mds", "21. Multidimensional Scaling", self.plot_20_multidimensional_scaling),
            ("21_concept_network_weighted", "22. Weighted Concept Network", self.plot_21_concept_network_weighted)
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
                fig.savefig(img_buffer, format='png', dpi=300,
                          bbox_inches='tight', facecolor='white',
                          edgecolor='black')
                img_buffer.seek(0)
                
                filename = f"plot_{i+1:02d}_{name}.png"
                zip_file.writestr(filename, img_buffer.read())
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
        ("2_country_network", "2. Country Collaboration Network"),
        ("3_internationality", "3. Internationality vs Citations"),
        ("4_journal_heatmap", "4. Journal-Year Heatmap"),
        ("5_collab_linear", "5. Collaboration vs Citations (Linear)"),
        ("6_collab_log", "6. Collaboration vs Citations (Log Y Scale)"),
        ("6_1_bubble_chart", "6.1 References vs Impact (Linear)"),
        ("6_2_bubble_chart", "6.2 References vs Impact (Log)"),
        ("7_concepts", "7. Concepts Analysis"),
        ("8_concept_cooccurrence", "8. Concept Co-occurrence"),
        ("9_concept_influence", "9. Concept Influence Analysis"),
        ("10_temporal_evolution", "10. Temporal Evolution"),
        ("11_temporal_heatmap", "11. Temporal Heatmap"),
        ("11_team_size", "12. Team Size Analysis"),
        ("12_correlation", "13. Correlation Matrix"),
        ("13_cr_vs_oa", "14. CR vs OA Comparison"),
        ("14_domain_citations", "15. Citations by Domain"),
        ("15_cumulative_influence", "16. Cumulative Influence"),
        ("16_references_impact", "17. References vs Impact"),
        ("17_journal_impact", "18. Journal Impact"),
        ("18_18_1_affiliation_network", "19.1 Affiliation Network (Top 20)"),
        ("18_18_2_affiliation_network", "19.2 Affiliation Network (Top 30)"),
        ("18_18_3_affiliation_network", "19.3 Affiliation Network (Top 50)"),
        ("19_hierarchical_sankey", "20. Hierarchical Sankey Diagram"),
        ("20_mds", "21. Multidimensional Scaling"),
        ("21_concept_network_weighted", "22. Weighted Concept Network")
    ]
    
    # Инициализация состояния сессии
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ScientificDataAnalyzer()
    
    if 'plots_generated' not in st.session_state:
        st.session_state.plots_generated = False
    
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
        
        # Список графиков для выбора
        all_plots = [
            ("1_distribution", "1. Distribution of Attention"),
            ("2_country_network", "2. Country Collaboration Network"),
            ("3_internationality", "3. Internationality vs Citations"),
            ("4_journal_heatmap", "4. Journal-Year Heatmap"),
            ("5_collab_linear", "5. Collaboration vs Citations (Linear)"),
            ("6_collab_log", "6. Collaboration vs Citations (Log-Log)"),
            ("6_1_bubble_chart", "6.1 References vs Impact (Linear)"),
            ("6_2_bubble_chart", "6.2 References vs Impact (Log)"),
            ("7_concepts", "7. Concepts Analysis"),
            ("8_concept_cooccurrence", "8. Concept Co-occurrence"),
            ("9_concept_influence", "9. Concept Influence Analysis"),
            ("10_temporal_evolution", "10. Temporal Evolution"),
            ("11_temporal_heatmap", "11. Temporal Heatmap"),
            ("11_team_size", "12. Team Size Analysis"),
            ("12_correlation", "13. Correlation Matrix"),
            ("13_cr_vs_oa", "14. CR vs OA Comparison"),
            ("14_domain_citations", "15. Citations by Domain"),
            ("15_cumulative_influence", "16. Cumulative Influence"),
            ("16_references_impact", "17. References vs Impact"),
            ("17_journal_impact", "18. Journal Impact"),
            ("18_18_1_affiliation_network", "19.1 Affiliation Network (Top 20)"),
            ("18_18_2_affiliation_network", "19.2 Affiliation Network (Top 30)"),
            ("18_18_3_affiliation_network", "19.3 Affiliation Network (Top 50)"),
            ("19_hierarchical_sankey", "20. Hierarchical Sankey Diagram"),
            ("20_mds", "21. Multidimensional Scaling"),
            ("21_concept_network_weighted", "22. Weighted Concept Network")
        ]
        
        # Выбор графиков
        st.subheader("🎯 Выберите графики для генерации")
        
        # Выбор всех графиков
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Выбрать все", use_container_width=True):
                st.session_state.selected_plots = [plot[0] for plot in all_plots]
                st.rerun()
        with col2:
            if st.button("❌ Очистить выбор", use_container_width=True):
                st.session_state.selected_plots = []
                st.rerun()
        
        # Инициализация выбранных графиков
        if 'selected_plots' not in st.session_state:
            st.session_state.selected_plots = [plot[0] for plot in all_plots]
        
        # Чекбоксы для выбора графиков
        st.markdown("### Доступные графики")
        
        # Группируем графики по категориям
        categories = {
            "📈 Основные распределения": ["1_distribution", "15_cumulative_influence"],
            "🌍 Международное сотрудничество": ["2_country_network", "3_internationality", "5_collab_linear", "6_collab_log"],
            "📚 Журналы и публикации": ["4_journal_heatmap", "17_journal_impact"],
            "🔗 Ссылки и цитирования": ["6_1_bubble_chart", "6_2_bubble_chart", "13_cr_vs_oa", "16_references_impact"],
            "🏷️ Концепты и темы": ["7_concepts", "8_concept_cooccurrence", "9_concept_influence", "21_concept_network_weighted"],
            "⏰ Временной анализ": ["10_temporal_evolution", "11_temporal_heatmap"],
            "👥 Команды и организации": ["11_team_size", "18_18_1_affiliation_network", 
                                       "18_18_2_affiliation_network", "18_18_3_affiliation_network"],
            "📊 Анализ метрик": ["12_correlation", "14_domain_citations", "20_mds"],
            "🏛️ Иерархическая структура": ["19_hierarchical_sankey"]
        }
        
        for category, plot_ids in categories.items():
            with st.expander(category):
                for plot_id in plot_ids:
                    plot_name = next(name for pid, name in all_plots if pid == plot_id)
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
                    with st.spinner("Генерация графиков..."):
                        st.session_state.analyzer.generate_all_plots(st.session_state.selected_plots)
                        st.session_state.plots_generated = True
                        st.success(f"✅ Сгенерировано {len(st.session_state.analyzer.all_figures)} графиков!")
        
        with col2:
            if st.button("🎯 Сгенерировать все графики", use_container_width=True):
                st.session_state.selected_plots = [plot[0] for plot in all_plots]
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
                    format_func=lambda x: next(name for pid, name in all_plots if pid == x)
                )
                
                # Показать выбранный график
                if selected_plot in st.session_state.analyzer.all_figures:
                    fig = st.session_state.analyzer.all_figures[selected_plot]
                    
                    # Проверяем тип графика (plotly или matplotlib)
                    if hasattr(fig, 'update_layout'):
                        # Это plotly фигура
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Это matplotlib фигура
                        st.pyplot(fig)
                    
                    # Информация о графике
                    plot_name = next(name for pid, name in all_plots if pid == selected_plot)
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
                
                # Сохранить график в буфер (для matplotlib)
                if hasattr(fig, 'update_layout'):
                    # plotly фигура
                    import plotly.io as pio
                    img_buffer = io.BytesIO()
                    pio.write_image(fig, img_buffer, format='png', width=1200, height=800, scale=2)
                    img_buffer.seek(0)
                else:
                    # matplotlib фигура
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                    img_buffer.seek(0)
                
                st.download_button(
                    label=f"📥 Скачать {selected_plot_name}",
                    data=img_buffer,
                    file_name=f"plot_{selected_plot_name[:20].replace(' ', '_')}.png",
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
