# -*- coding: utf-8 -*-
"""
Scientific Data Visualization Dashboard - Streamlit Version
Advanced Multi-Stage Scientific Data Analysis and Visualization Tool
"""

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
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CUSTOM CSS FOR MODERN DESIGN
# ============================================================================

def load_custom_css():
    """Load custom CSS for modern Streamlit design"""
    st.markdown("""
    <style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .custom-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        color: white;
        margin: 0.5rem;
    }
    
    .metric-card h3 {
        font-size: 0.9rem;
        margin: 0;
        opacity: 0.9;
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        font-weight: bold;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f5f7fa 0%, #e9ecef 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Success message */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #28a745;
    }
    
    /* Info message */
    .stAlert.info {
        border-left-color: #17a2b8;
    }
    
    /* Warning message */
    .stAlert.warning {
        border-left-color: #ffc107;
    }
    
    /* Error message */
    .stAlert.error {
        border-left-color: #dc3545;
    }
    
    /* Custom divider */
    .custom-divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        margin: 1rem 0;
        border-radius: 3px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        font-size: 0.9rem;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SCIENTIFIC PLOTTING STYLE CONFIGURATION
# ============================================================================

def set_scientific_style(fig_size=(12, 9), dpi=600):
    """Set scientific plotting style with specified dimensions"""
    plt.style.use('default')
    plt.rcParams.update({
        # Font sizes and weights
        'font.size': 10,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
        'axes.labelsize': 11,
        'axes.labelweight': 'bold',
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        
        # Axes appearance
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'axes.grid': False,
        
        # Tick parameters
        'xtick.color': 'black',
        'ytick.color': 'black',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.major.size': 4,
        'ytick.minor.size': 2,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        
        # Legend
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        
        # Figure
        'figure.dpi': dpi,
        'figure.figsize': fig_size,
        'savefig.dpi': dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'figure.facecolor': 'white',
        
        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'errorbar.capsize': 3,
    })

# Color palette options for different plot types
COLOR_PALETTES = {
    'heatmap': [
        'Blues', 'Reds', 'Greens', 'Purples', 'Oranges',
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'coolwarm', 'RdYlGn', 'RdYlBu', 'Spectral', 'PiYG'
    ],
    'scatter': [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'coolwarm', 'rainbow', 'jet', 'hsv', 'twilight'
    ],
    'bar': [
        'Blues', 'Reds', 'Greens', 'Purples', 'Oranges',
        'Set1', 'Set2', 'Set3', 'Paired', 'Dark2'
    ],
    'network': [
        'Blues', 'Reds', 'Greens', 'Purples', 'Oranges',
        'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    ],
    'sequential': [
        'Blues', 'Greens', 'Reds', 'Purples', 'Oranges',
        'Greys', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd'
    ]
}

# ============================================================================
# CLASS FOR DATA ANALYSIS AND VISUALIZATION
# ============================================================================

class ScientificDataAnalyzer:
    """Class for scientific data analysis and visualization"""
    
    def __init__(self):
        self.df = None
        self.df_processed = None
        self.all_figures = {}
        self.plot_data = {}
        self.errors = []
        self.warnings = []
        self.progress = 0
        self.plot_settings = {
            'show_regression': True,
            'heatmap_palette': 'Blues',
            'scatter_palette': 'viridis',
            'bar_palette': 'Blues',
            'network_palette': 'Blues',
            'sequential_palette': 'Blues'
        }
        
    def log_error(self, error_msg, details=""):
        """Log error message"""
        self.errors.append({
            'timestamp': datetime.now().isoformat(),
            'message': error_msg,
            'details': details
        })
        st.error(f"❌ ERROR: {error_msg}")
        if details:
            st.error(f"   Details: {details}")
    
    def log_warning(self, warning_msg):
        """Log warning message"""
        self.warnings.append({
            'timestamp': datetime.now().isoformat(),
            'message': warning_msg
        })
        st.warning(f"⚠️ WARNING: {warning_msg}")
    
    def update_progress(self, value):
        """Update progress value"""
        self.progress = value
    
    def update_plot_settings(self, settings):
        """Update plot settings"""
        self.plot_settings.update(settings)
    
    def parse_data(self, data_text):
        """Parse data from text input with extended diagnostics"""
        st.info("🔍 Parsing data...")
        
        try:
            # Split text into lines
            lines = data_text.strip().split('\n')
            if len(lines) < 2:
                self.log_error("Not enough data rows", f"Found {len(lines)} lines")
                return None
            
            # Parse headers
            headers = lines[0].split('\t')
            st.info(f"   Found {len(headers)} columns")
            st.info(f"   Headers: {headers}")
            
            # Check required columns
            required_columns = ['doi', 'Title', 'year', 'count']
            missing_columns = [col for col in required_columns if col not in headers]
            if missing_columns:
                self.log_warning(f"Missing columns: {missing_columns}")
            
            # Parse data
            data = []
            for i, line in enumerate(lines[1:]):
                if line.strip():
                    values = line.split('\t')
                    # Fill missing values
                    while len(values) < len(headers):
                        values.append('')
                    data.append(values)
            
            # Create DataFrame
            self.df = pd.DataFrame(data, columns=headers)
            st.success(f"✅ Successfully parsed {len(self.df)} rows")
            
            # Data diagnostics
            self._diagnose_data()
            
            # Preprocess data
            self.df_processed = self._preprocess_data(self.df)
            
            return self.df_processed
            
        except Exception as e:
            self.log_error(f"Error parsing data: {str(e)}", traceback.format_exc())
            return None
    
    def _diagnose_data(self):
        """Diagnose data quality"""
        st.info("🔬 Data Diagnostics:")
        st.write("---")
        
        # Check missing values
        missing_counts = self.df.isnull().sum()
        total_cells = np.prod(self.df.shape)
        missing_percent = (missing_counts.sum() / total_cells) * 100
        
        st.info(f"   Total cells: {total_cells:,}")
        st.info(f"   Missing values: {missing_counts.sum():,} ({missing_percent:.1f}%)")
        
        # Top columns with missing values
        top_missing = missing_counts[missing_counts > 0].sort_values(ascending=False).head(5)
        if len(top_missing) > 0:
            st.info("   Top columns with missing values:")
            for col, count in top_missing.items():
                percent = (count / len(self.df)) * 100
                st.info(f"     - {col}: {count:,} ({percent:.1f}%)")
        
        # Check numeric columns
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
        
        # Key statistics
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
        """Preprocess data"""
        st.info("🔄 Preprocessing data...")
        
        df_processed = df.copy()
        
        # Convert numeric columns
        numeric_cols = ['author count', 'year', 'Citation counts (CR)', 'Citation counts (OA)',
                       'Annual cit counts (CR)', 'Annual cit counts (OA)', 'references_count', 'count']
        
        for col in numeric_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Process date
        if 'publication_date' in df_processed.columns:
            df_processed['publication_date'] = pd.to_datetime(df_processed['publication_date'], errors='coerce')
            if 'year' not in df_processed.columns or df_processed['year'].isnull().all():
                df_processed['year'] = df_processed['publication_date'].dt.year
        
        # Process list columns
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
        
        # Calculate additional metrics
        current_year = datetime.now().year
        if 'year' in df_processed.columns:
            df_processed['article_age'] = current_year - df_processed['year']
            df_processed['article_age'] = df_processed['article_age'].clip(lower=1)
            
            if 'count' in df_processed.columns:
                df_processed['normalized_attention'] = df_processed['count'] / df_processed['article_age']
        
        # Calculate maximum citations between CR and OA
        if 'Citation counts (CR)' in df_processed.columns and 'Citation counts (OA)' in df_processed.columns:
            df_processed['max_citations'] = df_processed[['Citation counts (CR)', 'Citation counts (OA)']].max(axis=1)
            df_processed['max_annual_citations'] = df_processed[['Annual cit counts (CR)', 'Annual cit counts (OA)']].max(axis=1)
        
        # Count countries and affiliations
        if 'countries_list' in df_processed.columns:
            df_processed['num_countries'] = df_processed['countries_list'].apply(len)
        
        if 'affiliations_list' in df_processed.columns:
            df_processed['num_affiliations'] = df_processed['affiliations_list'].apply(len)
        
        st.success("✅ Data preprocessing complete")
        return df_processed
    
    # ============================================================================
    # PLOT FUNCTIONS (All plots with scientific style and enhanced legends)
    # ============================================================================
    
    def plot_1_distribution_attention(self):
        """1. Distribution of attention (log-log, CCDF, Lorenz)"""
        try:
            if 'count' not in self.df_processed.columns:
                return None
            
            counts = self.df_processed['count'].dropna().values
            counts = counts[counts > 0]
            
            if len(counts) < 10:
                self.log_warning("Insufficient data for distribution plot")
                return None
            
            fig, axes = plt.subplots(1, 3, figsize=(12, 9), dpi=600)
            fig.suptitle('Distribution of Research Attention', fontweight='bold', fontsize=14)
            
            # A: Log-log histogram
            axes[0].hist(counts, bins=np.logspace(np.log10(1), np.log10(max(100, counts.max())), 30),
                        edgecolor='black', alpha=0.7, color='#2E86AB')
            axes[0].set_xscale('log')
            axes[0].set_yscale('log')
            axes[0].set_xlabel('Number of Mentions', fontweight='bold')
            axes[0].set_ylabel('Frequency', fontweight='bold')
            axes[0].set_title('A. Log-Log Distribution', fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # Save data
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
            
            # C: Lorenz curve
            sorted_counts = np.sort(counts)
            cumulative_counts = np.cumsum(sorted_counts)
            cumulative_percent = cumulative_counts / cumulative_counts[-1]
            population_percent = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
            
            axes[2].plot(population_percent, cumulative_percent, linewidth=2.5,
                        color='#6B8E23', label='Lorenz curve')
            axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5, label='Perfect equality')
            axes[2].fill_between(population_percent, 0, cumulative_percent, alpha=0.2, color='#6B8E23')
            
            # Calculate Gini index
            gini = 0
            for i in range(1, len(population_percent)):
                gini += (population_percent[i] - population_percent[i-1]) * \
                       (cumulative_percent[i] + cumulative_percent[i-1])
            gini = 1 - gini
            
            axes[2].text(0.7, 0.3, f'Gini Index = {gini:.3f}',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
            
            axes[2].set_xlabel('Cumulative Proportion of Papers', fontweight='bold')
            axes[2].set_ylabel('Cumulative Proportion of Mentions', fontweight='bold')
            axes[2].set_title('C. Lorenz Curve', fontweight='bold')
            axes[2].legend(loc='upper left', fontsize=9)
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_1_distribution_attention: {str(e)}")
            return None
    
    def plot_2_country_collaboration_network(self):
        """2. Country collaboration network"""
        try:
            if 'countries_list' not in self.df_processed.columns:
                return None
            
            # Create graph
            G = nx.Graph()
            country_pairs = []
            
            for idx, row in self.df_processed.iterrows():
                if isinstance(row['countries_list'], list) and len(row['countries_list']) >= 2:
                    countries = [c.strip().upper() for c in row['countries_list']]
                    weight = row.get('count', 1)
                    
                    # Add nodes
                    for country in countries:
                        if not G.has_node(country):
                            G.add_node(country, weight=0, papers=0)
                        G.nodes[country]['weight'] += weight
                        G.nodes[country]['papers'] += 1
                    
                    # Add edges between all pairs
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
            
            # Save data
            self.plot_data['2_country_network'] = {
                'nodes': [{'country': node, 'weight': G.nodes[node]['weight'], 
                          'papers': G.nodes[node]['papers']} for node in G.nodes()],
                'edges': [{'country1': u, 'country2': v, 'weight': G[u][v]['weight'],
                          'papers': G[u][v]['papers']} for u, v in G.edges()]
            }
            
            # Visualization
            fig, ax = plt.subplots(figsize=(12, 9), dpi=600)
            
            # Filter weak connections
            min_weight = np.percentile([d['weight'] for u, v, d in G.edges(data=True)], 50)
            edges_to_keep = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] >= min_weight]
            H = G.edge_subgraph([u for u, v in edges_to_keep] + [v for u, v in edges_to_keep])
            
            if len(H.nodes()) == 0:
                H = G
            
            # Positioning
            pos = nx.spring_layout(H, k=2, seed=42)
            
            # Node sizes by weight
            node_sizes = [H.nodes[n]['weight'] * 0.5 + 500 for n in H.nodes()]
            node_colors = [H.nodes[n]['papers'] for n in H.nodes()]
            
            # Draw graph
            nodes = nx.draw_networkx_nodes(H, pos, node_size=node_sizes,
                                          node_color=node_colors, cmap=self.plot_settings['network_palette'],
                                          alpha=0.8, edgecolors='black', linewidths=1.5, ax=ax)
            
            # Edges
            if H.edges():
                edge_weights = [H[u][v]['weight'] * 0.1 for u, v in H.edges()]
                edges = nx.draw_networkx_edges(H, pos, width=edge_weights, alpha=0.5,
                                              edge_color='gray', style='solid', ax=ax)
            
            # Labels
            nx.draw_networkx_labels(H, pos, font_size=9, font_weight='bold', ax=ax)
            
            ax.set_title('Country Collaboration Network', fontweight='bold', fontsize=14, pad=20)
            ax.axis('off')
            
            # Colorbar
            sm = plt.cm.ScalarMappable(cmap=self.plot_settings['network_palette'],
                                      norm=plt.Normalize(vmin=min(node_colors), 
                                                       vmax=max(node_colors)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Number of Papers', fontweight='bold', fontsize=9)
            
            # Statistics
            stats_text = f"Nodes: {len(G.nodes())} | Edges: {len(G.edges())}"
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Legend for node sizes
            size_legend_elements = []
            size_values = [500, 1000, 1500, 2000]
            for size in size_values:
                size_legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                                  markerfacecolor='gray',
                                                  markersize=np.sqrt(size/10),
                                                  label=f'Weight ~{int(size/0.5-500):.0f}'))
            if size_legend_elements:
                ax.legend(handles=size_legend_elements, title='Node Weight',
                         loc='upper right', fontsize=8, title_fontsize=9)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_2_country_collaboration_network: {str(e)}")
            return None
    
    def plot_3_internationality_vs_citations(self):
        """3. Internationality vs Citations"""
        try:
            required_cols = ['num_countries', 'Citation counts (CR)', 'author count']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) < 10:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 9), dpi=600)
            
            scatter = ax.scatter(valid_data['num_countries'],
                               valid_data['Citation counts (CR)'],
                               c=valid_data.get('Annual cit counts (CR)', 1),
                               s=valid_data['author count'] * 20,
                               alpha=0.7,
                               cmap=self.plot_settings['scatter_palette'],
                               edgecolors='black',
                               linewidth=0.5)
            
            # Regression line if enabled
            if self.plot_settings['show_regression'] and len(valid_data) > 10:
                x = valid_data['num_countries'].values
                y = valid_data['Citation counts (CR)'].values
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() > 10:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                    y_line = intercept + slope * x_line
                    ax.plot(x_line, y_line, 'r--', linewidth=2, 
                           label=f'Regression: r = {r_value:.3f}, p = {p_value:.3f}')
                    ax.legend(loc='upper left', fontsize=9)
            
            ax.set_xlabel('Number of Collaborating Countries', fontweight='bold')
            ax.set_ylabel('Total Citations (CR)', fontweight='bold')
            ax.set_title('International Collaboration vs Citation Impact',
                        fontweight='bold', fontsize=14)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Annual Citation Rate (CR)', fontweight='bold')
            
            # Legend for bubble sizes
            size_legend_elements = []
            sizes = [5, 10, 15]
            labels = ['5 authors', '10 authors', '15 authors']
            for size, label in zip(sizes, labels):
                size_legend_elements.append(plt.scatter([], [], s=size*20, c='gray',
                                                       alpha=0.7, edgecolors='black',
                                                       label=label))
            ax.legend(handles=size_legend_elements, loc='upper left', 
                     title='Team Size', fontsize=8, title_fontsize=9)
            
            ax.grid(True, alpha=0.3)
            
            # Save data
            self.plot_data['3_internationality_vs_citations'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_3_internationality_vs_citations: {str(e)}")
            return None
    
    def plot_4_journal_year_heatmap(self, top_journals=15):
        """4. Heatmap: Journal vs Year"""
        try:
            required_cols = ['Full journal Name', 'year', 'Annual cit counts (CR)']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            # Select top journals
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
            
            # Filter years with all zeros
            row_sums = pivot_table.sum(axis=1)
            pivot_table = pivot_table[row_sums > 0]
            
            if len(pivot_table) < 2:
                self.log_warning("Insufficient years with data for heatmap")
                return None
            
            # Save data
            self.plot_data['4_journal_year_heatmap'] = {
                'pivot_data': pivot_table.to_dict(),
                'years': pivot_table.index.tolist(),
                'journals': pivot_table.columns.tolist()
            }
            
            fig, ax = plt.subplots(figsize=(12, 9), dpi=600)
            
            im = ax.imshow(pivot_table.values, cmap=self.plot_settings['heatmap_palette'], 
                          aspect='auto', vmin=0)
            
            ax.set_xticks(np.arange(len(pivot_table.columns)))
            ax.set_yticks(np.arange(len(pivot_table.index)))
            ax.set_xticklabels(pivot_table.columns, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(pivot_table.index.astype(int), fontsize=9)
            
            # Add values
            for i in range(len(pivot_table.index)):
                for j in range(len(pivot_table.columns)):
                    value = pivot_table.iloc[i, j]
                    if value > 0:
                        ax.text(j, i, f'{value:.1f}',
                              ha="center", va="center",
                              color="white" if value > pivot_table.values.max()/2 else "black",
                              fontsize=7, fontweight='bold')
            
            ax.set_xlabel('Journal', fontweight='bold')
            ax.set_ylabel('Publication Year', fontweight='bold')
            ax.set_title(f'Average Annual Citation Rate by Journal and Year (Top {top_journals} Journals)',
                        fontweight='bold', fontsize=14)
            
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Average Annual Citations (CR)', rotation=90, fontsize=11)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_4_journal_year_heatmap: {str(e)}")
            return None
    
    def plot_5_collaboration_vs_citations_linear(self):
        """5. Collaboration vs Citations (Linear scale)"""
        try:
            required_cols = ['author count', 'num_affiliations', 'num_countries', 'max_citations']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) < 10:
                return None
            
            fig, axes = plt.subplots(1, 3, figsize=(12, 9), dpi=600)
            fig.suptitle('Collaboration Scale vs Citation Impact (Linear Scale)', 
                        fontweight='bold', fontsize=14)
            
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
                                   cmap=self.plot_settings['scatter_palette'],
                                   edgecolors='black',
                                   linewidth=0.5)
                
                # Linear regression if enabled
                if self.plot_settings['show_regression'] and len(valid_data) > 10:
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
                ax.set_ylabel('Maximum Citations (max(CR, OA))', fontweight='bold')
                ax.set_title(f'{label} vs Citations', fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                if idx < 2:
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar_label = 'Number of Countries' if metric != 'num_countries' else 'Number of Authors'
                    cbar.set_label(cbar_label, fontweight='bold', fontsize=9)
            
            # Save data
            self.plot_data['5_collaboration_vs_citations_linear'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_5_collaboration_vs_citations_linear: {str(e)}")
            return None
    
    def plot_6_collaboration_vs_citations_log(self):
        """6. Collaboration vs Citations (Log Y scale)"""
        try:
            required_cols = ['author count', 'num_affiliations', 'num_countries', 'max_citations']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) < 10:
                return None
            
            fig, axes = plt.subplots(1, 3, figsize=(12, 9), dpi=600)
            fig.suptitle('Collaboration Scale vs Citation Impact (Log Y Scale)', 
                        fontweight='bold', fontsize=14)
            
            metrics = [
                ('author count', 'Number of Authors', axes[0]),
                ('num_affiliations', 'Number of Affiliations', axes[1]),
                ('num_countries', 'Number of Countries', axes[2])
            ]
            
            for idx, (metric, label, ax) in enumerate(metrics):
                # Filter data > 0 for log scale on Y
                plot_data = valid_data[valid_data['max_citations'] > 0].copy()
                if len(plot_data) < 10:
                    continue
                
                # Create scatter plot
                scatter = ax.scatter(plot_data[metric],
                                   plot_data['max_citations'],
                                   c=plot_data['num_countries'] if metric != 'num_countries' else plot_data['author count'],
                                   s=plot_data['author count'] * 10,
                                   alpha=0.6,
                                   cmap=self.plot_settings['scatter_palette'],
                                   edgecolors='black',
                                   linewidth=0.5)
                
                # Exponential regression (log Y) if enabled
                if self.plot_settings['show_regression'] and len(plot_data) > 10:
                    x = plot_data[metric].values
                    log_y = np.log(plot_data['max_citations'].values)
                    
                    # Remove infinite values
                    mask = np.isfinite(log_y)
                    if mask.sum() > 10:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], log_y[mask])
                        x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                        y_line = np.exp(intercept + slope * x_line)
                        ax.plot(x_line, y_line, 'r--', linewidth=2, 
                               label=f'exponential: y ∝ exp({slope:.3f}x), r = {r_value:.3f}')
                        ax.legend(loc='upper left', fontsize=8)
                
                ax.set_xlabel(label, fontweight='bold')
                ax.set_ylabel('Maximum Citations (max(CR, OA)) - Log Scale', fontweight='bold')
                ax.set_title(f'{label} vs Citations (Log Y Scale)', fontweight='bold')
                
                # Set log scale only for Y axis
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3, which='both')
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                if metric != 'num_countries':
                    cbar.set_label('Number of Countries', fontweight='bold', fontsize=9)
                else:
                    cbar.set_label('Number of Authors', fontweight='bold', fontsize=9)
                
                # Add legend for bubble sizes
                legend_elements = []
                size_values = [2, 5, 10, 15]
                for n_authors in size_values:
                    if n_authors <= plot_data['author count'].max():
                        marker_size = n_authors * 10
                        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                                     markerfacecolor='gray', 
                                                     markersize=np.sqrt(marker_size),
                                                     label=f'{n_authors} authors'))
                if legend_elements:
                    ax.legend(handles=legend_elements, title='Bubble size = Team size',
                             loc='lower right', fontsize=8, title_fontsize=9)
            
            # Save data
            self.plot_data['6_collaboration_vs_citations_log'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_6_collaboration_vs_citations_log: {str(e)}")
            return None
    
    def plot_6_1_bubble_chart(self):
        """6.1 Bubble chart: References vs Citations (linear scale)"""
        try:
            required_cols = ['references_count', 'Citation counts (CR)', 'author count', 'num_countries']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) < 10:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 9), dpi=600)
            
            scatter = ax.scatter(valid_data['references_count'],
                               valid_data['Citation counts (CR)'],
                               s=valid_data['author count'] * 40,
                               c=valid_data['num_countries'],
                               cmap=self.plot_settings['scatter_palette'],
                               alpha=0.7,
                               edgecolors='black',
                               linewidth=0.5)
            
            # Regression line if enabled
            if self.plot_settings['show_regression'] and len(valid_data) > 10:
                x = valid_data['references_count'].values
                y = valid_data['Citation counts (CR)'].values
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() > 10:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                    y_line = intercept + slope * x_line
                    ax.plot(x_line, y_line, 'r--', linewidth=2,
                           label=f'Regression: r = {r_value:.3f}, p = {p_value:.3f}')
                    ax.legend(loc='upper left', fontsize=9)
            
            ax.set_xlabel('Number of References', fontweight='bold')
            ax.set_ylabel('Total Citations (CR) - Linear Scale', fontweight='bold')
            ax.set_title('Research Breadth vs Impact (Linear Scale)',
                        fontweight='bold', fontsize=14)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Number of Collaborating Countries', fontweight='bold')
            
            # Legend for sizes
            legend_elements = []
            sizes = [5, 10, 15]
            labels = ['5 authors', '10 authors', '15 authors']
            for size, label in zip(sizes, labels):
                legend_elements.append(plt.scatter([], [], s=size*40, c='gray', alpha=0.7,
                                                  edgecolors='black', label=label))
            ax.legend(handles=legend_elements, title='Team Size', loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Save data
            self.plot_data['6_1_bubble_chart'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_6_1_bubble_chart: {str(e)}")
            return None
    
    def plot_6_2_bubble_chart(self):
        """6.2 Bubble chart: References vs Citations (logarithmic scale)"""
        try:
            required_cols = ['references_count', 'Citation counts (CR)', 'author count', 'num_countries']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) < 10:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 9), dpi=600)
            
            # Filter positive citations for log scale
            plot_data = valid_data[valid_data['Citation counts (CR)'] > 0].copy()
            
            scatter = ax.scatter(plot_data['references_count'],
                               plot_data['Citation counts (CR)'],
                               s=plot_data['author count'] * 40,
                               c=plot_data['num_countries'],
                               cmap=self.plot_settings['scatter_palette'],
                               alpha=0.7,
                               edgecolors='black',
                               linewidth=0.5)
            
            # Regression line for log-log if enabled
            if self.plot_settings['show_regression'] and len(plot_data) > 10:
                x = plot_data['references_count'].values
                log_y = np.log(plot_data['Citation counts (CR)'].values)
                mask = ~(np.isnan(x) | np.isnan(log_y))
                if mask.sum() > 10:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], log_y[mask])
                    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                    y_line = np.exp(intercept + slope * x_line)
                    ax.plot(x_line, y_line, 'r--', linewidth=2,
                           label=f'exponential: y ∝ exp({slope:.3f}x), r = {r_value:.3f}')
                    ax.legend(loc='upper left', fontsize=9)
            
            ax.set_xlabel('Number of References', fontweight='bold')
            ax.set_ylabel('Total Citations (CR) - Log Scale', fontweight='bold')
            ax.set_title('Research Breadth vs Impact (Logarithmic Scale)',
                        fontweight='bold', fontsize=14)
            
            # Set logarithmic scale for Y axis
            ax.set_yscale('log')
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Number of Collaborating Countries', fontweight='bold')
            
            # Legend for sizes
            legend_elements = []
            sizes = [5, 10, 15]
            labels = ['5 authors', '10 authors', '15 authors']
            for size, label in zip(sizes, labels):
                legend_elements.append(plt.scatter([], [], s=size*40, c='gray', alpha=0.7,
                                                  edgecolors='black', label=label))
            ax.legend(handles=legend_elements, title='Team Size', loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3, which='both')
            
            # Save data
            self.plot_data['6_2_bubble_chart'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_6_2_bubble_chart: {str(e)}")
            return None
    
    def plot_7_concepts_analysis(self, top_n=30):
        """7. Concepts analysis (top 30 concepts)"""
        try:
            if 'concepts_list' not in self.df_processed.columns:
                return None
            
            # Collect all concepts
            all_concepts = []
            for concepts in self.df_processed['concepts_list']:
                if isinstance(concepts, list):
                    all_concepts.extend([c.strip() for c in concepts])
            
            if len(all_concepts) == 0:
                return None
            
            concept_counts = pd.Series(all_concepts).value_counts()
            top_concepts = concept_counts.head(top_n)
            
            # Save data
            self.plot_data['7_concepts_analysis'] = {
                'top_concepts': top_concepts.to_dict(),
                'total_concepts': len(concept_counts)
            }
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9), dpi=600)
            
            # Graph 1: Bar chart
            y_pos = np.arange(len(top_concepts))
            colors = plt.cm.get_cmap(self.plot_settings['bar_palette'])(np.linspace(0.3, 0.9, len(top_concepts)))
            
            bars = ax1.barh(y_pos, top_concepts.values, color=colors, edgecolor='black')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(top_concepts.index, fontsize=8)
            ax1.set_xlabel('Frequency', fontweight='bold')
            ax1.set_title(f'Top {top_n} Research Concepts', fontweight='bold')
            ax1.invert_yaxis()
            
            # Add values
            for bar in bars:
                width = bar.get_width()
                ax1.text(width * 1.01, bar.get_y() + bar.get_height()/2,
                        f'{int(width)}', va='center', fontsize=8)
            
            # Graph 2: Word cloud
            fig_height = fig.get_size_inches()[1]
            wordcloud_height = fig_height * 0.8
            wordcloud_width = wordcloud_height * 1.6
            
            wordcloud = WordCloud(width=int(wordcloud_width*100), 
                                height=int(wordcloud_height*100), 
                                background_color='white',
                                colormap=self.plot_settings['sequential_palette'], 
                                max_words=100).generate_from_frequencies(concept_counts.to_dict())
            
            ax2.imshow(wordcloud, interpolation='bilinear')
            ax2.axis('off')
            ax2.set_title('Concept Word Cloud', fontweight='bold')
            
            plt.suptitle('Research Concepts Analysis', fontweight='bold', fontsize=14)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_7_concepts_analysis: {str(e)}")
            return None
    
    def plot_8_concept_cooccurrence(self, top_n=15):
        """8. Concept co-occurrence matrix"""
        try:
            if 'concepts_list' not in self.df_processed.columns:
                return None
            
            # Collect top concepts
            all_concepts = []
            for concepts in self.df_processed['concepts_list']:
                if isinstance(concepts, list):
                    all_concepts.extend([c.strip() for c in concepts])
            
            if len(all_concepts) == 0:
                return None
            
            concept_counts = pd.Series(all_concepts).value_counts()
            top_concepts = concept_counts.head(top_n).index.tolist()
            
            # Create co-occurrence matrix
            cooccurrence = pd.DataFrame(0, index=top_concepts, columns=top_concepts)
            
            for concepts in self.df_processed['concepts_list']:
                if isinstance(concepts, list):
                    concepts_clean = [c.strip() for c in concepts if c.strip() in top_concepts]
                    for i in range(len(concepts_clean)):
                        for j in range(i+1, len(concepts_clean)):
                            c1, c2 = concepts_clean[i], concepts_clean[j]
                            cooccurrence.loc[c1, c2] += 1
                            cooccurrence.loc[c2, c1] += 1
            
            # Save data
            self.plot_data['8_concept_cooccurrence'] = {
                'matrix': cooccurrence.to_dict(),
                'concepts': top_concepts
            }
            
            fig, ax = plt.subplots(figsize=(12, 9), dpi=600)
            
            im = ax.imshow(cooccurrence.values, cmap=self.plot_settings['heatmap_palette'])
            
            ax.set_xticks(np.arange(len(top_concepts)))
            ax.set_yticks(np.arange(len(top_concepts)))
            ax.set_xticklabels(top_concepts, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(top_concepts, fontsize=8)
            
            # Add values
            for i in range(len(top_concepts)):
                for j in range(len(top_concepts)):
                    value = cooccurrence.iloc[i, j]
                    if value > 0:
                        ax.text(j, i, str(value),
                               ha="center", va="center",
                               color="white" if value > cooccurrence.values.max()/2 else "black",
                               fontsize=8, fontweight='bold')
            
            ax.set_title(f'Concept Co-occurrence Matrix (Top {top_n} Concepts)',
                        fontweight='bold', fontsize=14)
            
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Co-occurrence Frequency', rotation=90, fontsize=11)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_8_concept_cooccurrence: {str(e)}")
            return None
    
    def plot_9_concept_influence(self):
        """9. Influence of key concepts"""
        try:
            if 'concepts_list' not in self.df_processed.columns or 'max_citations' not in self.df_processed.columns:
                return None
            
            # Expand concepts and link with citations
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
            
            # Aggregate by concept
            concept_stats = concept_df.groupby('concept').agg({
                'max_citations': ['sum', 'mean', 'median'],
                'max_annual_citations': 'mean',
                'count': 'size'
            }).round(2)
            
            concept_stats.columns = ['total_citations', 'mean_citations', 'median_citations',
                                   'mean_annual_citations', 'num_papers']
            
            concept_stats = concept_stats[concept_stats['num_papers'] >= 2]
            concept_stats = concept_stats.sort_values('mean_citations', ascending=False).head(20)
            
            # Save data
            self.plot_data['9_concept_influence'] = concept_stats.reset_index().to_dict('records')
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 9), dpi=600)
            
            # Graph 1: Mean citations
            y_pos = np.arange(len(concept_stats))
            colors = plt.cm.get_cmap(self.plot_settings['bar_palette'])(np.linspace(0.3, 0.9, len(concept_stats)))
            
            bars1 = axes[0].barh(y_pos, concept_stats['mean_citations'], color=colors, edgecolor='black')
            axes[0].set_yticks(y_pos)
            axes[0].set_yticklabels(concept_stats.index, fontsize=8)
            axes[0].set_xlabel('Mean Citations per Paper', fontweight='bold')
            axes[0].set_title('Top Concepts by Average Citation Impact', fontweight='bold')
            axes[0].invert_yaxis()
            
            for bar, (idx, row) in zip(bars1, concept_stats.iterrows()):
                width = bar.get_width()
                axes[0].text(width * 1.01, bar.get_y() + bar.get_height()/2,
                           f'n={int(row["num_papers"])}', va='center', fontsize=8)
            
            # Graph 2: Bubble chart
            scatter = axes[1].scatter(concept_stats['mean_annual_citations'],
                                    concept_stats['mean_citations'],
                                    s=concept_stats['num_papers'] * 15,
                                    c=concept_stats['total_citations'],
                                    cmap=self.plot_settings['scatter_palette'],
                                    alpha=0.7,
                                    edgecolors='black',
                                    linewidth=0.5)
            
            axes[1].set_xlabel('Mean Annual Citations', fontweight='bold')
            axes[1].set_ylabel('Mean Total Citations', fontweight='bold')
            axes[1].set_title('Concept Impact: Annual vs Total Citations', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=axes[1])
            cbar.set_label('Total Citations (Sum)', fontweight='bold')
            
            # Add annotations for top 5
            for idx, row in concept_stats.head(5).iterrows():
                short_name = idx[:20] + '...' if len(idx) > 20 else idx
                axes[1].annotate(short_name,
                               xy=(row['mean_annual_citations'], row['mean_citations']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
            
            # Legend for bubble sizes
            legend_elements = []
            size_values = [2, 5, 10, 20]
            for num_papers in size_values:
                if num_papers <= concept_stats['num_papers'].max():
                    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                                 markerfacecolor='gray',
                                                 markersize=np.sqrt(num_papers * 15 / 10),
                                                 label=f'{num_papers} papers'))
            if legend_elements:
                axes[1].legend(handles=legend_elements, title='Number of Papers',
                              loc='lower right', fontsize=8, title_fontsize=9)
            
            plt.suptitle('Concept Influence Analysis', fontweight='bold', fontsize=14)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_9_concept_influence: {str(e)}")
            return None
    
    def plot_10_temporal_evolution(self):
        """10. Temporal evolution of publication activity and impact"""
        try:
            if 'year' not in self.df_processed.columns or 'max_citations' not in self.df_processed.columns:
                return None
            
            valid_data = self.df_processed.dropna(subset=['year', 'max_citations'])
            if len(valid_data) < 10:
                return None
            
            # Group by year
            year_stats = valid_data.groupby('year').agg({
                'max_citations': ['sum', 'mean'],
                'max_annual_citations': 'mean',
                'doi': 'count'
            }).round(2)
            
            year_stats.columns = ['total_citations', 'mean_citations', 'mean_annual_citations', 'num_papers']
            year_stats = year_stats.sort_index()
            
            # Save data
            self.plot_data['10_temporal_evolution'] = year_stats.reset_index().to_dict('records')
            
            fig, ax1 = plt.subplots(figsize=(12, 9), dpi=600)
            
            # Bars: number of publications
            ax1.bar(year_stats.index, year_stats['num_papers'], 
                   alpha=0.4, color='steelblue', label='Number of Papers', edgecolor='black')
            ax1.set_xlabel('Publication Year', fontweight='bold')
            ax1.set_ylabel('Number of Papers', fontweight='bold', color='steelblue')
            ax1.tick_params(axis='y', labelcolor='steelblue')
            
            # Line: total citations (right axis)
            ax2 = ax1.twinx()
            line1 = ax2.plot(year_stats.index, year_stats['total_citations'], 
                           'o-', color='darkorange', linewidth=2.5, markersize=6,
                           label='Total Citations')
            ax2.set_ylabel('Total Citations', fontweight='bold', color='darkorange')
            ax2.tick_params(axis='y', labelcolor='darkorange')
            
            # Line: mean citations (additional)
            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('outward', 60))
            line2 = ax3.plot(year_stats.index, year_stats['mean_citations'], 
                           's-', color='darkgreen', linewidth=2, markersize=5,
                           label='Mean Citations per Paper')
            ax3.set_ylabel('Mean Citations per Paper', fontweight='bold', color='darkgreen')
            ax3.tick_params(axis='y', labelcolor='darkgreen')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left', fontsize=9)
            
            ax1.set_title('Temporal Evolution: Publications and Citation Impact',
                        fontweight='bold', fontsize=14)
            ax1.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_10_temporal_evolution: {str(e)}")
            return None
    
    def plot_11_temporal_heatmap(self):
        """11. Heatmap: Publication year vs Article age"""
        try:
            if 'year' not in self.df_processed.columns or 'max_annual_citations' not in self.df_processed.columns:
                return None
            
            valid_data = self.df_processed.dropna(subset=['year', 'max_annual_citations'])
            if len(valid_data) < 10:
                return None
            
            # Create data for heatmap
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
            
            # Create pivot table
            pivot_table = heatmap_df.pivot_table(
                values='annual_citations',
                index='age',
                columns='pub_year',
                aggfunc='mean',
                fill_value=0
            ).sort_index(ascending=False)
            
            # Save data
            self.plot_data['11_temporal_heatmap'] = {
                'pivot_data': pivot_table.to_dict(),
                'years': pivot_table.columns.tolist(),
                'ages': pivot_table.index.tolist()
            }
            
            fig, ax = plt.subplots(figsize=(12, 9), dpi=600)
            
            im = ax.imshow(pivot_table.values, cmap=self.plot_settings['heatmap_palette'], 
                          aspect='auto', vmin=0)
            
            ax.set_xticks(np.arange(len(pivot_table.columns)))
            ax.set_yticks(np.arange(len(pivot_table.index)))
            ax.set_xticklabels(pivot_table.columns.astype(int), rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(pivot_table.index, fontsize=9)
            ax.set_xlabel('Publication Year', fontweight='bold')
            ax.set_ylabel('Article Age (Years)', fontweight='bold')
            ax.set_title('Annual Citation Rate by Publication Year and Article Age',
                        fontweight='bold', fontsize=14)
            
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Mean Annual Citations (max(CR, OA))', rotation=90, fontsize=11)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_11_temporal_heatmap: {str(e)}")
            return None
    
    def plot_11_team_size_analysis(self):
        """11. Team size analysis"""
        try:
            if 'author count' not in self.df_processed.columns:
                return None
            
            # Categorize team size
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
            
            # Group data
            group_stats = self.df_processed.groupby('team_size_group').agg({
                'count': ['mean', 'median', 'std', 'size'],
                'Citation counts (CR)': 'mean',
                'references_count': 'mean'
            }).round(2)
            
            group_stats.columns = ['mean_attention', 'median_attention', 'std_attention',
                                 'num_papers', 'mean_citations', 'mean_references']
            
            # Order by increasing author count
            custom_order = ['Single author', '2 authors', '3 authors', '4-5 authors', 
                          '6-8 authors', '9-12 authors', '13+ authors', 'Unknown']
            
            existing_categories = [cat for cat in custom_order if cat in group_stats.index]
            group_stats = group_stats.loc[existing_categories]
            
            # Save data
            self.plot_data['11_team_size_analysis'] = group_stats.reset_index().to_dict('records')
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=600)
            axes = axes.flatten()
            
            # Graph 1: Distribution of team sizes
            team_size_counts = self.df_processed['team_size_group'].value_counts()
            team_size_counts = team_size_counts.reindex(existing_categories, fill_value=0)
            colors = plt.cm.get_cmap(self.plot_settings['bar_palette'])(np.linspace(0.3, 0.9, len(team_size_counts)))
            axes[0].bar(team_size_counts.index, team_size_counts.values,
                       alpha=0.7, color=colors, edgecolor='black')
            axes[0].set_xlabel('Team Size', fontweight='bold')
            axes[0].set_ylabel('Number of Papers', fontweight='bold')
            axes[0].set_title('Distribution of Team Sizes', fontweight='bold')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # Graph 2: Mean attention by team size
            axes[1].bar(group_stats.index, group_stats['mean_attention'],
                       alpha=0.7, color=colors, edgecolor='black')
            axes[1].errorbar(group_stats.index, group_stats['mean_attention'],
                           yerr=group_stats['std_attention'],
                           fmt='none', color='black', capsize=5)
            axes[1].set_xlabel('Team Size', fontweight='bold')
            axes[1].set_ylabel('Mean Attention', fontweight='bold')
            axes[1].set_title('Mean Attention by Team Size', fontweight='bold')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # Graph 3: Mean citations
            axes[2].bar(group_stats.index, group_stats['mean_citations'],
                       alpha=0.7, color=colors, edgecolor='black')
            axes[2].set_xlabel('Team Size', fontweight='bold')
            axes[2].set_ylabel('Mean Citations (CR)', fontweight='bold')
            axes[2].set_title('Mean Citations by Team Size', fontweight='bold')
            axes[2].tick_params(axis='x', rotation=45)
            axes[2].grid(True, alpha=0.3, axis='y')
            
            # Graph 4: Mean references
            axes[3].bar(group_stats.index, group_stats['mean_references'],
                       alpha=0.7, color=colors, edgecolor='black')
            axes[3].set_xlabel('Team Size', fontweight='bold')
            axes[3].set_ylabel('Mean References', fontweight='bold')
            axes[3].set_title('Mean References by Team Size', fontweight='bold')
            axes[3].tick_params(axis='x', rotation=45)
            axes[3].grid(True, alpha=0.3, axis='y')
            
            plt.suptitle('Team Size Analysis', fontweight='bold', fontsize=14)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_11_team_size_analysis: {str(e)}")
            return None
    
    def plot_12_correlation_matrix(self):
        """12. Correlation matrix with key parameters highlighted"""
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
            
            # Reorder matrix: key parameters first
            key_params = ['count', 'max_citations', 'max_annual_citations',
                         'Annual cit counts (CR)', 'Annual cit counts (OA)',
                         'Citation counts (CR)', 'Citation counts (OA)']
            
            existing_key_params = [p for p in key_params if p in corr_matrix.columns]
            other_params = [p for p in corr_matrix.columns if p not in existing_key_params]
            
            new_order = existing_key_params + other_params
            corr_matrix = corr_matrix.reindex(index=new_order, columns=new_order)
            
            # Save data
            self.plot_data['12_correlation_matrix'] = {
                'correlation_matrix': corr_matrix.to_dict(),
                'columns': available_cols,
                'method': 'spearman',
                'key_parameters': existing_key_params
            }
            
            fig, ax = plt.subplots(figsize=(12, 9), dpi=600)
            
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Draw heatmap
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                       cmap=self.plot_settings['heatmap_palette'], center=0, square=True,
                       linewidths=0.5, cbar_kws={'shrink': 0.8},
                       ax=ax, annot_kws={'fontsize': 8})
            
            # Highlight key parameters
            key_param_indices = [i for i, col in enumerate(corr_matrix.columns) if col in existing_key_params]
            for idx in key_param_indices:
                ax.add_patch(plt.Rectangle((0, idx), len(corr_matrix), 1, 
                                          fill=False, edgecolor='red', linewidth=2, 
                                          alpha=0.7))
                ax.add_patch(plt.Rectangle((idx, 0), 1, len(corr_matrix), 
                                          fill=False, edgecolor='red', linewidth=2, 
                                          alpha=0.7))
            
            legend_elements = [mpatches.Patch(facecolor='white', edgecolor='red', linewidth=2,
                                           alpha=0.7, label='Key parameters (Count & Citations)')]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
            
            ax.set_title('Correlation Matrix of Research Metrics (Spearman)\nKey Parameters Highlighted in Red', 
                        fontweight='bold', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(fontsize=8)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_12_correlation_matrix: {str(e)}")
            return None
    
    def plot_13_cr_vs_oa_comparison(self):
        """13. CR vs OA citations comparison"""
        try:
            required_cols = ['Citation counts (CR)', 'Citation counts (OA)']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) == 0:
                return None
            
            # Calculate difference
            valid_data['citation_diff'] = valid_data['Citation counts (OA)'] - valid_data['Citation counts (CR)']
            valid_data['citation_ratio'] = valid_data['Citation counts (OA)'] / valid_data['Citation counts (CR)'].replace(0, 1)
            
            # Save data
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
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9), dpi=600)
            
            # Graph 1: Scatter plot
            max_val = max(valid_data['Citation counts (CR)'].max(),
                         valid_data['Citation counts (OA)'].max())
            
            ax1.scatter(valid_data['Citation counts (CR)'],
                       valid_data['Citation counts (OA)'],
                       alpha=0.6, c='steelblue', edgecolors='black', linewidth=0.5)
            ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y = x')
            
            ax1.set_xlabel('Citations from Crossref (CR)', fontweight='bold')
            ax1.set_ylabel('Citations from OpenAlex (OA)', fontweight='bold')
            ax1.set_title('Comparison of Citation Counts', fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # Graph 2: Histogram of differences
            ax2.hist(valid_data['citation_diff'], bins=30,
                    alpha=0.7, color='darkorange', edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('Difference (OA - CR)', fontweight='bold')
            ax2.set_ylabel('Number of Articles', fontweight='bold')
            ax2.set_title('Distribution of Citation Differences', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Statistics
            stats_text = f"Mean difference: {valid_data['citation_diff'].mean():.1f}\n"
            stats_text += f"OA > CR: {(valid_data['citation_diff'] > 0).sum()} articles\n"
            stats_text += f"CR > OA: {(valid_data['citation_diff'] < 0).sum()} articles"
            
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.suptitle('Comparison of Citation Sources', fontweight='bold', fontsize=14)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_13_cr_vs_oa_comparison: {str(e)}")
            return None
    
    def plot_14_citation_by_domain(self):
        """14. Citation impact by research domain"""
        try:
            required_cols = ['Domain', 'Annual cit counts (CR)']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) == 0:
                return None
            
            # Aggregate by domain
            domain_stats = valid_data.groupby('Domain').agg({
                'Annual cit counts (CR)': ['median', 'mean', 'std', 'count'],
                'count': 'mean'
            }).round(2)
            
            domain_stats.columns = ['median_citations', 'mean_citations', 'std_citations',
                                  'num_papers', 'mean_attention']
            domain_stats = domain_stats.sort_values('median_citations', ascending=False)
            
            # Save data
            self.plot_data['14_citation_by_domain'] = domain_stats.reset_index().to_dict('records')
            
            # Select top domains
            top_domains = domain_stats.head(15).index.tolist()
            filtered_data = valid_data[valid_data['Domain'].isin(top_domains)]
            
            fig, ax = plt.subplots(figsize=(12, 9), dpi=600)
            
            # Boxplot
            box_data = []
            labels = []
            for domain in top_domains:
                data = filtered_data[filtered_data['Domain'] == domain]['Annual cit counts (CR)'].values
                if len(data) > 0:
                    box_data.append(data)
                    labels.append(domain)
            
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True, showfliers=False)
            
            colors = plt.cm.get_cmap(self.plot_settings['bar_palette'])(np.linspace(0.2, 0.8, len(box_data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add mean values
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
        """15. Cumulative influence curve"""
        try:
            if 'count' not in self.df_processed.columns:
                return None
            
            valid_data = self.df_processed.dropna(subset=['count'])
            if len(valid_data) == 0:
                return None
            
            # Sort by local citations
            sorted_counts = valid_data['count'].sort_values(ascending=False).reset_index(drop=True)
            
            # Calculate cumulative sums
            total_citations = sorted_counts.sum()
            cumulative_citations = sorted_counts.cumsum()
            cumulative_percentage = cumulative_citations / total_citations * 100
            article_percentage = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100
            
            # Save data
            self.plot_data['15_cumulative_influence'] = {
                'sorted_counts': sorted_counts.tolist(),
                'cumulative_percentage': cumulative_percentage.tolist(),
                'article_percentage': article_percentage.tolist(),
                'total_citations': float(total_citations),
                'gini_coefficient': self._calculate_gini(sorted_counts.values)
            }
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9), dpi=600)
            
            # Graph 1: Cumulative curve
            ax1.plot(article_percentage, cumulative_percentage,
                    linewidth=2.5, color='darkgreen')
            ax1.fill_between(article_percentage, 0, cumulative_percentage,
                            alpha=0.3, color='lightgreen')
            
            # 20/80 line
            twenty_percent_idx = int(len(sorted_counts) * 0.2)
            twenty_percent_cites = cumulative_percentage.iloc[twenty_percent_idx]
            
            ax1.axvline(x=20, color='red', linestyle='--', linewidth=1.5,
                       label=f'Top 20%: {twenty_percent_cites:.1f}% of citations')
            ax1.axhline(y=80, color='blue', linestyle='--', linewidth=1.5,
                       label='80% of citations')
            
            ax1.set_xlabel('Percentage of Articles', fontweight='bold')
            ax1.set_ylabel('Percentage of Total Mentions', fontweight='bold')
            ax1.set_title('Cumulative Influence Curve (Lorenz Curve)', fontweight='bold')
            ax1.legend(loc='lower right', fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # Graph 2: Distribution
            log_bins = np.logspace(0, np.log10(sorted_counts.max() + 1), 20)
            ax2.hist(sorted_counts, bins=log_bins, alpha=0.7,
                    color='steelblue', edgecolor='black')
            ax2.set_xscale('log')
            ax2.set_xlabel('Number of Local Citations (log scale)', fontweight='bold')
            ax2.set_ylabel('Number of Articles', fontweight='bold')
            ax2.set_title('Distribution of Local Citation Counts', fontweight='bold')
            ax2.grid(True, alpha=0.3, which='both')
            
            # Statistics
            gini = self._calculate_gini(sorted_counts.values)
            stats_text = f"Total articles: {len(sorted_counts):,}\n"
            stats_text += f"Total mentions: {total_citations:,}\n"
            stats_text += f"Mean: {sorted_counts.mean():.2f}\n"
            stats_text += f"Median: {sorted_counts.median():.1f}\n"
            stats_text += f"Gini coefficient: {gini:.3f}"
            
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.suptitle('Analysis of Local Influence Within Dataset', fontweight='bold', fontsize=14)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_15_cumulative_influence: {str(e)}")
            return None
    
    def _calculate_gini(self, x):
        """Calculate Gini coefficient"""
        x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(x, dtype=float)
        if cumx[-1] == 0:
            return 0
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    
    def plot_16_references_vs_impact(self):
        """16. References vs impact"""
        try:
            required_cols = ['references_count', 'count', 'Annual cit counts (CR)']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) < 10:
                return None
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9), dpi=600)
            
            # Graph 1: References vs Attention
            scatter1 = ax1.scatter(valid_data['references_count'],
                                 valid_data['count'],
                                 c=valid_data['Annual cit counts (CR)'],
                                 cmap=self.plot_settings['scatter_palette'], 
                                 alpha=0.6, s=30,
                                 edgecolors='black', linewidth=0.5)
            
            # Linear regression if enabled
            if self.plot_settings['show_regression'] and len(valid_data) > 10:
                x = valid_data['references_count'].values
                y = valid_data['count'].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = intercept + slope * x_line
                ax1.plot(x_line, y_line, 'r--', linewidth=2,
                        label=f'r = {r_value:.3f}, p = {p_value:.3f}')
                ax1.legend(fontsize=9)
            
            ax1.set_xlabel('Number of References', fontweight='bold')
            ax1.set_ylabel('Local Mentions (count)', fontweight='bold')
            ax1.set_title('References vs Local Attention', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            cbar1 = plt.colorbar(scatter1, ax=ax1)
            cbar1.set_label('Annual Citations (CR)', fontweight='bold')
            
            # Graph 2: References vs Citations
            scatter2 = ax2.scatter(valid_data['references_count'],
                                 valid_data['Annual cit counts (CR)'],
                                 c=valid_data['count'],
                                 cmap=self.plot_settings['scatter_palette'], 
                                 alpha=0.6, s=30,
                                 edgecolors='black', linewidth=0.5)
            
            if self.plot_settings['show_regression'] and len(valid_data) > 10:
                x = valid_data['references_count'].values
                y = valid_data['Annual cit counts (CR)'].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = intercept + slope * x_line
                ax2.plot(x_line, y_line, 'r--', linewidth=2,
                        label=f'r = {r_value:.3f}, p = {p_value:.3f}')
                ax2.legend(fontsize=9)
            
            ax2.set_xlabel('Number of References', fontweight='bold')
            ax2.set_ylabel('Annual Citations (CR)', fontweight='bold')
            ax2.set_title('References vs Citation Impact', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            cbar2 = plt.colorbar(scatter2, ax=ax2)
            cbar2.set_label('Local Mentions', fontweight='bold')
            
            plt.suptitle('Impact of Reference Count on Research Metrics', fontweight='bold', fontsize=14)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_16_references_vs_impact: {str(e)}")
            return None
    
    def plot_17_journal_impact(self):
        """17. Journal impact analysis"""
        try:
            required_cols = ['Full journal Name', 'count', 'Annual cit counts (CR)']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) == 0:
                return None
            
            # Aggregate by journal
            journal_stats = valid_data.groupby('Full journal Name').agg({
                'count': ['mean', 'median', 'std', 'size'],
                'Annual cit counts (CR)': 'mean',
                'references_count': 'mean'
            }).round(2)
            
            journal_stats.columns = ['mean_attention', 'median_attention', 'std_attention',
                                   'num_papers', 'mean_citations', 'mean_references']
            
            # Filter journals with sufficient papers
            journal_stats = journal_stats[journal_stats['num_papers'] >= 3]
            journal_stats = journal_stats.sort_values('mean_attention', ascending=False)
            
            # Save data
            self.plot_data['17_journal_impact'] = journal_stats.reset_index().to_dict('records')
            
            # Select top journals
            top_journals = journal_stats.head(15)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9), dpi=600)
            
            # Graph 1: Mean attention
            y_pos = np.arange(len(top_journals))
            colors1 = plt.cm.get_cmap(self.plot_settings['bar_palette'])(np.linspace(0.4, 0.9, len(top_journals)))
            
            bars1 = ax1.barh(y_pos, top_journals['mean_attention'],
                            color=colors1, edgecolor='black', alpha=0.8)
            
            ax1.set_yticks(y_pos)
            journal_names = [name[:25] + '...' if len(name) > 25 else name
                            for name in top_journals.index]
            ax1.set_yticklabels(journal_names, fontsize=8)
            ax1.set_xlabel('Mean Attention per Paper', fontweight='bold')
            ax1.set_title('Top Journals by Attention', fontweight='bold')
            ax1.invert_yaxis()
            
            # Add values
            for bar, (_, row) in zip(bars1, top_journals.iterrows()):
                width = bar.get_width()
                info_text = f"n={int(row['num_papers'])}"
                ax1.text(width * 1.01, bar.get_y() + bar.get_height()/2,
                        info_text, va='center', fontsize=8)
            
            # Graph 2: Bubble chart
            scatter = ax2.scatter(top_journals['mean_citations'],
                                top_journals['mean_attention'],
                                s=top_journals['num_papers'] * 10,
                                c=top_journals['mean_references'],
                                cmap=self.plot_settings['scatter_palette'], 
                                alpha=0.7,
                                edgecolors='black', linewidth=0.5)
            
            ax2.set_xlabel('Mean Annual Citations (CR)', fontweight='bold')
            ax2.set_ylabel('Mean Attention', fontweight='bold')
            ax2.set_title('Journal Impact: Citations vs Attention', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Mean References', fontweight='bold')
            
            # Add annotations
            for idx, row in top_journals.head(5).iterrows():
                short_name = idx[:15] + '...' if len(idx) > 15 else idx
                ax2.annotate(short_name,
                            xy=(row['mean_citations'], row['mean_attention']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, alpha=0.8)
            
            # Legend for bubble sizes
            legend_elements = []
            size_values = [3, 6, 9, 12]
            for num_papers in size_values:
                if num_papers <= top_journals['num_papers'].max():
                    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                                 markerfacecolor='gray',
                                                 markersize=np.sqrt(num_papers * 10 / 10),
                                                 label=f'{num_papers} papers'))
            if legend_elements:
                ax2.legend(handles=legend_elements, title='Number of Papers',
                          loc='lower right', fontsize=8, title_fontsize=9)
            
            plt.suptitle('Journal Impact Analysis', fontweight='bold', fontsize=14)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_17_journal_impact: {str(e)}")
            return None
    
    def plot_18_author_collaboration_network(self, top_n=30):
        """18. Author collaboration network (NEW PLOT)"""
        try:
            if 'authors_list' not in self.df_processed.columns:
                return None
            
            # Create graph
            G = nx.Graph()
            
            for idx, row in self.df_processed.iterrows():
                if isinstance(row['authors_list'], list) and len(row['authors_list']) >= 2:
                    authors = [a.strip() for a in row['authors_list']]
                    weight = row.get('count', 1)
                    
                    # Add nodes
                    for author in authors:
                        if not G.has_node(author):
                            G.add_node(author, weight=0, papers=0)
                        G.nodes[author]['weight'] += weight
                        G.nodes[author]['papers'] += 1
                    
                    # Add edges between all pairs
                    for i in range(len(authors)):
                        for j in range(i+1, len(authors)):
                            if G.has_edge(authors[i], authors[j]):
                                G[authors[i]][authors[j]]['weight'] += weight
                                G[authors[i]][authors[j]]['papers'] += 1
                            else:
                                G.add_edge(authors[i], authors[j], weight=weight, papers=1)
            
            if len(G.nodes()) < 3:
                self.log_warning("Insufficient data for author collaboration network")
                return None
            
            # Select top authors by degree
            degree_dict = dict(G.degree(weight='weight'))
            top_authors = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_author_names = [author[0] for author in top_authors]
            
            H = G.subgraph(top_author_names)
            
            # Remove isolated nodes
            isolated_nodes = [n for n in H.nodes() if H.degree(n) == 0]
            H.remove_nodes_from(isolated_nodes)
            
            if len(H.nodes()) < 2:
                self.log_warning("Insufficient connections in author network")
                return None
            
            # Save data
            self.plot_data['18_author_collaboration_network'] = {
                'nodes': [{'author': node, 'weight': H.nodes[node]['weight'],
                          'papers': H.nodes[node]['papers']} for node in H.nodes()],
                'edges': [{'author1': u, 'author2': v, 'weight': H[u][v]['weight'],
                          'papers': H[u][v]['papers']} for u, v in H.edges()]
            }
            
            # Visualization
            fig, ax = plt.subplots(figsize=(12, 9), dpi=600)
            
            pos = nx.spring_layout(H, k=2, seed=42)
            
            # Node sizes by weight
            node_sizes = [H.nodes[n]['weight'] * 0.5 + 500 for n in H.nodes()]
            node_colors = [H.nodes[n]['papers'] for n in H.nodes()]
            
            nodes = nx.draw_networkx_nodes(H, pos, node_size=node_sizes,
                                          node_color=node_colors, 
                                          cmap=self.plot_settings['network_palette'],
                                          alpha=0.8, edgecolors='black', linewidths=1.5, ax=ax)
            
            # Edges with width by weight
            if H.edges():
                edge_weights = [H[u][v]['weight'] * 0.1 for u, v in H.edges()]
                edges = nx.draw_networkx_edges(H, pos, width=edge_weights, alpha=0.5,
                                              edge_color='gray', style='solid', ax=ax)
            
            # Labels (truncated if too long)
            labels = {}
            for node in H.nodes():
                if len(node) > 25:
                    labels[node] = node[:22] + '...'
                else:
                    labels[node] = node
            
            nx.draw_networkx_labels(H, pos, labels=labels, font_size=7, font_weight='bold', ax=ax)
            
            ax.set_title(f'Author Collaboration Network (Top {top_n} Authors by Collaboration Strength)',
                        fontweight='bold', fontsize=14, pad=20)
            ax.axis('off')
            
            # Colorbar
            sm = plt.cm.ScalarMappable(cmap=self.plot_settings['network_palette'],
                                      norm=plt.Normalize(vmin=min(node_colors), 
                                                       vmax=max(node_colors)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Number of Papers', fontweight='bold', fontsize=9)
            
            # Statistics
            stats_text = f"Authors: {len(H.nodes())} | Collaborations: {len(H.edges())}"
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Legend for node sizes
            size_legend_elements = []
            size_values = [500, 1000, 1500, 2000]
            for size in size_values:
                weight_value = (size - 500) / 0.5
                if weight_value > 0:
                    size_legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                                      markerfacecolor='gray',
                                                      markersize=np.sqrt(size/10),
                                                      label=f'Weight ~{weight_value:.0f}'))
            if size_legend_elements:
                ax.legend(handles=size_legend_elements, title='Collaboration Weight',
                         loc='upper right', fontsize=8, title_fontsize=9)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_18_author_collaboration_network: {str(e)}")
            return None
    
    def plot_18_18_1_affiliation_network(self):
        """18.1 Affiliation network (Top 20)"""
        try:
            return self._plot_affiliation_network_impl(20, "1")
        except Exception as e:
            self.log_error(f"Error in plot_18_18_1_affiliation_network: {str(e)}")
            return None
    
    def plot_18_18_2_affiliation_network(self):
        """18.2 Affiliation network (Top 30)"""
        try:
            return self._plot_affiliation_network_impl(30, "2")
        except Exception as e:
            self.log_error(f"Error in plot_18_18_2_affiliation_network: {str(e)}")
            return None
    
    def plot_18_18_3_affiliation_network(self):
        """18.3 Affiliation network (Top 50)"""
        try:
            return self._plot_affiliation_network_impl(50, "3")
        except Exception as e:
            self.log_error(f"Error in plot_18_18_3_affiliation_network: {str(e)}")
            return None
    
    def _plot_affiliation_network_impl(self, top_n, suffix):
        """Implementation of affiliation network"""
        try:
            if 'affiliations_list' not in self.df_processed.columns:
                return None
            
            # Create graph
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
            
            # Select top affiliations
            degree_dict = dict(G.degree(weight='weight'))
            top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_node_names = [node[0] for node in top_nodes]
            
            H = G.subgraph(top_node_names)
            
            # Remove isolated nodes
            isolated_nodes = [n for n in H.nodes() if H.degree(n) == 0]
            H.remove_nodes_from(isolated_nodes)
            
            if len(H.nodes()) < 2:
                self.log_warning("Insufficient connections in affiliation network")
                return None
            
            # Save data
            self.plot_data[f'18_18_{suffix}_affiliation_network'] = {
                'nodes': [{'affiliation': node, 'weight': H.nodes[node]['weight'],
                          'papers': H.nodes[node]['papers']} for node in H.nodes()],
                'edges': [{'aff1': u, 'aff2': v, 'weight': H[u][v]['weight'],
                          'papers': H[u][v]['papers']} for u, v in H.edges()]
            }
            
            fig, ax = plt.subplots(figsize=(12, 9), dpi=600)
            
            pos = nx.spring_layout(H, k=3, seed=42)
            
            node_sizes = [H.nodes[n]['weight'] * 0.3 + 300 for n in H.nodes()]
            node_colors = [H.nodes[n]['papers'] for n in H.nodes()]
            
            nodes = nx.draw_networkx_nodes(H, pos, node_size=node_sizes,
                                          node_color=node_colors, 
                                          cmap=self.plot_settings['network_palette'],
                                          alpha=0.8, edgecolors='black', linewidths=1.5, ax=ax)
            
            if H.edges():
                edge_weights = [H[u][v]['weight'] * 0.05 for u, v in H.edges()]
                edges = nx.draw_networkx_edges(H, pos, width=edge_weights, alpha=0.4,
                                              edge_color='gray', style='solid', ax=ax)
            
            # Labels with line breaks
            labels = {}
            for node in H.nodes():
                words = node.split()
                if len(node) > 30:
                    mid = len(words) // 2
                    labels[node] = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                else:
                    labels[node] = node
            
            nx.draw_networkx_labels(H, pos, labels=labels, font_size=7, font_weight='bold', ax=ax)
            
            ax.set_title(f'Top {top_n} Affiliation Collaboration Network (Version {suffix})', 
                        fontweight='bold', fontsize=14)
            ax.axis('off')
            
            # Colorbar
            sm = plt.cm.ScalarMappable(cmap=self.plot_settings['network_palette'],
                                      norm=plt.Normalize(vmin=min(node_colors),
                                                       vmax=max(node_colors)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Number of Papers', fontweight='bold', fontsize=9)
            
            # Statistics
            stats_text = f"Affiliations: {len(H.nodes())} | Collaborations: {len(H.edges())}"
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in _plot_affiliation_network_impl: {str(e)}")
            return None
    
    def plot_19_hierarchical_sankey(self):
        """19. Hierarchical Sankey diagram: Domain → Field → Subfield → Topic"""
        try:
            required_cols = ['Domain', 'Field', 'Subfield', 'Topic', 'max_citations']
            available_cols = [col for col in required_cols if col in self.df_processed.columns]
            
            if len(available_cols) < 3:
                return None
            
            valid_data = self.df_processed.dropna(subset=available_cols)
            if len(valid_data) < 10:
                return None
            
            # Create hierarchical links
            links = []
            nodes = []
            node_indices = {}
            
            def add_node(name):
                if name not in node_indices:
                    node_indices[name] = len(nodes)
                    nodes.append(name)
                return node_indices[name]
            
            # Aggregate weights (total citations)
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
                
                # Add connections
                domain_idx = add_node(domain)
                field_idx = add_node(field)
                subfield_idx = add_node(subfield)
                topic_idx = add_node(topic)
                
                links.append({'source': domain_idx, 'target': field_idx, 'value': weight})
                links.append({'source': field_idx, 'target': subfield_idx, 'value': weight})
                links.append({'source': subfield_idx, 'target': topic_idx, 'value': weight})
            
            # Save data
            self.plot_data['19_hierarchical_sankey'] = {
                'nodes': nodes,
                'links': links,
                'total_weight': sum([l['value'] for l in links])
            }
            
            # Create Sankey diagram with plotly
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
        """20. Multidimensional scaling of important predictors"""
        try:
            # Select key predictors
            predictors = ['author count', 'references_count', 'num_countries',
                         'Annual cit counts (CR)', 'article_age', 'normalized_attention']
            
            available_predictors = [p for p in predictors if p in self.df_processed.columns]
            
            if len(available_predictors) < 3:
                return None
            
            # Prepare data
            analysis_data = self.df_processed[available_predictors + ['count']].dropna()
            
            if len(analysis_data) < 20:
                return None
            
            # Standardization
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(analysis_data[available_predictors])
            
            # PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            
            # Save data
            self.plot_data['20_mds_analysis'] = {
                'explained_variance': pca.explained_variance_ratio_.tolist(),
                'components': pca.components_.tolist(),
                'pca_coordinates': pca_result.tolist(),
                'predictors': available_predictors
            }
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9), dpi=600)
            
            # Graph 1: PCA scatter plot
            scatter = ax1.scatter(pca_result[:, 0], pca_result[:, 1],
                                 c=analysis_data['count'], 
                                 cmap=self.plot_settings['scatter_palette'],
                                 alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
            
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontweight='bold')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontweight='bold')
            ax1.set_title('PCA: Multidimensional Scaling of Predictors', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Local Mentions (count)', fontweight='bold')
            
            # Graph 2: Predictor loadings
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            
            for i, predictor in enumerate(available_predictors):
                ax2.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                         color='red', alpha=0.5, head_width=0.05)
                ax2.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15,
                        predictor, color='red', fontsize=9, fontweight='bold')
            
            # Correlation circle
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
            
            plt.suptitle('Multidimensional Analysis of Research Predictors', fontweight='bold', fontsize=14)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_20_multidimensional_scaling: {str(e)}")
            return None
    
    def plot_21_concept_network_weighted(self):
        """21. Weighted concept network by influence"""
        try:
            if 'concepts_list' not in self.df_processed.columns or 'max_citations' not in self.df_processed.columns:
                return None
            
            # Collect top concepts by frequency
            all_concepts = []
            for concepts in self.df_processed['concepts_list']:
                if isinstance(concepts, list):
                    all_concepts.extend([c.strip() for c in concepts])
            
            if len(all_concepts) == 0:
                return None
            
            concept_counts = pd.Series(all_concepts).value_counts()
            top_concepts = concept_counts.head(25).index.tolist()
            
            # Create weighted graph
            G = nx.Graph()
            
            # Add nodes with citation weights
            for concept in top_concepts:
                concept_papers = []
                for idx, row in self.df_processed.iterrows():
                    if isinstance(row['concepts_list'], list) and concept in row['concepts_list']:
                        concept_papers.append(row.get('max_citations', 0))
                
                total_citations = sum(concept_papers)
                G.add_node(concept, citations=total_citations, papers=len(concept_papers))
            
            # Add edges with co-occurrence weights
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
            
            # Save data
            self.plot_data['21_concept_network_weighted'] = {
                'nodes': [{'concept': node, 'citations': G.nodes[node]['citations'],
                          'papers': G.nodes[node]['papers']} for node in G.nodes()],
                'edges': [{'concept1': u, 'concept2': v, 'weight': G[u][v]['weight']} 
                         for u, v in G.edges()]
            }
            
            # Visualization
            fig, ax = plt.subplots(figsize=(12, 9), dpi=600)
            
            pos = nx.spring_layout(G, k=2, seed=42)
            
            # Node sizes by citations
            node_sizes = [G.nodes[n]['citations'] * 0.2 + 500 for n in G.nodes()]
            node_colors = [G.nodes[n]['papers'] for n in G.nodes()]
            
            nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                          node_color=node_colors, 
                                          cmap=self.plot_settings['network_palette'],
                                          alpha=0.8, edgecolors='black', linewidths=1.5, ax=ax)
            
            # Edges with width by weight
            if G.edges():
                edge_weights = [G[u][v]['weight'] * 0.01 for u, v in G.edges()]
                edges = nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5,
                                              edge_color='gray', style='solid', ax=ax)
            
            # Labels
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
            
            ax.set_title('Concept Network with Citation Impact', fontweight='bold', fontsize=14)
            ax.axis('off')
            
            # Colorbar
            sm = plt.cm.ScalarMappable(cmap=self.plot_settings['network_palette'],
                                      norm=plt.Normalize(vmin=min(node_colors),
                                                       vmax=max(node_colors)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Number of Papers', fontweight='bold', fontsize=9)
            
            # Legend for node sizes
            size_legend_elements = []
            size_values = [500, 1000, 1500, 2000]
            for size in size_values:
                citation_value = (size - 500) / 0.2
                if citation_value > 0:
                    size_legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                                      markerfacecolor='gray',
                                                      markersize=np.sqrt(size/10),
                                                      label=f'Citations ~{citation_value:.0f}'))
            if size_legend_elements:
                ax.legend(handles=size_legend_elements, title='Citation Impact',
                         loc='upper right', fontsize=8, title_fontsize=9)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_21_concept_network_weighted: {str(e)}")
            return None
    
    def generate_all_plots(self, selected_plots=None):
        """Generate all plots with progress bar"""
        self.all_figures = {}
        self.plot_data = {}
        self.errors = []
        self.warnings = []
        
        # List of all plot functions
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
            ("18_author_network", "19. Author Collaboration Network", lambda: self.plot_18_author_collaboration_network(30)),
            ("18_18_1_affiliation_network", "20.1 Affiliation Network (Top 20)", self.plot_18_18_1_affiliation_network),
            ("18_18_2_affiliation_network", "20.2 Affiliation Network (Top 30)", self.plot_18_18_2_affiliation_network),
            ("18_18_3_affiliation_network", "20.3 Affiliation Network (Top 50)", self.plot_18_18_3_affiliation_network),
            ("19_hierarchical_sankey", "21. Hierarchical Sankey Diagram", self.plot_19_hierarchical_sankey),
            ("20_mds", "22. Multidimensional Scaling", self.plot_20_multidimensional_scaling),
            ("21_concept_network_weighted", "23. Weighted Concept Network", self.plot_21_concept_network_weighted)
        ]
        
        # If specific plots selected
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
        """Create Excel file with data for all plots"""
        if not self.plot_data:
            st.warning("⚠️ No plot data available for Excel report")
            return None
        
        # Create Excel writer
        excel_buffer = io.BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # 1. Basic statistics
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
            
            # 2. Data for each plot
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
            
            # 3. Errors and warnings
            if self.errors:
                errors_df = pd.DataFrame(self.errors)
                errors_df.to_excel(writer, sheet_name='Errors', index=False)
            
            if self.warnings:
                warnings_df = pd.DataFrame(self.warnings)
                warnings_df.to_excel(writer, sheet_name='Warnings', index=False)
            
            # 4. Terminology and formulas sheet
            self._add_terminology_sheet(writer)
        
        excel_buffer.seek(0)
        return excel_buffer
    
    def _add_terminology_sheet(self, writer):
        """Add terminology and formulas sheet"""
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
                'article_age = 2026 - year (assuming current year 2026)',
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
        """Save all plots and reports to ZIP archive"""
        if not self.all_figures:
            st.error("❌ No plots to save!")
            return None
        
        # Create ZIP archive
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 1. Save plots
            for i, (name, fig) in enumerate(self.all_figures.items()):
                img_buffer = io.BytesIO()
                
                # Check if it's a plotly figure
                if hasattr(fig, 'write_image'):
                    # Plotly figure
                    import plotly.io as pio
                    pio.write_image(fig, img_buffer, format='png', width=1200, height=900, scale=2)
                else:
                    # Matplotlib figure
                    fig.savefig(img_buffer, format='png', dpi=600,
                              bbox_inches='tight', facecolor='white',
                              edgecolor='black')
                img_buffer.seek(0)
                
                filename = f"plot_{i+1:02d}_{name}.png"
                zip_file.writestr(filename, img_buffer.read())
                
                # Close matplotlib figures
                if not hasattr(fig, 'write_image'):
                    plt.close(fig)
            
            # 2. Save Excel report
            if include_excel:
                excel_buffer = self.create_excel_report()
                if excel_buffer:
                    zip_file.writestr("plot_data.xlsx", excel_buffer.read())
            
            # 3. Save metadata
            metadata = {
                'generated_date': datetime.now().isoformat(),
                'total_plots': len(self.all_figures),
                'dataset_statistics': {
                    'total_rows': len(self.df_processed) if self.df_processed is not None else 0,
                    'year_range': f"{int(self.df_processed['year'].min())}-{int(self.df_processed['year'].max())}" if self.df_processed is not None and 'year' in self.df_processed.columns else 'N/A',
                    'total_mentions': int(self.df_processed['count'].sum()) if self.df_processed is not None and 'count' in self.df_processed.columns else 0
                },
                'errors_count': len(self.errors),
                'warnings_count': len(self.warnings),
                'plot_settings': self.plot_settings
            }
            
            zip_file.writestr('metadata.json', json.dumps(metadata, indent=2))
        
        zip_buffer.seek(0)
        return zip_buffer

# ============================================================================
# STREAMLIT APPLICATION WITH MODULAR 4-STAGE INTERFACE
# ============================================================================

def main():
    """Main Streamlit application function"""
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = 1  # 1=Upload, 2=Settings, 3=Visualization, 4=Export
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ScientificDataAnalyzer()
    if 'plots_generated' not in st.session_state:
        st.session_state.plots_generated = False
    if 'selected_plots' not in st.session_state:
        st.session_state.selected_plots = None
    if 'sample_data_loaded' not in st.session_state:
        st.session_state.sample_data_loaded = ''
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Scientific Data Visualization Dashboard</h1>
        <p>Advanced Multi-Stage Analysis Tool for Research Metrics and Collaboration Networks</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        # Stage indicators
        stages = ["1. Data Upload", "2. Plot Selection", "3. Visualization", "4. Export"]
        current_stage = st.session_state.current_stage - 1
        
        for i, stage in enumerate(stages):
            if i == current_stage:
                st.markdown(f"**→ {stage}**")
            else:
                if i < current_stage:
                    if st.button(f"✓ {stage}", key=f"nav_{i}", use_container_width=True):
                        st.session_state.current_stage = i + 1
                        st.rerun()
                else:
                    st.markdown(f"   {stage}")
        
        st.markdown("---")
        
        # Data information (if loaded)
        if st.session_state.analyzer.df_processed is not None:
            st.markdown("### 📈 Data Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Papers", len(st.session_state.analyzer.df_processed))
            with col2:
                if 'count' in st.session_state.analyzer.df_processed.columns:
                    st.metric("Total Mentions", 
                             f"{int(st.session_state.analyzer.df_processed['count'].sum()):,}")
            
            if 'year' in st.session_state.analyzer.df_processed.columns:
                st.metric("Year Range", 
                         f"{int(st.session_state.analyzer.df_processed['year'].min())}-{int(st.session_state.analyzer.df_processed['year'].max())}")
        
        st.markdown("---")
        st.info("💡 **Tips:**\n• Use TSV format (tab-separated)\n• First row must be headers\n• Required: doi, Title, year, count")
    
    # Main content based on current stage
    if st.session_state.current_stage == 1:
        stage1_data_upload()
    elif st.session_state.current_stage == 2:
        stage2_plot_selection()
    elif st.session_state.current_stage == 3:
        stage3_visualization()
    elif st.session_state.current_stage == 4:
        stage4_export()

def stage1_data_upload():
    """Stage 1: Data upload and preprocessing"""
    st.header("📋 Stage 1: Data Upload")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.analyzer = ScientificDataAnalyzer()
            st.session_state.plots_generated = False
            st.session_state.selected_plots = None
            st.session_state.sample_data_loaded = ''
            st.rerun()
    
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    # Data input
    data_input = st.text_area(
        "📝 Paste your TSV data here",
        value=st.session_state.sample_data_loaded,
        height=300,
        help="First row should contain column headers. Columns should be tab-separated."
    )
    
    # Sample data button
    sample_data = """doi\tpublication_date\tTitle\tauthors\tORCID ID 1; ORCID ID 2... ORCID ID last\tauthor count\taffiliations {aff 1; aff 2... aff last}\tcountries {country 1; ... country last}\tFull journal Name\tyear\tVolume\tPages (or article number)\tCitation counts (CR)\tCitation counts (OA)\tAnnual cit counts (CR)\tAnnual cit counts (OA)\treferences_count\tcount\tTopic\tSubfield\tField\tDomain\tConcepts
10.1021/acs.chemrev.6b00284\t2016-11-09\tStrategies for Carbon and Sulfur Tolerant Solid Oxide Fuel Cell Materials, Incorporating Lessons from Heterogeneous Catalysis\tPaul Boldrin; Enrique Ruiz-Trejo; Joshua Mermelstein; José Miguel Bermúdez Menéndez; Tomás Ramı́rez Reina; Nigel P. Brandon\thttps://orcid.org/0000-0003-0058-6876; https://orcid.org/0000-0001-5560-5750; https://orcid.org/0000-0001-7211-2958; https://orcid.org/0000-0001-9693-5107; https://orcid.org/0000-0003-2230-8666\t6\tUniversity of Surrey; Imperial College London; Boeing (United States)\tUS; GB\tChemical Reviews\t2016\t116\t13633-13684\t289\t296\t26.27\t26.91\t465\t5\tAdvancements in Solid Oxide Fuel Cells\tChemistry\tCarbon fibers\tCatalysis\tSulfur; Chemistry; Carbon fibers; Catalysis; Oxide; Solid oxide fuel cell; Fuel cells; Nanotechnology; Environmental chemistry; Chemical engineering; Organic chemistry; Materials science; Engineering; Composite number; Physical chemistry; Composite material; Anode; Electrode
10.1126/science.aab3987\t2015-07-23\tReadily processed protonic ceramic fuel cells with high performance at low temperatures\tChuancheng Duan; Jianhua Tong; Meng Shang; Stefan Nikodemski; Michael Sanders; Sandrine Ricote; Ali Almansoori; Ryan O'Hayre\thttps://orcid.org/0000-0002-1826-1415; https://orcid.org/0000-0002-0684-1658; https://orcid.org/0000-0001-6366-5219; https://orcid.org/0000-0001-7565-0284; https://orcid.org/0000-0002-0789-5105; https://orcid.org/0000-0003-3762-3052\t8\tAmerican Petroleum Institute; Colorado School of Mines\tUS\tScience\t2015\t349\t1321-1326\t1325\t1352\t110.42\t112.67\t91\t5\tAdvancements in Solid Oxide Fuel Cells\tOxide\tFuel cells\tMaterials science\tCeramic; Oxide; Fuel cells; Materials science; Methane; Electrolyte; Chemical engineering; Cathode; Ion; Solid oxide fuel cell; Chemistry; Composite material; Electrode; Metallurgy; Organic chemistry; Engineering; Physical chemistry"""
    
    if st.button("📋 Load Sample Data", use_container_width=True):
        st.session_state.sample_data_loaded = sample_data
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load button
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📥 Load Data", type="primary", use_container_width=True):
            if data_input.strip():
                with st.spinner("Processing data..."):
                    result = st.session_state.analyzer.parse_data(data_input)
                    if result is not None:
                        st.success("✅ Data successfully loaded!")
                        st.session_state.plots_generated = False
                    else:
                        st.error("❌ Failed to parse data. Please check format.")
            else:
                st.error("❌ Please paste data first")
    
    with col2:
        if st.button("⬅️ Reset", use_container_width=True):
            st.session_state.analyzer = ScientificDataAnalyzer()
            st.session_state.plots_generated = False
            st.session_state.selected_plots = None
            st.session_state.sample_data_loaded = ''
            st.rerun()
    
    # Display data info if loaded
    if st.session_state.analyzer.df_processed is not None:
        st.markdown("---")
        st.subheader("📊 Data Preview")
        
        # Metrics row
        df = st.session_state.analyzer.df_processed
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Papers", len(df))
        with col2:
            if 'year' in df.columns:
                st.metric("Year Range", f"{int(df['year'].min())}-{int(df['year'].max())}")
        with col3:
            if 'count' in df.columns:
                st.metric("Total Mentions", f"{int(df['count'].sum()):,}")
        with col4:
            if 'max_citations' in df.columns:
                st.metric("Mean Max Citations", f"{df['max_citations'].mean():.1f}")
        
        with st.expander("🔍 View Data Table"):
            st.dataframe(df.head(10))
        
        # Navigation to next stage
        st.markdown("---")
        if st.button("➡️ Proceed to Plot Selection", type="primary", use_container_width=True):
            st.session_state.current_stage = 2
            st.rerun()

def stage2_plot_selection():
    """Stage 2: Plot selection and settings configuration"""
    st.header("🎨 Stage 2: Plot Selection & Settings")
    
    if st.session_state.analyzer.df_processed is None:
        st.warning("⚠️ No data loaded. Please go back to Stage 1 and upload data first.")
        if st.button("⬅️ Back to Data Upload"):
            st.session_state.current_stage = 1
            st.rerun()
        return
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back to Data Upload", use_container_width=True):
            st.session_state.current_stage = 1
            st.rerun()
    with col2:
        if st.button("🏠 Back to Start", use_container_width=True):
            st.session_state.current_stage = 1
            st.session_state.analyzer = ScientificDataAnalyzer()
            st.session_state.plots_generated = False
            st.session_state.selected_plots = None
            st.rerun()
    
    st.markdown("---")
    
    # All plots list
    all_plots = [
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
        ("18_author_network", "19. Author Collaboration Network"),
        ("18_18_1_affiliation_network", "20.1 Affiliation Network (Top 20)"),
        ("18_18_2_affiliation_network", "20.2 Affiliation Network (Top 30)"),
        ("18_18_3_affiliation_network", "20.3 Affiliation Network (Top 50)"),
        ("19_hierarchical_sankey", "21. Hierarchical Sankey Diagram"),
        ("20_mds", "22. Multidimensional Scaling"),
        ("21_concept_network_weighted", "23. Weighted Concept Network")
    ]
    
    # Plot selection section
    st.subheader("📊 Select Plots to Generate")
    
    # Select all buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Select All Plots", use_container_width=True):
            st.session_state.selected_plots = [plot[0] for plot in all_plots]
            st.rerun()
    with col2:
        if st.button("❌ Clear All", use_container_width=True):
            st.session_state.selected_plots = []
            st.rerun()
    
    st.markdown("---")
    
    # Group plots by category
    categories = {
        "📈 Distributions & Statistics": ["1_distribution", "15_cumulative_influence", "12_correlation"],
        "🌍 International Collaboration": ["2_country_network", "3_internationality", "5_collab_linear", "6_collab_log"],
        "📚 Journals & Publications": ["4_journal_heatmap", "17_journal_impact"],
        "🔗 References & Citations": ["6_1_bubble_chart", "6_2_bubble_chart", "13_cr_vs_oa", "16_references_impact"],
        "🏷️ Concepts & Topics": ["7_concepts", "8_concept_cooccurrence", "9_concept_influence", "21_concept_network_weighted"],
        "⏰ Temporal Analysis": ["10_temporal_evolution", "11_temporal_heatmap"],
        "👥 Teams & Organizations": ["11_team_size", "18_author_network", "18_18_1_affiliation_network", "18_18_2_affiliation_network", "18_18_3_affiliation_network"],
        "📊 Advanced Analysis": ["14_domain_citations", "20_mds", "19_hierarchical_sankey"]
    }
    
    # Initialize selected_plots if None
    if st.session_state.selected_plots is None:
        st.session_state.selected_plots = [plot[0] for plot in all_plots]
    
    # Display checkboxes by category
    for category, plot_ids in categories.items():
        with st.expander(category, expanded=True):
            cols = st.columns(2)
            for idx, plot_id in enumerate(plot_ids):
                plot_name = next(name for pid, name in all_plots if pid == plot_id)
                with cols[idx % 2]:
                    if st.checkbox(plot_name, 
                                 value=plot_id in st.session_state.selected_plots,
                                 key=f"cb_{plot_id}"):
                        if plot_id not in st.session_state.selected_plots:
                            st.session_state.selected_plots.append(plot_id)
                    else:
                        if plot_id in st.session_state.selected_plots:
                            st.session_state.selected_plots.remove(plot_id)
    
    st.markdown("---")
    
    # Plot settings section
    st.subheader("⚙️ Plot Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Regression toggle
        show_regression = st.toggle(
            "📈 Show Regression Lines", 
            value=st.session_state.analyzer.plot_settings['show_regression'],
            help="Display trend lines on scatter plots"
        )
        st.session_state.analyzer.plot_settings['show_regression'] = show_regression
    
    with col2:
        # Color palette selections
        st.markdown("**🎨 Color Palettes**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        heatmap_palette = st.selectbox(
            "Heatmap Palette",
            options=COLOR_PALETTES['heatmap'],
            index=COLOR_PALETTES['heatmap'].index(st.session_state.analyzer.plot_settings['heatmap_palette'])
        )
        st.session_state.analyzer.plot_settings['heatmap_palette'] = heatmap_palette
        
        scatter_palette = st.selectbox(
            "Scatter Plot Palette",
            options=COLOR_PALETTES['scatter'],
            index=COLOR_PALETTES['scatter'].index(st.session_state.analyzer.plot_settings['scatter_palette'])
        )
        st.session_state.analyzer.plot_settings['scatter_palette'] = scatter_palette
    
    with col2:
        bar_palette = st.selectbox(
            "Bar Chart Palette",
            options=COLOR_PALETTES['bar'],
            index=COLOR_PALETTES['bar'].index(st.session_state.analyzer.plot_settings['bar_palette'])
        )
        st.session_state.analyzer.plot_settings['bar_palette'] = bar_palette
        
        network_palette = st.selectbox(
            "Network Graph Palette",
            options=COLOR_PALETTES['network'],
            index=COLOR_PALETTES['network'].index(st.session_state.analyzer.plot_settings['network_palette'])
        )
        st.session_state.analyzer.plot_settings['network_palette'] = network_palette
    
    st.markdown("---")
    
    # Generate button
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🚀 Generate Selected Plots", type="primary", use_container_width=True):
            if not st.session_state.selected_plots:
                st.error("❌ Please select at least one plot")
            else:
                with st.spinner("Generating plots..."):
                    st.session_state.analyzer.generate_all_plots(st.session_state.selected_plots)
                    st.session_state.plots_generated = True
                    st.success(f"✅ Successfully generated {len(st.session_state.analyzer.all_figures)} plots!")
                    
                    # Auto-advance to visualization stage
                    st.session_state.current_stage = 3
                    st.rerun()
    
    with col2:
        if st.button("🎯 Generate All Plots", use_container_width=True):
            st.session_state.selected_plots = [plot[0] for plot in all_plots]
            with st.spinner("Generating all plots..."):
                st.session_state.analyzer.generate_all_plots()
                st.session_state.plots_generated = True
                st.success(f"✅ Successfully generated {len(st.session_state.analyzer.all_figures)} plots!")
                st.session_state.current_stage = 3
                st.rerun()

def stage3_visualization():
    """Stage 3: View generated plots"""
    st.header("👁️ Stage 3: Visualization")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back to Settings", use_container_width=True):
            st.session_state.current_stage = 2
            st.rerun()
    with col2:
        if st.button("🏠 Back to Start", use_container_width=True):
            st.session_state.current_stage = 1
            st.rerun()
    with col3:
        if st.button("➡️ Proceed to Export", type="primary", use_container_width=True):
            st.session_state.current_stage = 4
            st.rerun()
    
    st.markdown("---")
    
    if not st.session_state.plots_generated or not st.session_state.analyzer.all_figures:
        st.warning("⚠️ No plots generated yet. Please go back to Stage 2 and generate plots.")
        if st.button("⬅️ Back to Plot Selection"):
            st.session_state.current_stage = 2
            st.rerun()
        return
    
    # Plot navigation
    plot_names = list(st.session_state.analyzer.all_figures.keys())
    
    # All plots list for reference
    all_plots = [
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
        ("18_author_network", "19. Author Collaboration Network"),
        ("18_18_1_affiliation_network", "20.1 Affiliation Network (Top 20)"),
        ("18_18_2_affiliation_network", "20.2 Affiliation Network (Top 30)"),
        ("18_18_3_affiliation_network", "20.3 Affiliation Network (Top 50)"),
        ("19_hierarchical_sankey", "21. Hierarchical Sankey Diagram"),
        ("20_mds", "22. Multidimensional Scaling"),
        ("21_concept_network_weighted", "23. Weighted Concept Network")
    ]
    
    # Create mapping for display names
    plot_display_names = {pid: name for pid, name in all_plots}
    
    # Plot selector
    selected_plot_id = st.selectbox(
        "📊 Select Plot to View",
        options=plot_names,
        format_func=lambda x: plot_display_names.get(x, x)
    )
    
    # Display plot
    if selected_plot_id in st.session_state.analyzer.all_figures:
        fig = st.session_state.analyzer.all_figures[selected_plot_id]
        
        # Check plot type
        if hasattr(fig, 'update_layout'):
            # Plotly figure
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Matplotlib figure
            st.pyplot(fig)
        
        # Plot info
        st.info(f"**{plot_display_names.get(selected_plot_id, selected_plot_id)}**")
        
        # Navigation between plots
        current_index = plot_names.index(selected_plot_id)
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if current_index > 0:
                if st.button("◀️ Previous Plot", use_container_width=True):
                    st.session_state.current_plot_index = current_index - 1
                    st.rerun()
        
        with col2:
            st.markdown(f"<div style='text-align: center'>Plot {current_index + 1} of {len(plot_names)}</div>", 
                       unsafe_allow_html=True)
        
        with col3:
            if current_index < len(plot_names) - 1:
                if st.button("Next Plot ▶️", use_container_width=True):
                    st.session_state.current_plot_index = current_index + 1
                    st.rerun()
    
    # Show statistics
    st.markdown("---")
    st.subheader("📊 Generation Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Generated Plots", len(st.session_state.analyzer.all_figures))
    with col2:
        st.metric("Errors", len(st.session_state.analyzer.errors))
    with col3:
        st.metric("Warnings", len(st.session_state.analyzer.warnings))
    
    if st.session_state.analyzer.errors:
        with st.expander("❌ Errors"):
            for error in st.session_state.analyzer.errors:
                st.error(f"{error['timestamp']}: {error['message']}")
    
    if st.session_state.analyzer.warnings:
        with st.expander("⚠️ Warnings"):
            for warning in st.session_state.analyzer.warnings:
                st.warning(f"{warning['timestamp']}: {warning['message']}")

def stage4_export():
    """Stage 4: Export results"""
    st.header("📥 Stage 4: Export Results")
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back to Visualization", use_container_width=True):
            st.session_state.current_stage = 3
            st.rerun()
    with col2:
        if st.button("🏠 Back to Start", use_container_width=True):
            st.session_state.current_stage = 1
            st.rerun()
    
    st.markdown("---")
    
    if not st.session_state.plots_generated or not st.session_state.analyzer.all_figures:
        st.warning("⚠️ No plots generated yet. Please go back and generate plots first.")
        if st.button("⬅️ Back to Plot Selection"):
            st.session_state.current_stage = 2
            st.rerun()
        return
    
    st.success(f"✅ Ready to export: {len(st.session_state.analyzer.all_figures)} plots available")
    
    # Export options
    st.subheader("📦 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Individual plot download
        st.markdown("### 📸 Individual Plots")
        
        all_plots = [
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
            ("18_author_network", "19. Author Collaboration Network"),
            ("18_18_1_affiliation_network", "20.1 Affiliation Network (Top 20)"),
            ("18_18_2_affiliation_network", "20.2 Affiliation Network (Top 30)"),
            ("18_18_3_affiliation_network", "20.3 Affiliation Network (Top 50)"),
            ("19_hierarchical_sankey", "21. Hierarchical Sankey Diagram"),
            ("20_mds", "22. Multidimensional Scaling"),
            ("21_concept_network_weighted", "23. Weighted Concept Network")
        ]
        
        plot_display_names = {pid: name for pid, name in all_plots}
        
        available_plots = [(pid, plot_display_names.get(pid, pid)) 
                          for pid in st.session_state.analyzer.all_figures.keys()]
        
        if available_plots:
            selected_plot_name = st.selectbox(
                "Select plot to download",
                options=[pid for pid, _ in available_plots],
                format_func=lambda x: plot_display_names.get(x, x)
            )
            
            if selected_plot_name:
                fig = st.session_state.analyzer.all_figures[selected_plot_name]
                
                # Save to buffer based on plot type
                img_buffer = io.BytesIO()
                
                if hasattr(fig, 'write_image'):
                    # Plotly figure
                    import plotly.io as pio
                    pio.write_image(fig, img_buffer, format='png', width=1200, height=900, scale=2)
                else:
                    # Matplotlib figure
                    fig.savefig(img_buffer, format='png', dpi=600, bbox_inches='tight')
                
                img_buffer.seek(0)
                
                st.download_button(
                    label=f"📥 Download {plot_display_names.get(selected_plot_name, selected_plot_name)}",
                    data=img_buffer,
                    file_name=f"{selected_plot_name}.png",
                    mime="image/png",
                    use_container_width=True
                )
    
    with col2:
        # Bulk export
        st.markdown("### 📦 Bulk Export")
        
        include_excel = st.checkbox("Include Excel Data Report", value=True)
        
        if st.button("📥 Download Complete ZIP Archive", type="primary", use_container_width=True):
            with st.spinner("Creating ZIP archive..."):
                zip_buffer = st.session_state.analyzer.save_all_to_zip(include_excel=include_excel)
                
                if zip_buffer:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"scientific_analysis_{timestamp}.zip"
                    
                    st.download_button(
                        label="⬇️ Download ZIP",
                        data=zip_buffer,
                        file_name=filename,
                        mime="application/zip",
                        use_container_width=True
                    )
    
    # Statistics
    st.markdown("---")
    st.subheader("📊 Export Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Plots", len(st.session_state.analyzer.all_figures))
    with col2:
        st.metric("Data Points", len(st.session_state.analyzer.plot_data) if st.session_state.analyzer.plot_data else 0)
    with col3:
        st.metric("Errors", len(st.session_state.analyzer.errors))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>Scientific Data Visualization Dashboard | Powered by Streamlit, Matplotlib, Plotly, NetworkX</p>
        <p>All plots generated at 600 DPI with scientific styling (12×9 inches)</p>
    </div>
    """, unsafe_allow_html=True)

# Run application
if __name__ == "__main__":
    main()
