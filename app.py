# -*- coding: utf-8 -*-
"""Scientific Data Visualization Dashboard - Streamlit Version (Modular 4-Stage Interface)"""

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

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Scientific Data Visualization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR MODERN DESIGN
# ============================================================================

st.markdown("""
<style>
    /* Modern color scheme */
    :root {
        --primary: #2E86AB;
        --primary-dark: #1a5a7a;
        --secondary: #F18F01;
        --success: #6B8E23;
        --danger: #C73E1D;
        --info: #4ECDC4;
        --dark: #2C3E50;
        --light: #ECF0F1;
        --white: #FFFFFF;
    }
    
    /* Main container styling */
    .main {
        background-color: #F8F9FA;
    }
    
    /* Card styling */
    .css-1r6slb0, .stCard {
        background-color: var(--white);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border: 1px solid #E9ECEF;
        margin-bottom: 20px;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: var(--dark);
        font-weight: 600;
    }
    
    h1 {
        border-bottom: 3px solid var(--primary);
        padding-bottom: 10px;
        display: inline-block;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background-color: var(--primary);
        color: white;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: var(--primary-dark);
    }
    
    /* Metric styling */
    .stMetric {
        background-color: var(--white);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #E9ECEF;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: var(--primary);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: var(--light);
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--white);
        border-right: 1px solid #E9ECEF;
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 8px;
        border-left-width: 4px;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Navigation buttons container */
    .nav-buttons {
        display: flex;
        gap: 10px;
        margin: 20px 0;
        padding: 15px;
        background-color: var(--light);
        border-radius: 12px;
    }
    
    /* Status indicator */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
    }
    
    .status-success {
        background-color: #d4edda;
        color: #155724;
    }
    
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SCIENTIFIC PLOTTING STYLE (9×12 inches, 600 DPI)
# ============================================================================

def set_scientific_style():
    """Set scientific plotting style with 9×12 inches and 600 DPI"""
    plt.style.use('default')
    plt.rcParams.update({
        # Figure dimensions
        'figure.figsize': (12, 9),
        'figure.dpi': 600,
        'savefig.dpi': 600,
        
        # Font settings
        'font.size': 10,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino'],
        
        # Axes
        'axes.labelsize': 11,
        'axes.labelweight': 'bold',
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'axes.grid': False,
        
        # Ticks
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
        
        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'errorbar.capsize': 3,
        
        # Save
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.facecolor': 'white',
        'figure.facecolor': 'white',
    })

set_scientific_style()

# ============================================================================
# COLOR PALETTE OPTIONS (10+ variants per category)
# ============================================================================

HEATMAP_PALETTES = {
    'Blues': 'Blues',
    'Reds': 'Reds',
    'Greens': 'Greens',
    'RdYlGn': 'RdYlGn',
    'Viridis': 'viridis',
    'Plasma': 'plasma',
    'Inferno': 'inferno',
    'Magma': 'magma',
    'Cividis': 'cividis',
    'Coolwarm': 'coolwarm',
    'Spectral': 'Spectral',
    'PiYG': 'PiYG',
    'BrBG': 'BrBG',
    'PRGn': 'PRGn',
    'PuOr': 'PuOr',
    'RdBu': 'RdBu',
}

BAR_PALETTES = {
    'Blues': 'Blues',
    'Reds': 'Reds',
    'Greens': 'Greens',
    'Oranges': 'Oranges',
    'Purples': 'Purples',
    'Viridis': 'viridis',
    'Plasma': 'plasma',
    'Set1': 'Set1',
    'Set2': 'Set2',
    'Set3': 'Set3',
    'Paired': 'Paired',
    'Accent': 'Accent',
    'Dark2': 'Dark2',
    'Tab10': 'tab10',
}

SCATTER_PALETTES = {
    'Viridis': 'viridis',
    'Plasma': 'plasma',
    'Inferno': 'inferno',
    'Magma': 'magma',
    'Cividis': 'cividis',
    'Coolwarm': 'coolwarm',
    'Spectral': 'Spectral',
    'Rainbow': 'rainbow',
    'Jet': 'jet',
    'Blues': 'Blues',
    'Reds': 'Reds',
}

NETWORK_PALETTES = {
    'Blues': 'Blues',
    'Reds': 'Reds',
    'Greens': 'Greens',
    'Purples': 'Purples',
    'Oranges': 'Oranges',
    'Viridis': 'viridis',
    'Plasma': 'plasma',
    'RdYlGn': 'RdYlGn',
}

BOX_PALETTES = {
    'Set1': 'Set1',
    'Set2': 'Set2',
    'Set3': 'Set3',
    'Pastel1': 'Pastel1',
    'Pastel2': 'Pastel2',
    'Paired': 'Paired',
    'Accent': 'Accent',
    'Dark2': 'Dark2',
    'Tab10': 'tab10',
}

# ============================================================================
# SCIENTIFIC DATA ANALYZER CLASS
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
        
        # Color palette settings
        self.heatmap_palette = 'Blues'
        self.bar_palette = 'Blues'
        self.scatter_palette = 'viridis'
        self.network_palette = 'Blues'
        self.box_palette = 'Set2'
        
        # Regression line toggle
        self.show_regression = True
        
    def log_error(self, error_msg, details=""):
        """Log an error"""
        self.errors.append({
            'timestamp': datetime.now().isoformat(),
            'message': error_msg,
            'details': details
        })
        st.error(f"❌ ERROR: {error_msg}")
        if details:
            st.error(f"   Details: {details}")
    
    def log_warning(self, warning_msg):
        """Log a warning"""
        self.warnings.append({
            'timestamp': datetime.now().isoformat(),
            'message': warning_msg
        })
        st.warning(f"⚠️ WARNING: {warning_msg}")
    
    def update_progress(self, value):
        """Update progress"""
        self.progress = value
    
    def parse_data(self, data_text):
        """Parse data from text input with extended diagnostics"""
        st.info("🔍 Parsing data...")
        
        try:
            lines = data_text.strip().split('\n')
            if len(lines) < 2:
                self.log_error("Not enough data rows", f"Found {len(lines)} lines")
                return None
            
            headers = lines[0].split('\t')
            st.info(f"   Found {len(headers)} columns")
            st.info(f"   Headers: {headers}")
            
            required_columns = ['doi', 'Title', 'year', 'count']
            missing_columns = [col for col in required_columns if col not in headers]
            if missing_columns:
                self.log_warning(f"Missing columns: {missing_columns}")
            
            data = []
            for i, line in enumerate(lines[1:]):
                if line.strip():
                    values = line.split('\t')
                    while len(values) < len(headers):
                        values.append('')
                    data.append(values)
            
            self.df = pd.DataFrame(data, columns=headers)
            st.success(f"✅ Successfully parsed {len(self.df)} rows")
            
            self._diagnose_data()
            self.df_processed = self._preprocess_data(self.df)
            
            return self.df_processed
            
        except Exception as e:
            self.log_error(f"Error parsing data: {str(e)}", traceback.format_exc())
            return None
    
    def _diagnose_data(self):
        """Diagnose data quality"""
        st.info("🔬 Data Diagnostics:")
        st.write("---")
        
        missing_counts = self.df.isnull().sum()
        total_cells = np.prod(self.df.shape)
        missing_percent = (missing_counts.sum() / total_cells) * 100
        
        st.info(f"   Total cells: {total_cells:,}")
        st.info(f"   Missing values: {missing_counts.sum():,} ({missing_percent:.1f}%)")
        
        top_missing = missing_counts[missing_counts > 0].sort_values(ascending=False).head(5)
        if len(top_missing) > 0:
            st.info("   Top columns with missing values:")
            for col, count in top_missing.items():
                percent = (count / len(self.df)) * 100
                st.info(f"     - {col}: {count:,} ({percent:.1f}%)")
        
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
        
        numeric_cols = ['author count', 'year', 'Citation counts (CR)', 'Citation counts (OA)',
                       'Annual cit counts (CR)', 'Annual cit counts (OA)', 'references_count', 'count']
        
        for col in numeric_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        if 'publication_date' in df_processed.columns:
            df_processed['publication_date'] = pd.to_datetime(df_processed['publication_date'], errors='coerce')
            if 'year' not in df_processed.columns or df_processed['year'].isnull().all():
                df_processed['year'] = df_processed['publication_date'].dt.year
        
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
        
        current_year = datetime.now().year
        if 'year' in df_processed.columns:
            df_processed['article_age'] = current_year - df_processed['year']
            df_processed['article_age'] = df_processed['article_age'].clip(lower=1)
            
            if 'count' in df_processed.columns:
                df_processed['normalized_attention'] = df_processed['count'] / df_processed['article_age']
        
        if 'Citation counts (CR)' in df_processed.columns and 'Citation counts (OA)' in df_processed.columns:
            df_processed['max_citations'] = df_processed[['Citation counts (CR)', 'Citation counts (OA)']].max(axis=1)
            df_processed['max_annual_citations'] = df_processed[['Annual cit counts (CR)', 'Annual cit counts (OA)']].max(axis=1)
        
        if 'countries_list' in df_processed.columns:
            df_processed['num_countries'] = df_processed['countries_list'].apply(len)
        
        if 'affiliations_list' in df_processed.columns:
            df_processed['num_affiliations'] = df_processed['affiliations_list'].apply(len)
        
        # Process authors for collaboration network
        if 'authors_list' in df_processed.columns:
            df_processed['authors_processed'] = df_processed['authors_list'].apply(
                lambda authors: [self._format_author_name(a) for a in authors if a.strip()]
            )
        
        st.success("✅ Data preprocessing complete")
        return df_processed
    
    def _format_author_name(self, full_name):
        """Format author name: keep last name + first initial only"""
        parts = full_name.strip().split()
        if len(parts) >= 2:
            last_name = parts[-1]
            first_initial = parts[0][0] if parts[0] else ''
            return f"{first_initial}. {last_name}"
        return full_name.strip()
    
    def _get_cmap(self, palette_name, plot_type='heatmap'):
        """Get colormap by name and type"""
        if plot_type == 'heatmap':
            cmap_name = HEATMAP_PALETTES.get(palette_name, 'Blues')
        elif plot_type == 'bar':
            cmap_name = BAR_PALETTES.get(palette_name, 'Blues')
        elif plot_type == 'scatter':
            cmap_name = SCATTER_PALETTES.get(palette_name, 'viridis')
        elif plot_type == 'network':
            cmap_name = NETWORK_PALETTES.get(palette_name, 'Blues')
        else:
            cmap_name = palette_name
        
        if isinstance(cmap_name, str) and cmap_name in plt.colormaps():
            return plt.cm.get_cmap(cmap_name)
        return plt.cm.Blues
    
    def _get_bar_colors(self, n_colors, palette_name='Blues'):
        """Get colors for bar charts"""
        cmap = self._get_cmap(palette_name, 'bar')
        return [cmap(i / max(1, n_colors - 1)) for i in range(n_colors)]
    
    # ============================================================================
    # PLOTTING FUNCTIONS (23 types)
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
            
            fig, axes = plt.subplots(1, 3, figsize=(12, 9))
            fig.suptitle('Distribution of Research Attention', fontweight='bold', fontsize=12)
            
            axes[0].hist(counts, bins=np.logspace(np.log10(1), np.log10(max(100, counts.max())), 30),
                        edgecolor='black', alpha=0.7, color=self._get_bar_colors(1, self.bar_palette)[0])
            axes[0].set_xscale('log')
            axes[0].set_yscale('log')
            axes[0].set_xlabel('Number of Mentions', fontweight='bold')
            axes[0].set_ylabel('Frequency', fontweight='bold')
            axes[0].set_title('A. Log-Log Distribution', fontweight='bold')
            
            self.plot_data['1_distribution'] = {
                'counts': counts.tolist(),
                'log_bins': np.logspace(np.log10(1), np.log10(max(100, counts.max())), 30).tolist()
            }
            
            sorted_counts = np.sort(counts)
            ccdf = 1 - np.arange(len(sorted_counts)) / len(sorted_counts)
            
            axes[1].loglog(sorted_counts, ccdf, 'o-', markersize=2, linewidth=1.5, color='#C73E1D')
            axes[1].set_xlabel('Number of Mentions', fontweight='bold')
            axes[1].set_ylabel('CCDF (P(X ≥ x))', fontweight='bold')
            axes[1].set_title('B. Complementary CDF', fontweight='bold')
            
            sorted_counts = np.sort(counts)
            cumulative_counts = np.cumsum(sorted_counts)
            cumulative_percent = cumulative_counts / cumulative_counts[-1]
            population_percent = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
            
            axes[2].plot(population_percent, cumulative_percent, linewidth=2.5,
                        color='#6B8E23', label='Lorenz curve')
            axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5, label='Perfect equality')
            axes[2].fill_between(population_percent, 0, cumulative_percent, alpha=0.2, color='#6B8E23')
            
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
            
            G = nx.Graph()
            
            for idx, row in self.df_processed.iterrows():
                if isinstance(row['countries_list'], list) and len(row['countries_list']) >= 2:
                    countries = [c.strip().upper() for c in row['countries_list']]
                    weight = row.get('count', 1)
                    
                    for country in countries:
                        if not G.has_node(country):
                            G.add_node(country, weight=0, papers=0)
                        G.nodes[country]['weight'] += weight
                        G.nodes[country]['papers'] += 1
                    
                    for i in range(len(countries)):
                        for j in range(i+1, len(countries)):
                            if G.has_edge(countries[i], countries[j]):
                                G[countries[i]][countries[j]]['weight'] += weight
                                G[countries[i]][countries[j]]['papers'] += 1
                            else:
                                G.add_edge(countries[i], countries[j], weight=weight, papers=1)
            
            if len(G.nodes()) < 3:
                self.log_warning("Insufficient data for country network")
                return None
            
            self.plot_data['2_country_network'] = {
                'nodes': [{'country': node, 'weight': G.nodes[node]['weight'], 
                          'papers': G.nodes[node]['papers']} for node in G.nodes()],
                'edges': [{'country1': u, 'country2': v, 'weight': G[u][v]['weight'],
                          'papers': G[u][v]['papers']} for u, v in G.edges()]
            }
            
            fig, ax = plt.subplots(figsize=(12, 9))
            
            min_weight = np.percentile([d['weight'] for u, v, d in G.edges(data=True)], 50) if G.edges() else 0
            edges_to_keep = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] >= min_weight]
            H = G.edge_subgraph([u for u, v in edges_to_keep] + [v for u, v in edges_to_keep]) if edges_to_keep else G
            
            pos = nx.spring_layout(H, k=2, seed=42)
            node_sizes = [H.nodes[n]['weight'] * 0.5 + 500 for n in H.nodes()]
            node_colors = [H.nodes[n]['papers'] for n in H.nodes()]
            
            cmap = self._get_cmap(self.network_palette, 'network')
            nodes = nx.draw_networkx_nodes(H, pos, node_size=node_sizes,
                                          node_color=node_colors, cmap=cmap,
                                          alpha=0.8, edgecolors='black', linewidths=1.5, ax=ax)
            
            if H.edges():
                edge_weights = [H[u][v]['weight'] * 0.1 for u, v in H.edges()]
                edges = nx.draw_networkx_edges(H, pos, width=edge_weights, alpha=0.5,
                                              edge_color='gray', style='solid', ax=ax)
            
            nx.draw_networkx_labels(H, pos, font_size=10, font_weight='bold', ax=ax)
            
            ax.set_title('Country Collaboration Network', fontweight='bold', fontsize=12, pad=20)
            ax.axis('off')
            
            sm = plt.cm.ScalarMappable(cmap=cmap,
                                      norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Number of Papers', fontweight='bold', fontsize=10)
            
            stats_text = f"Nodes: {len(G.nodes())} | Edges: {len(G.edges())}"
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
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
            
            fig, ax = plt.subplots(figsize=(12, 9))
            
            cmap = self._get_cmap(self.scatter_palette, 'scatter')
            scatter = ax.scatter(valid_data['num_countries'],
                               valid_data['Citation counts (CR)'],
                               c=valid_data.get('Annual cit counts (CR)', 1),
                               s=valid_data['author count'] * 20,
                               alpha=0.7,
                               cmap=cmap,
                               edgecolors='black',
                               linewidths=0.5)
            
            if self.show_regression and len(valid_data) > 10:
                x = valid_data['num_countries'].values
                y = valid_data['Citation counts (CR)'].values
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() > 10:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                    y_line = intercept + slope * x_line
                    ax.plot(x_line, y_line, 'r--', linewidth=2, 
                           label=f'r = {r_value:.3f}, p = {p_value:.3f}')
                    ax.legend(loc='upper left', fontsize=8)
            
            ax.set_xlabel('Number of Collaborating Countries', fontweight='bold')
            ax.set_ylabel('Total Citations (CR)', fontweight='bold')
            ax.set_title('International Collaboration vs Citation Impact', fontweight='bold', fontsize=12)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Annual Citation Rate (CR)', fontweight='bold')
            
            sizes = [5, 10, 15]
            labels = ['5 authors', '10 authors', '15 authors']
            legend_elements = []
            for size, label in zip(sizes, labels):
                legend_elements.append(plt.scatter([], [], s=size*20, c='gray',
                                                  alpha=0.7, edgecolors='black',
                                                  label=label))
            ax.legend(handles=legend_elements, loc='upper left', title='Team Size')
            
            self.plot_data['3_internationality_vs_citations'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_3_internationality_vs_citations: {str(e)}")
            return None
    
    def plot_4_journal_year_heatmap(self, top_journals=15):
        """4. Journal-Year heatmap"""
        try:
            required_cols = ['Full journal Name', 'year', 'Annual cit counts (CR)']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
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
            
            row_sums = pivot_table.sum(axis=1)
            pivot_table = pivot_table[row_sums > 0]
            
            if len(pivot_table) < 2:
                self.log_warning("Insufficient years with data for heatmap")
                return None
            
            self.plot_data['4_journal_year_heatmap'] = {
                'pivot_data': pivot_table.to_dict(),
                'years': pivot_table.index.tolist(),
                'journals': pivot_table.columns.tolist()
            }
            
            fig, ax = plt.subplots(figsize=(12, 9))
            
            cmap = self._get_cmap(self.heatmap_palette, 'heatmap')
            im = ax.imshow(pivot_table.values, cmap=cmap, aspect='auto', vmin=0)
            
            ax.set_xticks(np.arange(len(pivot_table.columns)))
            ax.set_yticks(np.arange(len(pivot_table.index)))
            ax.set_xticklabels(pivot_table.columns, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(pivot_table.index.astype(int), fontsize=10)
            
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
                        fontweight='bold', fontsize=12)
            
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Average Annual Citations (CR)', rotation=90, fontsize=12)
            
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
            
            fig, axes = plt.subplots(1, 3, figsize=(12, 9))
            fig.suptitle('Collaboration Scale vs Citation Impact (Linear Scale)', 
                        fontweight='bold', fontsize=12)
            
            metrics = [
                ('author count', 'Number of Authors', axes[0]),
                ('num_affiliations', 'Number of Affiliations', axes[1]),
                ('num_countries', 'Number of Countries', axes[2])
            ]
            
            for idx, (metric, label, ax) in enumerate(metrics):
                cmap = self._get_cmap(self.scatter_palette, 'scatter')
                scatter = ax.scatter(valid_data[metric],
                                   valid_data['max_citations'],
                                   c=valid_data['num_countries'] if metric != 'num_countries' else valid_data['author count'],
                                   s=valid_data['author count'] * 10,
                                   alpha=0.6,
                                   cmap=cmap,
                                   edgecolors='black',
                                   linewidths=0.5)
                
                if self.show_regression and len(valid_data) > 10:
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
                ax.grid(True, alpha=0.3)
                if self.show_regression:
                    ax.legend(loc='upper left', fontsize=8)
                
                if idx < 2:
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar_label = 'Number of Countries' if metric != 'num_countries' else 'Number of Authors'
                    cbar.set_label(cbar_label, fontweight='bold')
            
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
            
            fig, axes = plt.subplots(1, 3, figsize=(12, 9))
            fig.suptitle('Collaboration Scale vs Citation Impact (Log Y Scale)', 
                        fontweight='bold', fontsize=12)
            
            metrics = [
                ('author count', 'Number of Authors', axes[0]),
                ('num_affiliations', 'Number of Affiliations', axes[1]),
                ('num_countries', 'Number of Countries', axes[2])
            ]
            
            for idx, (metric, label, ax) in enumerate(metrics):
                plot_data = valid_data[valid_data['max_citations'] > 0].copy()
                if len(plot_data) < 10:
                    continue
                
                cmap = self._get_cmap(self.scatter_palette, 'scatter')
                scatter = ax.scatter(plot_data[metric],
                                   plot_data['max_citations'],
                                   c=plot_data['num_countries'] if metric != 'num_countries' else plot_data['author count'],
                                   s=plot_data['author count'] * 10,
                                   alpha=0.6,
                                   cmap=cmap,
                                   edgecolors='black',
                                   linewidths=0.5)
                
                if self.show_regression and len(plot_data) > 10:
                    x = plot_data[metric].values
                    log_y = np.log(plot_data['max_citations'].values)
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
                ax.set_yscale('log')
                
                if self.show_regression:
                    ax.legend(loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.3, which='both')
                
                cbar = plt.colorbar(scatter, ax=ax)
                if metric != 'num_countries':
                    cbar.set_label('Number of Countries', fontweight='bold', fontsize=10)
                else:
                    cbar.set_label('Number of Authors', fontweight='bold', fontsize=10)
                
                from matplotlib.lines import Line2D
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
            
            self.plot_data['6_collaboration_vs_citations_log'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_6_collaboration_vs_citations_log: {str(e)}")
            return None
    
    def plot_6_1_bubble_chart(self):
        """6.1 Bubble chart: References vs Citations (linear)"""
        try:
            required_cols = ['references_count', 'Citation counts (CR)', 'author count', 'num_countries']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) < 10:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 9))
            
            cmap = self._get_cmap(self.scatter_palette, 'scatter')
            scatter = ax.scatter(valid_data['references_count'],
                               valid_data['Citation counts (CR)'],
                               s=valid_data['author count'] * 40,
                               c=valid_data['num_countries'],
                               cmap=cmap,
                               alpha=0.7,
                               edgecolors='black',
                               linewidths=0.5)
            
            if self.show_regression and len(valid_data) > 10:
                x = valid_data['references_count'].values
                y = valid_data['Citation counts (CR)'].values
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() > 10:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                    y_line = intercept + slope * x_line
                    ax.plot(x_line, y_line, 'r--', linewidth=2, 
                           label=f'r = {r_value:.3f}, p = {p_value:.3f}')
                    ax.legend(loc='upper left')
            
            ax.set_xlabel('Number of References', fontweight='bold')
            ax.set_ylabel('Total Citations (CR) - Linear Scale', fontweight='bold')
            ax.set_title('Research Breadth vs Impact (Linear Scale)', fontweight='bold', fontsize=12)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Number of Collaborating Countries', fontweight='bold')
            
            sizes = [5, 10, 15]
            labels = ['5 authors', '10 authors', '15 authors']
            legend_elements = [plt.scatter([], [], s=size*40, c='gray', alpha=0.7,
                                          edgecolors='black', label=label)
                              for size, label in zip(sizes, labels)]
            ax.legend(handles=legend_elements, title='Team Size', loc='upper left')
            ax.grid(True, alpha=0.3)
            
            self.plot_data['6_1_bubble_chart'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_6_1_bubble_chart: {str(e)}")
            return None
    
    def plot_6_2_bubble_chart(self):
        """6.2 Bubble chart: References vs Citations (logarithmic)"""
        try:
            required_cols = ['references_count', 'Citation counts (CR)', 'author count', 'num_countries']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) < 10:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 9))
            
            cmap = self._get_cmap(self.scatter_palette, 'scatter')
            scatter = ax.scatter(valid_data['references_count'],
                               valid_data['Citation counts (CR)'],
                               s=valid_data['author count'] * 40,
                               c=valid_data['num_countries'],
                               cmap=cmap,
                               alpha=0.7,
                               edgecolors='black',
                               linewidths=0.5)
            
            ax.set_xlabel('Number of References', fontweight='bold')
            ax.set_ylabel('Total Citations (CR) - Log Scale', fontweight='bold')
            ax.set_title('Research Breadth vs Impact (Logarithmic Scale)', fontweight='bold', fontsize=12)
            ax.set_yscale('log')
            
            if self.show_regression and len(valid_data) > 10:
                valid_log_data = valid_data[valid_data['Citation counts (CR)'] > 0].copy()
                if len(valid_log_data) > 10:
                    x = valid_log_data['references_count'].values
                    log_y = np.log(valid_log_data['Citation counts (CR)'].values)
                    mask = np.isfinite(log_y)
                    if mask.sum() > 10:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], log_y[mask])
                        x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                        y_line = np.exp(intercept + slope * x_line)
                        ax.plot(x_line, y_line, 'r--', linewidth=2, 
                               label=f'r = {r_value:.3f}, p = {p_value:.3f}')
                        ax.legend(loc='upper left')
            
            min_citation = valid_data['Citation counts (CR)'].min()
            if min_citation <= 0:
                valid_log_data = valid_data[valid_data['Citation counts (CR)'] > 0].copy()
                if len(valid_log_data) > 0:
                    scatter = ax.scatter(valid_log_data['references_count'],
                                       valid_log_data['Citation counts (CR)'],
                                       s=valid_log_data['author count'] * 40,
                                       c=valid_log_data['num_countries'],
                                       cmap=cmap,
                                       alpha=0.7,
                                       edgecolors='black',
                                       linewidths=0.5)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Number of Collaborating Countries', fontweight='bold')
            
            sizes = [5, 10, 15]
            labels = ['5 authors', '10 authors', '15 authors']
            legend_elements = [plt.scatter([], [], s=size*40, c='gray', alpha=0.7,
                                          edgecolors='black', label=label)
                              for size, label in zip(sizes, labels)]
            ax.legend(handles=legend_elements, title='Team Size', loc='upper left')
            ax.grid(True, alpha=0.3, which='both')
            
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
            
            all_concepts = []
            for concepts in self.df_processed['concepts_list']:
                if isinstance(concepts, list):
                    all_concepts.extend([c.strip() for c in concepts])
            
            if len(all_concepts) == 0:
                return None
            
            concept_counts = pd.Series(all_concepts).value_counts()
            top_concepts = concept_counts.head(top_n)
            
            self.plot_data['7_concepts_analysis'] = {
                'top_concepts': top_concepts.to_dict(),
                'total_concepts': len(concept_counts)
            }
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9))
            
            y_pos = np.arange(len(top_concepts))
            colors = self._get_bar_colors(len(top_concepts), self.bar_palette)
            
            bars = ax1.barh(y_pos, top_concepts.values, color=colors, edgecolor='black')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(top_concepts.index, fontsize=9)
            ax1.set_xlabel('Frequency', fontweight='bold')
            ax1.set_title(f'Top {top_n} Research Concepts', fontweight='bold')
            ax1.invert_yaxis()
            
            for bar in bars:
                width = bar.get_width()
                ax1.text(width * 1.01, bar.get_y() + bar.get_height()/2,
                        f'{int(width)}', va='center', fontsize=8)
            
            fig_height = fig.get_size_inches()[1]
            wordcloud_height = fig_height * 0.8
            wordcloud_width = wordcloud_height * 1.6
            
            wordcloud = WordCloud(width=int(wordcloud_width*100), 
                                height=int(wordcloud_height*100), 
                                background_color='white',
                                colormap=self.scatter_palette, max_words=100).generate_from_frequencies(concept_counts.to_dict())
            
            ax2.imshow(wordcloud, interpolation='bilinear')
            ax2.axis('off')
            ax2.set_title('Concept Word Cloud', fontweight='bold')
            
            ax1.set_ylim(-0.5, len(top_concepts) - 0.5)
            
            plt.suptitle('Research Concepts Analysis', fontweight='bold', fontsize=12)
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
            
            all_concepts = []
            for concepts in self.df_processed['concepts_list']:
                if isinstance(concepts, list):
                    all_concepts.extend([c.strip() for c in concepts])
            
            if len(all_concepts) == 0:
                return None
            
            concept_counts = pd.Series(all_concepts).value_counts()
            top_concepts = concept_counts.head(top_n).index.tolist()
            
            cooccurrence = pd.DataFrame(0, index=top_concepts, columns=top_concepts)
            
            for concepts in self.df_processed['concepts_list']:
                if isinstance(concepts, list):
                    concepts_clean = [c.strip() for c in concepts if c.strip() in top_concepts]
                    for i in range(len(concepts_clean)):
                        for j in range(i+1, len(concepts_clean)):
                            c1, c2 = concepts_clean[i], concepts_clean[j]
                            cooccurrence.loc[c1, c2] += 1
                            cooccurrence.loc[c2, c1] += 1
            
            self.plot_data['8_concept_cooccurrence'] = {
                'matrix': cooccurrence.to_dict(),
                'concepts': top_concepts
            }
            
            fig, ax = plt.subplots(figsize=(12, 9))
            
            cmap = self._get_cmap(self.heatmap_palette, 'heatmap')
            im = ax.imshow(cooccurrence.values, cmap=cmap)
            
            ax.set_xticks(np.arange(len(top_concepts)))
            ax.set_yticks(np.arange(len(top_concepts)))
            ax.set_xticklabels(top_concepts, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(top_concepts, fontsize=9)
            
            for i in range(len(top_concepts)):
                for j in range(len(top_concepts)):
                    value = cooccurrence.iloc[i, j]
                    if value > 0:
                        ax.text(j, i, str(value),
                               ha="center", va="center",
                               color="white" if value > cooccurrence.values.max()/2 else "black",
                               fontsize=8, fontweight='bold')
            
            ax.set_title(f'Concept Co-occurrence Matrix (Top {top_n} Concepts)',
                        fontweight='bold', fontsize=12)
            
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Co-occurrence Frequency', rotation=90, fontsize=12)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_8_concept_cooccurrence: {str(e)}")
            return None
    
    def plot_9_concept_influence(self):
        """9. Concept influence analysis"""
        try:
            if 'concepts_list' not in self.df_processed.columns or 'max_citations' not in self.df_processed.columns:
                return None
            
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
            
            concept_stats = concept_df.groupby('concept').agg({
                'max_citations': ['sum', 'mean', 'median'],
                'max_annual_citations': 'mean',
                'count': 'size'
            }).round(2)
            
            concept_stats.columns = ['total_citations', 'mean_citations', 'median_citations',
                                   'mean_annual_citations', 'num_papers']
            
            concept_stats = concept_stats[concept_stats['num_papers'] >= 2]
            concept_stats = concept_stats.sort_values('mean_citations', ascending=False).head(20)
            
            self.plot_data['9_concept_influence'] = concept_stats.reset_index().to_dict('records')
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 9))
            
            y_pos = np.arange(len(concept_stats))
            colors = self._get_bar_colors(len(concept_stats), self.bar_palette)
            
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
            
            cmap = self._get_cmap(self.scatter_palette, 'scatter')
            scatter = axes[1].scatter(concept_stats['mean_annual_citations'],
                                    concept_stats['mean_citations'],
                                    s=concept_stats['num_papers'] * 15,
                                    c=concept_stats['total_citations'],
                                    cmap=cmap,
                                    alpha=0.7,
                                    edgecolors='black',
                                    linewidths=0.5)
            
            axes[1].set_xlabel('Mean Annual Citations', fontweight='bold')
            axes[1].set_ylabel('Mean Total Citations', fontweight='bold')
            axes[1].set_title('Concept Impact: Annual vs Total Citations', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=axes[1])
            cbar.set_label('Total Citations (Sum)', fontweight='bold')
            
            for idx, row in concept_stats.head(5).iterrows():
                short_name = idx[:20] + '...' if len(idx) > 20 else idx
                axes[1].annotate(short_name,
                               xy=(row['mean_annual_citations'], row['mean_citations']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
            
            plt.suptitle('Concept Influence Analysis', fontweight='bold', fontsize=12)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_9_concept_influence: {str(e)}")
            return None
    
    def plot_10_temporal_evolution(self):
        """10. Temporal evolution of publications and citations"""
        try:
            if 'year' not in self.df_processed.columns or 'max_citations' not in self.df_processed.columns:
                return None
            
            valid_data = self.df_processed.dropna(subset=['year', 'max_citations'])
            if len(valid_data) < 10:
                return None
            
            year_stats = valid_data.groupby('year').agg({
                'max_citations': ['sum', 'mean'],
                'max_annual_citations': 'mean',
                'doi': 'count'
            }).round(2)
            
            year_stats.columns = ['total_citations', 'mean_citations', 'mean_annual_citations', 'num_papers']
            year_stats = year_stats.sort_index()
            
            self.plot_data['10_temporal_evolution'] = year_stats.reset_index().to_dict('records')
            
            fig, ax1 = plt.subplots(figsize=(12, 9))
            
            ax1.bar(year_stats.index, year_stats['num_papers'], 
                   alpha=0.4, color='steelblue', label='Number of Papers', edgecolor='black')
            ax1.set_xlabel('Publication Year', fontweight='bold')
            ax1.set_ylabel('Number of Papers', fontweight='bold', color='steelblue')
            ax1.tick_params(axis='y', labelcolor='steelblue')
            
            ax2 = ax1.twinx()
            line1 = ax2.plot(year_stats.index, year_stats['total_citations'], 
                           'o-', color='darkorange', linewidth=2.5, markersize=6,
                           label='Total Citations')
            ax2.set_ylabel('Total Citations', fontweight='bold', color='darkorange')
            ax2.tick_params(axis='y', labelcolor='darkorange')
            
            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('outward', 60))
            line2 = ax3.plot(year_stats.index, year_stats['mean_citations'], 
                           's-', color='darkgreen', linewidth=2, markersize=5,
                           label='Mean Citations per Paper')
            ax3.set_ylabel('Mean Citations per Paper', fontweight='bold', color='darkgreen')
            ax3.tick_params(axis='y', labelcolor='darkgreen')
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
            
            ax1.set_title('Temporal Evolution: Publications and Citation Impact',
                        fontweight='bold', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_10_temporal_evolution: {str(e)}")
            return None
    
    def plot_11_temporal_heatmap(self):
        """11. Temporal heatmap: Publication year vs Article age"""
        try:
            if 'year' not in self.df_processed.columns or 'max_annual_citations' not in self.df_processed.columns:
                return None
            
            valid_data = self.df_processed.dropna(subset=['year', 'max_annual_citations'])
            if len(valid_data) < 10:
                return None
            
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
            
            pivot_table = heatmap_df.pivot_table(
                values='annual_citations',
                index='age',
                columns='pub_year',
                aggfunc='mean',
                fill_value=0
            ).sort_index(ascending=False)
            
            self.plot_data['11_temporal_heatmap'] = {
                'pivot_data': pivot_table.to_dict(),
                'years': pivot_table.columns.tolist(),
                'ages': pivot_table.index.tolist()
            }
            
            fig, ax = plt.subplots(figsize=(12, 9))
            
            cmap = self._get_cmap(self.heatmap_palette, 'heatmap')
            im = ax.imshow(pivot_table.values, cmap=cmap, aspect='auto', vmin=0)
            
            ax.set_xticks(np.arange(len(pivot_table.columns)))
            ax.set_yticks(np.arange(len(pivot_table.index)))
            ax.set_xticklabels(pivot_table.columns.astype(int), rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(pivot_table.index, fontsize=10)
            ax.set_xlabel('Publication Year', fontweight='bold')
            ax.set_ylabel('Article Age (Years)', fontweight='bold')
            ax.set_title('Annual Citation Rate by Publication Year and Article Age',
                        fontweight='bold', fontsize=12)
            
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Mean Annual Citations (max(CR, OA))', rotation=90, fontsize=12)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_11_temporal_heatmap: {str(e)}")
            return None
    
    def plot_11_team_size_analysis(self):
        """12. Team size analysis"""
        try:
            if 'author count' not in self.df_processed.columns:
                return None
            
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
            
            group_stats = self.df_processed.groupby('team_size_group').agg({
                'count': ['mean', 'median', 'std', 'size'],
                'Citation counts (CR)': 'mean',
                'references_count': 'mean'
            }).round(2)
            
            group_stats.columns = ['mean_attention', 'median_attention', 'std_attention',
                                 'num_papers', 'mean_citations', 'mean_references']
            
            custom_order = ['Single author', '2 authors', '3 authors', '4-5 authors', 
                          '6-8 authors', '9-12 authors', '13+ authors', 'Unknown']
            
            existing_categories = [cat for cat in custom_order if cat in group_stats.index]
            group_stats = group_stats.loc[existing_categories]
            
            self.plot_data['11_team_size_analysis'] = group_stats.reset_index().to_dict('records')
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            axes = axes.flatten()
            
            team_size_counts = self.df_processed['team_size_group'].value_counts()
            team_size_counts = team_size_counts.reindex(existing_categories, fill_value=0)
            
            colors = self._get_bar_colors(len(team_size_counts), self.bar_palette)
            axes[0].bar(team_size_counts.index, team_size_counts.values,
                       alpha=0.7, color=colors, edgecolor='black')
            axes[0].set_xlabel('Team Size', fontweight='bold')
            axes[0].set_ylabel('Number of Papers', fontweight='bold')
            axes[0].set_title('Distribution of Team Sizes', fontweight='bold')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3, axis='y')
            
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
            
            axes[2].bar(group_stats.index, group_stats['mean_citations'],
                       alpha=0.7, color=colors, edgecolor='black')
            axes[2].set_xlabel('Team Size', fontweight='bold')
            axes[2].set_ylabel('Mean Citations (CR)', fontweight='bold')
            axes[2].set_title('Mean Citations by Team Size', fontweight='bold')
            axes[2].tick_params(axis='x', rotation=45)
            axes[2].grid(True, alpha=0.3, axis='y')
            
            axes[3].bar(group_stats.index, group_stats['mean_references'],
                       alpha=0.7, color=colors, edgecolor='black')
            axes[3].set_xlabel('Team Size', fontweight='bold')
            axes[3].set_ylabel('Mean References', fontweight='bold')
            axes[3].set_title('Mean References by Team Size', fontweight='bold')
            axes[3].tick_params(axis='x', rotation=45)
            axes[3].grid(True, alpha=0.3, axis='y')
            
            plt.suptitle('Team Size Analysis', fontweight='bold', fontsize=12)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_11_team_size_analysis: {str(e)}")
            return None
    
    def plot_12_correlation_matrix(self):
        """13. Correlation matrix with key parameters highlighted"""
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
            
            key_params = ['count', 'max_citations', 'max_annual_citations',
                         'Annual cit counts (CR)', 'Annual cit counts (OA)',
                         'Citation counts (CR)', 'Citation counts (OA)']
            
            existing_key_params = [p for p in key_params if p in corr_matrix.columns]
            other_params = [p for p in corr_matrix.columns if p not in existing_key_params]
            new_order = existing_key_params + other_params
            corr_matrix = corr_matrix.reindex(index=new_order, columns=new_order)
            
            self.plot_data['12_correlation_matrix'] = {
                'correlation_matrix': corr_matrix.to_dict(),
                'columns': available_cols,
                'method': 'spearman',
                'key_parameters': existing_key_params
            }
            
            fig, ax = plt.subplots(figsize=(12, 9))
            
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            cmap = self._get_cmap(self.heatmap_palette, 'heatmap')
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                       cmap=cmap, center=0, square=True,
                       linewidths=0.5, cbar_kws={'shrink': 0.8},
                       ax=ax, annot_kws={'fontsize': 9})
            
            key_param_indices = [i for i, col in enumerate(corr_matrix.columns) if col in existing_key_params]
            for idx in key_param_indices:
                ax.add_patch(plt.Rectangle((0, idx), len(corr_matrix), 1, 
                                          fill=False, edgecolor='red', linewidth=2, 
                                          alpha=0.7))
                ax.add_patch(plt.Rectangle((idx, 0), 1, len(corr_matrix), 
                                          fill=False, edgecolor='red', linewidth=2, 
                                          alpha=0.7))
            
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='white', edgecolor='red', linewidth=2,
                                   alpha=0.7, label='Key parameters (Count & Citations)')]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            ax.set_title('Correlation Matrix of Research Metrics (Spearman)\nKey Parameters Highlighted in Red', 
                        fontweight='bold', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_12_correlation_matrix: {str(e)}")
            return None
    
    def plot_13_cr_vs_oa_comparison(self):
        """14. CR vs OA citations comparison"""
        try:
            required_cols = ['Citation counts (CR)', 'Citation counts (OA)']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) == 0:
                return None
            
            valid_data['citation_diff'] = valid_data['Citation counts (OA)'] - valid_data['Citation counts (CR)']
            valid_data['citation_ratio'] = valid_data['Citation counts (OA)'] / valid_data['Citation counts (CR)'].replace(0, 1)
            
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
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9))
            
            max_val = max(valid_data['Citation counts (CR)'].max(),
                         valid_data['Citation counts (OA)'].max())
            
            ax1.scatter(valid_data['Citation counts (CR)'],
                       valid_data['Citation counts (OA)'],
                       alpha=0.6, c='steelblue', edgecolors='black', linewidths=0.5)
            ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y = x')
            
            ax1.set_xlabel('Citations from Crossref (CR)', fontweight='bold')
            ax1.set_ylabel('Citations from OpenAlex (OA)', fontweight='bold')
            ax1.set_title('Comparison of Citation Counts', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            colors = self._get_bar_colors(30, self.bar_palette)
            ax2.hist(valid_data['citation_diff'], bins=30,
                    alpha=0.7, color=colors[0], edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('Difference (OA - CR)', fontweight='bold')
            ax2.set_ylabel('Number of Articles', fontweight='bold')
            ax2.set_title('Distribution of Citation Differences', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            stats_text = f"Mean difference: {valid_data['citation_diff'].mean():.1f}\n"
            stats_text += f"OA > CR: {(valid_data['citation_diff'] > 0).sum()} articles\n"
            stats_text += f"CR > OA: {(valid_data['citation_diff'] < 0).sum()} articles"
            
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.suptitle('Comparison of Citation Sources', fontweight='bold', fontsize=12)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_13_cr_vs_oa_comparison: {str(e)}")
            return None
    
    def plot_14_citation_by_domain(self):
        """15. Citations by research domain"""
        try:
            required_cols = ['Domain', 'Annual cit counts (CR)']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) == 0:
                return None
            
            domain_stats = valid_data.groupby('Domain').agg({
                'Annual cit counts (CR)': ['median', 'mean', 'std', 'count'],
                'count': 'mean'
            }).round(2)
            
            domain_stats.columns = ['median_citations', 'mean_citations', 'std_citations',
                                  'num_papers', 'mean_attention']
            domain_stats = domain_stats.sort_values('median_citations', ascending=False)
            
            self.plot_data['14_citation_by_domain'] = domain_stats.reset_index().to_dict('records')
            
            top_domains = domain_stats.head(15).index.tolist()
            filtered_data = valid_data[valid_data['Domain'].isin(top_domains)]
            
            fig, ax = plt.subplots(figsize=(12, 9))
            
            box_data = []
            labels = []
            for domain in top_domains:
                data = filtered_data[filtered_data['Domain'] == domain]['Annual cit counts (CR)'].values
                if len(data) > 0:
                    box_data.append(data)
                    labels.append(domain)
            
            colors = self._get_cmap(self.box_palette, 'bar')
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True, showfliers=False)
            
            for patch, color in zip(bp['boxes'], [colors(i/len(box_data)) for i in range(len(box_data))]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
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
        """16. Cumulative influence curve"""
        try:
            if 'count' not in self.df_processed.columns:
                return None
            
            valid_data = self.df_processed.dropna(subset=['count'])
            if len(valid_data) == 0:
                return None
            
            sorted_counts = valid_data['count'].sort_values(ascending=False).reset_index(drop=True)
            
            total_citations = sorted_counts.sum()
            cumulative_citations = sorted_counts.cumsum()
            cumulative_percentage = cumulative_citations / total_citations * 100
            article_percentage = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100
            
            self.plot_data['15_cumulative_influence'] = {
                'sorted_counts': sorted_counts.tolist(),
                'cumulative_percentage': cumulative_percentage.tolist(),
                'article_percentage': article_percentage.tolist(),
                'total_citations': float(total_citations),
                'gini_coefficient': self._calculate_gini(sorted_counts.values)
            }
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9))
            
            ax1.plot(article_percentage, cumulative_percentage,
                    linewidth=2.5, color='darkgreen')
            ax1.fill_between(article_percentage, 0, cumulative_percentage,
                            alpha=0.3, color='lightgreen')
            
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
            
            log_bins = np.logspace(0, np.log10(sorted_counts.max() + 1), 20)
            colors = self._get_bar_colors(20, self.bar_palette)
            ax2.hist(sorted_counts, bins=log_bins, alpha=0.7,
                    color=colors[0], edgecolor='black')
            ax2.set_xscale('log')
            ax2.set_xlabel('Number of Local Citations (log scale)', fontweight='bold')
            ax2.set_ylabel('Number of Articles', fontweight='bold')
            ax2.set_title('Distribution of Local Citation Counts', fontweight='bold')
            ax2.grid(True, alpha=0.3, which='both')
            
            gini = self._calculate_gini(sorted_counts.values)
            stats_text = f"Total articles: {len(sorted_counts):,}\n"
            stats_text += f"Total mentions: {total_citations:,}\n"
            stats_text += f"Mean: {sorted_counts.mean():.2f}\n"
            stats_text += f"Median: {sorted_counts.median():.1f}\n"
            stats_text += f"Gini coefficient: {gini:.3f}"
            
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.suptitle('Analysis of Local Influence Within Dataset', fontweight='bold', fontsize=12)
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
        """17. References vs impact"""
        try:
            required_cols = ['references_count', 'count', 'Annual cit counts (CR)']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) < 10:
                return None
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9))
            
            cmap = self._get_cmap(self.scatter_palette, 'scatter')
            scatter1 = ax1.scatter(valid_data['references_count'],
                                 valid_data['count'],
                                 c=valid_data['Annual cit counts (CR)'],
                                 cmap=cmap, alpha=0.6, s=30,
                                 edgecolors='black', linewidths=0.5)
            
            if self.show_regression and len(valid_data) > 10:
                x = valid_data['references_count'].values
                y = valid_data['count'].values
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() > 10:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                    y_line = intercept + slope * x_line
                    ax1.plot(x_line, y_line, 'r--', linewidth=2,
                            label=f'r = {r_value:.3f}, p = {p_value:.3f}')
                    ax1.legend()
            
            ax1.set_xlabel('Number of References', fontweight='bold')
            ax1.set_ylabel('Local Mentions (count)', fontweight='bold')
            ax1.set_title('References vs Local Attention', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            cbar1 = plt.colorbar(scatter1, ax=ax1)
            cbar1.set_label('Annual Citations (CR)', fontweight='bold')
            
            scatter2 = ax2.scatter(valid_data['references_count'],
                                 valid_data['Annual cit counts (CR)'],
                                 c=valid_data['count'],
                                 cmap=cmap, alpha=0.6, s=30,
                                 edgecolors='black', linewidths=0.5)
            
            if self.show_regression and len(valid_data) > 10:
                x = valid_data['references_count'].values
                y = valid_data['Annual cit counts (CR)'].values
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() > 10:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                    y_line = intercept + slope * x_line
                    ax2.plot(x_line, y_line, 'r--', linewidth=2,
                            label=f'r = {r_value:.3f}, p = {p_value:.3f}')
                    ax2.legend()
            
            ax2.set_xlabel('Number of References', fontweight='bold')
            ax2.set_ylabel('Annual Citations (CR)', fontweight='bold')
            ax2.set_title('References vs Citation Impact', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            cbar2 = plt.colorbar(scatter2, ax=ax2)
            cbar2.set_label('Local Mentions', fontweight='bold')
            
            plt.suptitle('Impact of Reference Count on Research Metrics', fontweight='bold', fontsize=12)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_16_references_vs_impact: {str(e)}")
            return None
    
    def plot_17_journal_impact(self):
        """18. Journal impact analysis"""
        try:
            required_cols = ['Full journal Name', 'count', 'Annual cit counts (CR)']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) == 0:
                return None
            
            journal_stats = valid_data.groupby('Full journal Name').agg({
                'count': ['mean', 'median', 'std', 'size'],
                'Annual cit counts (CR)': 'mean',
                'references_count': 'mean'
            }).round(2)
            
            journal_stats.columns = ['mean_attention', 'median_attention', 'std_attention',
                                   'num_papers', 'mean_citations', 'mean_references']
            
            journal_stats = journal_stats[journal_stats['num_papers'] >= 3]
            journal_stats = journal_stats.sort_values('mean_attention', ascending=False)
            
            self.plot_data['17_journal_impact'] = journal_stats.reset_index().to_dict('records')
            
            top_journals = journal_stats.head(15)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9))
            
            y_pos = np.arange(len(top_journals))
            colors = self._get_bar_colors(len(top_journals), self.bar_palette)
            
            bars1 = ax1.barh(y_pos, top_journals['mean_attention'],
                            color=colors, edgecolor='black', alpha=0.8)
            
            ax1.set_yticks(y_pos)
            journal_names = [name[:25] + '...' if len(name) > 25 else name
                            for name in top_journals.index]
            ax1.set_yticklabels(journal_names, fontsize=9)
            ax1.set_xlabel('Mean Attention per Paper', fontweight='bold')
            ax1.set_title('Top Journals by Attention', fontweight='bold')
            ax1.invert_yaxis()
            
            for bar, (_, row) in zip(bars1, top_journals.iterrows()):
                width = bar.get_width()
                info_text = f"n={int(row['num_papers'])}"
                ax1.text(width * 1.01, bar.get_y() + bar.get_height()/2,
                        info_text, va='center', fontsize=8)
            
            cmap = self._get_cmap(self.scatter_palette, 'scatter')
            scatter = ax2.scatter(top_journals['mean_citations'],
                                top_journals['mean_attention'],
                                s=top_journals['num_papers'] * 10,
                                c=top_journals['mean_references'],
                                cmap=cmap, alpha=0.7,
                                edgecolors='black', linewidths=0.5)
            
            ax2.set_xlabel('Mean Annual Citations (CR)', fontweight='bold')
            ax2.set_ylabel('Mean Attention', fontweight='bold')
            ax2.set_title('Journal Impact: Citations vs Attention', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Mean References', fontweight='bold')
            
            for idx, row in top_journals.head(5).iterrows():
                short_name = idx[:15] + '...' if len(idx) > 15 else idx
                ax2.annotate(short_name,
                            xy=(row['mean_citations'], row['mean_attention']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, alpha=0.8)
            
            plt.suptitle('Journal Impact Analysis', fontweight='bold', fontsize=12)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_17_journal_impact: {str(e)}")
            return None
    
    def plot_18_18_1_affiliation_network(self):
        """19.1 Affiliation network (Top 20)"""
        try:
            return self._plot_affiliation_network_impl(20, "1")
        except Exception as e:
            self.log_error(f"Error in plot_18_18_1_affiliation_network: {str(e)}")
            return None
    
    def plot_18_18_2_affiliation_network(self):
        """19.2 Affiliation network (Top 30)"""
        try:
            return self._plot_affiliation_network_impl(30, "2")
        except Exception as e:
            self.log_error(f"Error in plot_18_18_2_affiliation_network: {str(e)}")
            return None
    
    def plot_18_18_3_affiliation_network(self):
        """19.3 Affiliation network (Top 50)"""
        try:
            return self._plot_affiliation_network_impl(50, "3")
        except Exception as e:
            self.log_error(f"Error in plot_18_18_3_affiliation_network: {str(e)}")
            return None
    
    def _plot_affiliation_network_impl(self, top_n, suffix):
        """Affiliation network implementation"""
        try:
            if 'affiliations_list' not in self.df_processed.columns:
                return None
            
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
            
            degree_dict = dict(G.degree(weight='weight'))
            top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_node_names = [node[0] for node in top_nodes]
            
            H = G.subgraph(top_node_names)
            
            self.plot_data[f'18_18_{suffix}_affiliation_network'] = {
                'nodes': [{'affiliation': node, 'weight': H.nodes[node]['weight'],
                          'papers': H.nodes[node]['papers']} for node in H.nodes()],
                'edges': [{'aff1': u, 'aff2': v, 'weight': H[u][v]['weight'],
                          'papers': H[u][v]['papers']} for u, v in H.edges()]
            }
            
            fig, ax = plt.subplots(figsize=(12, 9))
            
            pos = nx.spring_layout(H, k=3, seed=42)
            
            node_sizes = [H.nodes[n]['weight'] * 0.3 + 300 for n in H.nodes()]
            node_colors = [H.nodes[n]['papers'] for n in H.nodes()]
            
            cmap = self._get_cmap(self.network_palette, 'network')
            nodes = nx.draw_networkx_nodes(H, pos, node_size=node_sizes,
                                          node_color=node_colors, cmap=cmap,
                                          alpha=0.8, edgecolors='black', linewidths=1.5, ax=ax)
            
            if H.edges():
                edge_weights = [H[u][v]['weight'] * 0.05 for u, v in H.edges()]
                edges = nx.draw_networkx_edges(H, pos, width=edge_weights, alpha=0.4,
                                              edge_color='gray', style='solid', ax=ax)
            
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
                        fontweight='bold', fontsize=12)
            ax.axis('off')
            
            sm = plt.cm.ScalarMappable(cmap=cmap,
                                      norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Number of Papers', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in _plot_affiliation_network_impl: {str(e)}")
            return None
    
    def plot_19_hierarchical_sankey(self):
        """20. Hierarchical Sankey diagram: Domain → Field → Subfield → Topic"""
        try:
            required_cols = ['Domain', 'Field', 'Subfield', 'Topic', 'max_citations']
            available_cols = [col for col in required_cols if col in self.df_processed.columns]
            
            if len(available_cols) < 3:
                return None
            
            valid_data = self.df_processed.dropna(subset=available_cols)
            if len(valid_data) < 10:
                return None
            
            links = []
            nodes = []
            node_indices = {}
            
            def add_node(name):
                if name not in node_indices:
                    node_indices[name] = len(nodes)
                    nodes.append(name)
                return node_indices[name]
            
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
                
                domain_idx = add_node(domain)
                field_idx = add_node(field)
                subfield_idx = add_node(subfield)
                topic_idx = add_node(topic)
                
                links.append({'source': domain_idx, 'target': field_idx, 'value': weight})
                links.append({'source': field_idx, 'target': subfield_idx, 'value': weight})
                links.append({'source': subfield_idx, 'target': topic_idx, 'value': weight})
            
            self.plot_data['19_hierarchical_sankey'] = {
                'nodes': nodes,
                'links': links,
                'total_weight': sum([l['value'] for l in links])
            }
            
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
        """21. Multidimensional scaling of predictors"""
        try:
            predictors = ['author count', 'references_count', 'num_countries',
                         'Annual cit counts (CR)', 'article_age', 'normalized_attention']
            
            available_predictors = [p for p in predictors if p in self.df_processed.columns]
            
            if len(available_predictors) < 3:
                return None
            
            analysis_data = self.df_processed[available_predictors + ['count']].dropna()
            
            if len(analysis_data) < 20:
                return None
            
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(analysis_data[available_predictors])
            
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            
            self.plot_data['20_mds_analysis'] = {
                'explained_variance': pca.explained_variance_ratio_.tolist(),
                'components': pca.components_.tolist(),
                'pca_coordinates': pca_result.tolist(),
                'predictors': available_predictors
            }
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9))
            
            cmap = self._get_cmap(self.scatter_palette, 'scatter')
            scatter = ax1.scatter(pca_result[:, 0], pca_result[:, 1],
                                 c=analysis_data['count'], cmap=cmap,
                                 alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
            
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontweight='bold')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontweight='bold')
            ax1.set_title('PCA: Multidimensional Scaling of Predictors', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Local Mentions (count)', fontweight='bold')
            
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            
            for i, predictor in enumerate(available_predictors):
                ax2.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                         color='red', alpha=0.5, head_width=0.05)
                ax2.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15,
                        predictor, color='red', fontsize=10, fontweight='bold')
            
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
            
            plt.suptitle('Multidimensional Analysis of Research Predictors', fontweight='bold', fontsize=12)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_20_multidimensional_scaling: {str(e)}")
            return None
    
    def plot_21_concept_network_weighted(self):
        """22. Weighted concept network"""
        try:
            if 'concepts_list' not in self.df_processed.columns or 'max_citations' not in self.df_processed.columns:
                return None
            
            all_concepts = []
            for concepts in self.df_processed['concepts_list']:
                if isinstance(concepts, list):
                    all_concepts.extend([c.strip() for c in concepts])
            
            if len(all_concepts) == 0:
                return None
            
            concept_counts = pd.Series(all_concepts).value_counts()
            top_concepts = concept_counts.head(25).index.tolist()
            
            G = nx.Graph()
            
            for concept in top_concepts:
                concept_papers = []
                for idx, row in self.df_processed.iterrows():
                    if isinstance(row['concepts_list'], list) and concept in row['concepts_list']:
                        concept_papers.append(row.get('max_citations', 0))
                
                total_citations = sum(concept_papers)
                G.add_node(concept, citations=total_citations, papers=len(concept_papers))
            
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
            
            self.plot_data['21_concept_network_weighted'] = {
                'nodes': [{'concept': node, 'citations': G.nodes[node]['citations'],
                          'papers': G.nodes[node]['papers']} for node in G.nodes()],
                'edges': [{'concept1': u, 'concept2': v, 'weight': G[u][v]['weight']} 
                         for u, v in G.edges()]
            }
            
            fig, ax = plt.subplots(figsize=(12, 9))
            
            pos = nx.spring_layout(G, k=2, seed=42)
            
            node_sizes = [G.nodes[n]['citations'] * 0.2 + 500 for n in G.nodes()]
            node_colors = [G.nodes[n]['papers'] for n in G.nodes()]
            
            cmap = self._get_cmap(self.network_palette, 'network')
            nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                          node_color=node_colors, cmap=cmap,
                                          alpha=0.8, edgecolors='black', linewidths=1.5, ax=ax)
            
            if G.edges():
                edge_weights = [G[u][v]['weight'] * 0.01 for u, v in G.edges()]
                edges = nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5,
                                              edge_color='gray', style='solid', ax=ax)
            
            nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
            
            ax.set_title('Concept Network with Citation Impact', fontweight='bold', fontsize=12)
            ax.axis('off')
            
            sm = plt.cm.ScalarMappable(cmap=cmap,
                                      norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Number of Papers', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_21_concept_network_weighted: {str(e)}")
            return None
    
    def plot_23_author_collaboration_network(self, top_n=30):
        """23. Author collaboration network"""
        try:
            if 'authors_processed' not in self.df_processed.columns:
                return None
            
            G = nx.Graph()
            
            for idx, row in self.df_processed.iterrows():
                authors = row['authors_processed']
                if isinstance(authors, list) and len(authors) >= 2:
                    weight = row.get('count', 1)
                    
                    for author in authors:
                        if not G.has_node(author):
                            G.add_node(author, weight=0, papers=0)
                        G.nodes[author]['weight'] += weight
                        G.nodes[author]['papers'] += 1
                    
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
            
            degree_dict = dict(G.degree(weight='weight'))
            top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_node_names = [node[0] for node in top_nodes]
            
            H = G.subgraph(top_node_names)
            
            self.plot_data['23_author_network'] = {
                'nodes': [{'author': node, 'weight': H.nodes[node]['weight'],
                          'papers': H.nodes[node]['papers']} for node in H.nodes()],
                'edges': [{'author1': u, 'author2': v, 'weight': H[u][v]['weight'],
                          'papers': H[u][v]['papers']} for u, v in H.edges()]
            }
            
            fig, ax = plt.subplots(figsize=(12, 9))
            
            pos = nx.spring_layout(H, k=3, seed=42)
            
            node_sizes = [H.nodes[n]['weight'] * 0.5 + 500 for n in H.nodes()]
            node_colors = [H.nodes[n]['papers'] for n in H.nodes()]
            
            cmap = self._get_cmap(self.network_palette, 'network')
            nodes = nx.draw_networkx_nodes(H, pos, node_size=node_sizes,
                                          node_color=node_colors, cmap=cmap,
                                          alpha=0.8, edgecolors='black', linewidths=1.5, ax=ax)
            
            if H.edges():
                edge_weights = [H[u][v]['weight'] * 0.1 for u, v in H.edges()]
                edges = nx.draw_networkx_edges(H, pos, width=edge_weights, alpha=0.5,
                                              edge_color='gray', style='solid', ax=ax)
            
            nx.draw_networkx_labels(H, pos, font_size=9, font_weight='bold', ax=ax)
            
            ax.set_title(f'Author Collaboration Network (Top {top_n} Authors by Connection Strength)',
                        fontweight='bold', fontsize=12)
            ax.axis('off')
            
            sm = plt.cm.ScalarMappable(cmap=cmap,
                                      norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Number of Papers', fontweight='bold', fontsize=10)
            
            stats_text = f"Nodes: {len(H.nodes())} | Edges: {len(H.edges())}"
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_23_author_collaboration_network: {str(e)}")
            return None
    
    def generate_all_plots(self, selected_plots=None):
        """Generate all plots with progress bar"""
        self.all_figures = {}
        self.plot_data = {}
        self.errors = []
        self.warnings = []
        
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
            ("21_concept_network_weighted", "22. Weighted Concept Network", self.plot_21_concept_network_weighted),
            ("23_author_network", "23. Author Collaboration Network", lambda: self.plot_23_author_collaboration_network(30)),
        ]
        
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
        
        excel_buffer = io.BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            if self.df_processed is not None:
                basic_stats = pd.DataFrame({
                    'Metric': [
                        'Total papers', 'Year range', 'Total mentions',
                        'Mean mentions per paper', 'Median mentions',
                        'Unique countries', 'Unique journals',
                        'Max citations (max(CR, OA))', 'Max annual citations (max(annual CR, annual OA))'
                    ],
                    'Value': [
                        len(self.df_processed),
                        f"{int(self.df_processed['year'].min())}-{int(self.df_processed['year'].max())}" if 'year' in self.df_processed.columns else 'N/A',
                        int(self.df_processed['count'].sum()) if 'count' in self.df_processed.columns else 'N/A',
                        f"{self.df_processed['count'].mean():.2f}" if 'count' in self.df_processed.columns else 'N/A',
                        f"{self.df_processed['count'].median():.1f}" if 'count' in self.df_processed.columns else 'N/A',
                        self.df_processed['num_countries'].nunique() if 'num_countries' in self.df_processed.columns else 'N/A',
                        self.df_processed['Full journal Name'].nunique() if 'Full journal Name' in self.df_processed.columns else 'N/A',
                        f"{self.df_processed['max_citations'].mean():.1f}" if 'max_citations' in self.df_processed.columns else 'N/A',
                        f"{self.df_processed['max_annual_citations'].mean():.1f}" if 'max_annual_citations' in self.df_processed.columns else 'N/A'
                    ]
                })
                basic_stats.to_excel(writer, sheet_name='Basic_Statistics', index=False)
            
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
            
            if self.errors:
                errors_df = pd.DataFrame(self.errors)
                errors_df.to_excel(writer, sheet_name='Errors', index=False)
            
            if self.warnings:
                warnings_df = pd.DataFrame(self.warnings)
                warnings_df.to_excel(writer, sheet_name='Warnings', index=False)
            
            self._add_terminology_sheet(writer)
        
        excel_buffer.seek(0)
        return excel_buffer
    
    def _add_terminology_sheet(self, writer):
        """Add terminology explanation sheet"""
        terminology_data = {
            'Term': [
                'PC1 (Principal Component 1)', 'PC2 (Principal Component 2)',
                'normalized_attention', 'attention (count)', 'total mentions',
                'article_age', 'max_citations', 'max_annual_citations',
                'Annual cit counts (CR)', 'Annual cit counts (OA)',
                'num_countries', 'num_affiliations', 'Gini coefficient',
                'CCDF', 'Lorenz curve', 'Spearman correlation', 'PCA'
            ],
            'Description': [
                'First principal component explaining maximum variance',
                'Second principal component orthogonal to PC1',
                'Attention normalized by article age: normalized_attention = count / article_age',
                'Local mentions count within the dataset (column "count")',
                'Sum of all attention (count) values across the dataset',
                'Age of article in years: article_age = current_year - publication_year',
                'Maximum citations between CR and OA',
                'Maximum annual citations between CR and OA',
                'Annual citation rate from Crossref',
                'Annual citation rate from OpenAlex',
                'Number of collaborating countries',
                'Number of affiliations',
                'Measure of inequality (0 = perfect equality, 1 = maximum inequality)',
                'Probability that a variable X is greater than or equal to x',
                'Graphical representation of distribution inequality',
                'Non-parametric rank correlation coefficient',
                'Dimensionality reduction technique'
            ],
            'Formula/Calculation': [
                'PC1 = w₁₁*x₁ + w₁₂*x₂ + ...', 'PC2 = w₂₁*x₁ + w₂₂*x₂ + ...',
                'normalized_attention = count / max(1, current_year - year)',
                'Directly from "count" column', 'total_mentions = Σ count_i',
                'article_age = 2024 - year', 'max(CR_citations, OA_citations)',
                'max(CR_annual, OA_annual)', 'Citation counts (CR) / article_age',
                'Citation counts (OA) / article_age', 'len(split(countries, ";"))',
                'len(split(affiliations, ";"))', 'G = (A / (A+B))',
                'CCDF(x) = 1 - (observations ≤ x) / (total observations)',
                'Plot of (cumulative % of papers, cumulative % of mentions)',
                'ρ = 1 - (6Σd²)/(n(n²-1))', 'PCA = eigendecomposition of covariance matrix'
            ],
            'Data Source': [
                'Calculated from standardized predictors', 'Calculated from standardized predictors',
                'Calculated from original data', 'Original data column "count"',
                'Calculated: sum of all "count"', 'Calculated: current year minus publication year',
                'Calculated: max of CR and OA', 'Calculated: max of annual rates',
                'Calculated from CR citations', 'Calculated from OA citations',
                'Calculated from countries column', 'Calculated from affiliations column',
                'Calculated from sorted attention values', 'Calculated from sorted attention values',
                'Calculated from sorted attention values', 'Calculated using pandas corr(method="spearman")',
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
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, (name, fig) in enumerate(self.all_figures.items()):
                img_buffer = io.BytesIO()
                
                if hasattr(fig, 'update_layout'):
                    import plotly.io as pio
                    pio.write_image(fig, img_buffer, format='png', width=1200, height=900, scale=2)
                else:
                    fig.savefig(img_buffer, format='png', dpi=600,
                              bbox_inches='tight', facecolor='white',
                              edgecolor='black')
                
                img_buffer.seek(0)
                filename = f"plot_{i+1:02d}_{name}.png"
                zip_file.writestr(filename, img_buffer.read())
                plt.close(fig) if not hasattr(fig, 'update_layout') else None
            
            if include_excel:
                excel_buffer = self.create_excel_report()
                if excel_buffer:
                    zip_file.writestr("plot_data.xlsx", excel_buffer.read())
            
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
# STREAMLIT 4-STAGE MODULAR INTERFACE
# ============================================================================

def main():
    """Main Streamlit application with 4-stage modular interface"""
    
    st.title("📊 Scientific Data Visualization Dashboard")
    st.markdown("---")
    
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
        ("21_concept_network_weighted", "22. Weighted Concept Network"),
        ("23_author_network", "23. Author Collaboration Network")
    ]
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ScientificDataAnalyzer()
    
    if 'stage' not in st.session_state:
        st.session_state.stage = 1
    
    if 'plots_generated' not in st.session_state:
        st.session_state.plots_generated = False
    
    if 'selected_plots' not in st.session_state:
        st.session_state.selected_plots = [plot[0] for plot in ALL_PLOTS]
    
    # Navigation buttons container
    st.markdown('<div class="nav-buttons">', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
    
    with col1:
        if st.button("🏠 Start", use_container_width=True):
            st.session_state.stage = 1
            st.rerun()
    
    with col2:
        if st.button("⬅️ Back", use_container_width=True) and st.session_state.stage > 1:
            st.session_state.stage -= 1
            st.rerun()
    
    with col3:
        st.markdown(f"**Stage {st.session_state.stage}/4**")
    
    with col4:
        if st.button("Next ➡️", use_container_width=True) and st.session_state.stage < 4:
            if st.session_state.stage == 1 and st.session_state.analyzer.df_processed is None:
                st.error("Please load data first!")
            elif st.session_state.stage == 2 and not st.session_state.selected_plots:
                st.error("Please select at least one plot!")
            else:
                st.session_state.stage += 1
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # STAGE 1: DATA UPLOAD
    # ========================================================================
    
    if st.session_state.stage == 1:
        st.header("📋 Stage 1: Data Upload")
        
        data_input = st.text_area(
            "Paste your data in TSV format (tab-separated values)",
            value=st.session_state.get('sample_data_loaded', ''),
            height=300,
            help="Copy and paste data from Excel/Google Sheets. First row must contain column headers."
        )
        
        sample_data = """doi	publication_date	Title	authors	ORCID ID 1; ORCID ID 2... ORCID ID last	author count	affiliations {aff 1; aff 2... aff last}	countries {country 1; ... country last}	Full journal Name	year	Volume	Pages (or article number)	Citation counts (CR)	Citation counts (OA)	Annual cit counts (CR)	Annual cit counts (OA)	references_count	count	Topic	Subfield	Field	Domain	Concepts
10.1021/acs.chemrev.6b00284	2016-11-09	Strategies for Carbon and Sulfur Tolerant Solid Oxide Fuel Cell Materials, Incorporating Lessons from Heterogeneous Catalysis	Paul Boldrin; Enrique Ruiz-Trejo; Joshua Mermelstein; José Miguel Bermúdez Menéndez; Tomás Ramı́rez Reina; Nigel P. Brandon	https://orcid.org/0000-0003-0058-6876; https://orcid.org/0000-0001-5560-5750; https://orcid.org/0000-0001-7211-2958; https://orcid.org/0000-0001-9693-5107; https://orcid.org/0000-0003-2230-8666	6	University of Surrey; Imperial College London; Boeing (United States)	US; GB	Chemical Reviews	2016	116	13633-13684	289	296	26.27	26.91	465	5	Advancements in Solid Oxide Fuel Cells	Chemistry	Carbon fibers	Catalysis	Sulfur; Chemistry; Carbon fibers; Catalysis; Oxide; Solid oxide fuel cell; Fuel cells; Nanotechnology; Environmental chemistry; Chemical engineering; Organic chemistry; Materials science; Engineering; Composite number; Physical chemistry; Composite material; Anode; Electrode
10.1126/science.aab3987	2015-07-23	Readily processed protonic ceramic fuel cells with high performance at low temperatures	Chuancheng Duan; Jianhua Tong; Meng Shang; Stefan Nikodemski; Michael Sanders; Sandrine Ricote; Ali Almansoori; Ryan O'Hayre	https://orcid.org/0000-0002-1826-1415; https://orcid.org/0000-0002-0684-1658; https://orcid.org/0000-0001-6366-5219; https://orcid.org/0000-0001-7565-0284; https://orcid.org/0000-0002-0789-5105; https://orcid.org/0000-0003-3762-3052	8	American Petroleum Institute; Colorado School of Mines	US	Science	2015	349	1321-1326	1325	1352	110.42	112.67	91	5	Advancements in Solid Oxide Fuel Cells	Oxide	Fuel cells	Materials science	Ceramic; Oxide; Fuel cells; Materials science; Methane; Electrolyte; Chemical engineering; Cathode; Ion; Solid oxide fuel cell; Chemistry; Composite material; Electrode; Metallurgy; Organic chemistry; Engineering; Physical chemistry"""
        
        if st.button("📋 Load Sample Data", type="secondary", use_container_width=True):
            st.session_state.sample_data_loaded = sample_data
            st.rerun()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Load Data", type="primary", use_container_width=True):
                if data_input.strip():
                    with st.spinner("Processing data..."):
                        st.session_state.analyzer.parse_data(data_input)
                        st.success("✅ Data successfully loaded!")
                        st.session_state.plots_generated = False
                else:
                    st.error("❌ Please paste data first")
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.sample_data_loaded = ''
                st.rerun()
        
        if st.session_state.analyzer.df_processed is not None:
            st.subheader("📊 Data Information")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Papers", len(st.session_state.analyzer.df_processed))
            with col2:
                if 'year' in st.session_state.analyzer.df_processed.columns:
                    year_min = int(st.session_state.analyzer.df_processed['year'].min())
                    year_max = int(st.session_state.analyzer.df_processed['year'].max())
                    st.metric("Years", f"{year_min}-{year_max}")
            with col3:
                if 'count' in st.session_state.analyzer.df_processed.columns:
                    total_mentions = int(st.session_state.analyzer.df_processed['count'].sum())
                    st.metric("Total Mentions", f"{total_mentions:,}")
            with col4:
                if 'max_citations' in st.session_state.analyzer.df_processed.columns:
                    mean_max_cit = st.session_state.analyzer.df_processed['max_citations'].mean()
                    st.metric("Mean Max Citations", f"{mean_max_cit:.1f}")
            
            with st.expander("👁️ Preview Data"):
                st.dataframe(st.session_state.analyzer.df_processed.head(10))
    
    # ========================================================================
    # STAGE 2: PLOT SELECTION & SETTINGS
    # ========================================================================
    
    elif st.session_state.stage == 2:
        st.header("🎯 Stage 2: Plot Selection & Settings")
        
        if st.session_state.analyzer.df_processed is None:
            st.warning("⚠️ Please load data in Stage 1 first")
            return
        
        st.subheader("🎨 Visualization Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Color Palettes**")
            st.session_state.analyzer.heatmap_palette = st.selectbox(
                "Heatmap Palette", list(HEATMAP_PALETTES.keys()), index=0,
                help="Color scheme for heatmaps (journal-year, temporal, correlation)"
            )
            st.session_state.analyzer.bar_palette = st.selectbox(
                "Bar Chart Palette", list(BAR_PALETTES.keys()), index=0,
                help="Color scheme for bar charts"
            )
            st.session_state.analyzer.scatter_palette = st.selectbox(
                "Scatter Plot Palette", list(SCATTER_PALETTES.keys()), index=0,
                help="Color scheme for scatter plots and bubbles"
            )
        
        with col2:
            st.markdown("**Network & Box Plots**")
            st.session_state.analyzer.network_palette = st.selectbox(
                "Network Palette", list(NETWORK_PALETTES.keys()), index=0,
                help="Color scheme for network graphs"
            )
            st.session_state.analyzer.box_palette = st.selectbox(
                "Box Plot Palette", list(BOX_PALETTES.keys()), index=0,
                help="Color scheme for box plots"
            )
            st.session_state.analyzer.show_regression = st.checkbox(
                "Show Regression Lines", value=True,
                help="Display regression trend lines on scatter plots"
            )
        
        st.markdown("---")
        st.subheader("📊 Select Plots to Generate")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Select All", use_container_width=True):
                st.session_state.selected_plots = [plot[0] for plot in ALL_PLOTS]
                st.rerun()
        with col2:
            if st.button("❌ Clear All", use_container_width=True):
                st.session_state.selected_plots = []
                st.rerun()
        
        categories = {
            "📈 Distributions": ["1_distribution", "15_cumulative_influence"],
            "🌍 International Collaboration": ["2_country_network", "3_internationality", "5_collab_linear", "6_collab_log"],
            "📚 Journals & Publications": ["4_journal_heatmap", "17_journal_impact"],
            "🔗 Citations & References": ["6_1_bubble_chart", "6_2_bubble_chart", "13_cr_vs_oa", "16_references_impact"],
            "🏷️ Concepts & Topics": ["7_concepts", "8_concept_cooccurrence", "9_concept_influence", "21_concept_network_weighted"],
            "⏰ Temporal Analysis": ["10_temporal_evolution", "11_temporal_heatmap"],
            "👥 Teams & Organizations": ["11_team_size", "18_18_1_affiliation_network", "18_18_2_affiliation_network", "18_18_3_affiliation_network", "23_author_network"],
            "📊 Metrics Analysis": ["12_correlation", "14_domain_citations", "20_mds"],
            "🏛️ Hierarchical Structure": ["19_hierarchical_sankey"]
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
        
        st.markdown("---")
        if st.button("🚀 Generate Selected Plots", type="primary", use_container_width=True):
            if not st.session_state.selected_plots:
                st.error("❌ Please select at least one plot")
            else:
                with st.spinner("Generating plots..."):
                    st.session_state.analyzer.generate_all_plots(st.session_state.selected_plots)
                    st.session_state.plots_generated = True
                    st.success(f"✅ Generated {len(st.session_state.analyzer.all_figures)} plots!")
    
    # ========================================================================
    # STAGE 3: VISUALIZATION
    # ========================================================================
    
    elif st.session_state.stage == 3:
        st.header("📈 Stage 3: Visualization")
        
        if not st.session_state.plots_generated or not st.session_state.analyzer.all_figures:
            st.warning("⚠️ Please generate plots in Stage 2 first")
            return
        
        plot_names = list(st.session_state.analyzer.all_figures.keys())
        
        if len(plot_names) > 0:
            if 'current_plot_index' not in st.session_state:
                st.session_state.current_plot_index = 0
            
            selected_plot = st.selectbox(
                "Select Plot to View",
                options=plot_names,
                index=min(st.session_state.current_plot_index, len(plot_names)-1),
                format_func=lambda x: next(name for pid, name in ALL_PLOTS if pid == x)
            )
            
            st.session_state.current_plot_index = plot_names.index(selected_plot)
            
            if selected_plot in st.session_state.analyzer.all_figures:
                fig = st.session_state.analyzer.all_figures[selected_plot]
                
                if hasattr(fig, 'update_layout'):
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.pyplot(fig)
                
                plot_name = next(name for pid, name in ALL_PLOTS if pid == selected_plot)
                st.info(f"**{plot_name}**")
                
                col1, col2, col3 = st.columns(3)
                current_index = plot_names.index(selected_plot)
                
                with col1:
                    if current_index > 0:
                        if st.button("◀️ Previous"):
                            st.session_state.current_plot_index = current_index - 1
                            st.rerun()
                
                with col2:
                    st.write(f"Plot {current_index + 1} of {len(plot_names)}")
                
                with col3:
                    if current_index < len(plot_names) - 1:
                        if st.button("Next ▶️"):
                            st.session_state.current_plot_index = current_index + 1
                            st.rerun()
    
    # ========================================================================
    # STAGE 4: EXPORT
    # ========================================================================
    
    elif st.session_state.stage == 4:
        st.header("📥 Stage 4: Export Results")
        
        if not st.session_state.plots_generated or not st.session_state.analyzer.all_figures:
            st.warning("⚠️ Please generate plots in Stage 2 first")
            return
        
        st.success(f"✅ Available for download: {len(st.session_state.analyzer.all_figures)} plots")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Individual Plots")
            
            plot_options = {}
            for pid in st.session_state.analyzer.all_figures.keys():
                for plot_id, name in ALL_PLOTS:
                    if plot_id == pid:
                        plot_options[name] = pid
                        break
            
            selected_plot_name = st.selectbox("Select Plot", options=list(plot_options.keys()))
            
            if selected_plot_name:
                plot_id = plot_options[selected_plot_name]
                fig = st.session_state.analyzer.all_figures[plot_id]
                
                img_buffer = io.BytesIO()
                if hasattr(fig, 'update_layout'):
                    import plotly.io as pio
                    pio.write_image(fig, img_buffer, format='png', width=1200, height=900, scale=2)
                else:
                    fig.savefig(img_buffer, format='png', dpi=600, bbox_inches='tight')
                img_buffer.seek(0)
                
                st.download_button(
                    label=f"📥 Download {selected_plot_name}",
                    data=img_buffer,
                    file_name=f"plot_{selected_plot_name[:20].replace(' ', '_')}.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        with col2:
            st.subheader("📦 All Results")
            
            if st.button("📥 Download ZIP Archive", type="primary", use_container_width=True):
                with st.spinner("Creating ZIP archive..."):
                    zip_buffer = st.session_state.analyzer.save_all_to_zip(include_excel=True)
                    
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
        
        st.markdown("---")
        st.subheader("📊 Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Plots", len(st.session_state.analyzer.all_figures))
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


if __name__ == "__main__":
    main()
