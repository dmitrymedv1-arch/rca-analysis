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
import colorsys  # For smooth gradients in chord diagrams
import plotly.io as pio  # For HTML export of animations

# Streamlit page configuration
st.set_page_config(
    page_title="Scientific Data Visualization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style configuration for scientific plots
def set_scientific_style():
    """Set scientific style for matplotlib plots"""
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
    
    # Color palette for scientific graphics
    scientific_palette = [
        '#2E86AB', '#C73E1D', '#F18F01', '#6B8E23', '#8B5FBF',
        '#00A896', '#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0',
        '#118AB2', '#EF476F', '#073B4C', '#7209B7', '#F72585'
    ]
    
    plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=scientific_palette)

set_scientific_style()

# ============================================================================
# CLASS FOR DATA PROCESSING AND VISUALIZATION
# ============================================================================

class ScientificDataAnalyzer:
    """Class for scientific data analysis and visualization"""
    
    def __init__(self):
        self.df = None
        self.df_processed = None
        self.all_figures = {}
        self.plot_data = {}  # Stores data for each plot
        self.errors = []
        self.warnings = []
        self.progress = 0
        # Visualization settings
        self.show_regression_trends = True
        self.top_countries_chord = 20
        self.top_fields_sankey = 10
        # Temporal analysis settings
        self.analysis_year = datetime.now().year
        self.years_lookback = 5
        
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
        """Update progress bar value"""
        self.progress = value
    
    def update_visualization_settings(self, show_regression_trends=None, top_countries_chord=None,
                                       top_fields_sankey=None, analysis_year=None, years_lookback=None):
        """Update visualization settings"""
        if show_regression_trends is not None:
            self.show_regression_trends = show_regression_trends
        if top_countries_chord is not None:
            self.top_countries_chord = top_countries_chord
        if top_fields_sankey is not None:
            self.top_fields_sankey = top_fields_sankey
        if analysis_year is not None:
            self.analysis_year = analysis_year
        if years_lookback is not None:
            self.years_lookback = years_lookback
    
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
            
            # Check for required columns
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
        
        # Check for missing values
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
        
        # Key column statistics
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
            # Extract year from date if year column is missing
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
            
            # Normalized attention
            if 'count' in df_processed.columns:
                df_processed['normalized_attention'] = df_processed['count'] / df_processed['article_age']
        
        # Calculate maximum citations between CR and OA (use max_citations everywhere)
        if 'Citation counts (CR)' in df_processed.columns and 'Citation counts (OA)' in df_processed.columns:
            df_processed['max_citations'] = df_processed[['Citation counts (CR)', 'Citation counts (OA)']].max(axis=1)
            df_processed['max_annual_citations'] = df_processed[['Annual cit counts (CR)', 'Annual cit counts (OA)']].max(axis=1)
        elif 'Citation counts (CR)' in df_processed.columns:
            df_processed['max_citations'] = df_processed['Citation counts (CR)']
            df_processed['max_annual_citations'] = df_processed['Annual cit counts (CR)']
        elif 'Citation counts (OA)' in df_processed.columns:
            df_processed['max_citations'] = df_processed['Citation counts (OA)']
            df_processed['max_annual_citations'] = df_processed['Annual cit counts (OA)']
        
        # Number of countries and affiliations
        if 'countries_list' in df_processed.columns:
            df_processed['num_countries'] = df_processed['countries_list'].apply(len)
        
        if 'affiliations_list' in df_processed.columns:
            df_processed['num_affiliations'] = df_processed['affiliations_list'].apply(len)
        
        st.success("✅ Data preprocessing complete")
        return df_processed
    
    # ============================================================================
    # PLOT FUNCTIONS (30+ VISUALIZATION TYPES)
    # ============================================================================
    
    # ==================== PLOT 1: DISTRIBUTION OF ATTENTION ====================
    def plot_1_distribution_attention(self):
        """1. Distribution of attention (log-log, CCDF, Lorenz curve)"""
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
    
    # ==================== PLOT 2: COUNTRY CHORD DIAGRAM (ENHANCED) ====================
    def plot_2_country_chord_diagram(self):
        """2. Circular chord diagram for country collaborations (enhanced: chord thickness proportional to weight, text outside)"""
        try:
            if 'countries_list' not in self.df_processed.columns:
                return None
            
            # Collect collaboration data
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
            
            # Select top N countries
            top_countries = sorted(country_weights.items(), key=lambda x: x[1], reverse=True)[:self.top_countries_chord]
            top_country_names = [c[0] for c in top_countries]
            
            # Create adjacency matrix
            n = len(top_country_names)
            country_to_idx = {name: i for i, name in enumerate(top_country_names)}
            adjacency_matrix = np.zeros((n, n))
            
            for pair_data in country_pairs:
                if pair_data['country1'] in country_to_idx and pair_data['country2'] in country_to_idx:
                    i = country_to_idx[pair_data['country1']]
                    j = country_to_idx[pair_data['country2']]
                    adjacency_matrix[i, j] += pair_data['weight']
                    adjacency_matrix[j, i] += pair_data['weight']
            
            # Create color scheme
            colors = []
            for i in range(n):
                hue = i / n
                rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                color_hex = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
                colors.append(color_hex)
            
            # Arrange nodes in a circle
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            radius = 1.0
            x_positions = radius * np.cos(angles)
            y_positions = radius * np.sin(angles)
            
            # Text positions outside (at increased radius)
            text_radius = 1.25
            text_x = text_radius * np.cos(angles)
            text_y = text_radius * np.sin(angles)
            
            # Create figure
            fig = go.Figure()
            
            # Add external country labels (outside chords)
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
            
            # Add nodes (points on the circle)
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
            
            # Add chords (connections) with thickness proportional to weight
            max_weight = max(adjacency_matrix.flatten()) if adjacency_matrix.size > 0 else 1
            
            for i in range(n):
                for j in range(i+1, n):
                    weight = adjacency_matrix[i, j]
                    if weight > 0:
                        # Create smooth curve between points i and j
                        t = np.linspace(0, 1, 100)
                        
                        # Bezier curve between two points on the circle
                        p0 = np.array([x_positions[i], y_positions[i]])
                        p3 = np.array([x_positions[j], y_positions[j]])
                        
                        # Control points for outward curvature
                        mid_angle = (angles[i] + angles[j]) / 2
                        if abs(angles[i] - angles[j]) > np.pi:
                            mid_angle += np.pi
                        control_offset = 0.4 * (1 + weight / max_weight)
                        ctrl_point = np.array([control_offset * np.cos(mid_angle), control_offset * np.sin(mid_angle)])
                        
                        # 3rd order Bezier curve
                        curve_x = (1-t)**3 * p0[0] + 3*(1-t)**2*t * ctrl_point[0] + 3*(1-t)*t**2 * ctrl_point[0] + t**3 * p3[0]
                        curve_y = (1-t)**3 * p0[1] + 3*(1-t)**2*t * ctrl_point[1] + 3*(1-t)*t**2 * ctrl_point[1] + t**3 * p3[1]
                        
                        # Gradient chord color
                        start_rgb = [int(colors[i][1:3], 16), int(colors[i][3:5], 16), int(colors[i][5:7], 16)]
                        end_rgb = [int(colors[j][1:3], 16), int(colors[j][3:5], 16), int(colors[j][5:7], 16)]
                        
                        # Create single line with gradient (use one color for entire chord)
                        mixed_rgb = [(start_rgb[k] + end_rgb[k]) // 2 for k in range(3)]
                        chord_color = f'rgba({mixed_rgb[0]}, {mixed_rgb[1]}, {mixed_rgb[2]}, 0.8)'
                        
                        # Line thickness proportional to weight (normalized)
                        line_width = 3 + 15 * weight / max_weight
                        
                        fig.add_trace(go.Scatter(
                            x=curve_x,
                            y=curve_y,
                            mode='lines',
                            line=dict(width=line_width, color=chord_color),
                            hoverinfo='none',
                            showlegend=False
                        ))
            
            # Add inner circle
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
            
            # Add outer circumference
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
            
            # Save data
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
    
    # ==================== PLOT 3: INTERNATIONALITY VS CITATIONS (LINEAR) ====================
    def plot_3_internationality_vs_citations_linear(self):
        """3. Internationality vs Citations (linear scale)"""
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
            
            # Linear regression (if enabled)
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
            
            # Legend for point sizes
            sizes = [5, 10, 15]
            labels = ['5 authors', '10 authors', '15 authors']
            legend_elements = []
            for size, label in zip(sizes, labels):
                legend_elements.append(plt.scatter([], [], s=size*20, c='gray',
                                                  alpha=0.7, edgecolors='black',
                                                  label=label))
            
            ax.legend(handles=legend_elements, loc='upper left', title='Team Size')
            ax.grid(True, alpha=0.3)
            
            # Save data
            self.plot_data['3_internationality_vs_citations_linear'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_3_internationality_vs_citations_linear: {str(e)}")
            return None
    
    # ==================== PLOT 4: INTERNATIONALITY VS CITATIONS (LOGARITHMIC) ====================
    def plot_4_internationality_vs_citations_log(self):
        """4. Internationality vs Citations (logarithmic scale for Y)"""
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
            
            # Exponential regression (if enabled)
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
            
            # Legend for point sizes
            sizes = [5, 10, 15]
            labels = ['5 authors', '10 authors', '15 authors']
            legend_elements = []
            for size, label in zip(sizes, labels):
                legend_elements.append(plt.scatter([], [], s=size*20, c='gray',
                                                  alpha=0.7, edgecolors='black',
                                                  label=label))
            
            ax.legend(handles=legend_elements, loc='upper left', title='Team Size')
            ax.grid(True, alpha=0.3, which='both')
            
            # Save data
            self.plot_data['4_internationality_vs_citations_log'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_4_internationality_vs_citations_log: {str(e)}")
            return None
    
    # ==================== PLOT 5: JOURNAL HEATMAP ====================
    def plot_5_journal_year_heatmap(self, top_journals=15):
        """5. Heatmap: Journal vs Year (using max_annual_citations)"""
        try:
            required_cols = ['Full journal Name', 'year', 'max_annual_citations']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            # Select top journals
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
            
            # Filter years where all values are zero
            row_sums = pivot_table.sum(axis=1)
            pivot_table = pivot_table[row_sums > 0]
            
            if len(pivot_table) < 2:
                self.log_warning("Insufficient years with data for heatmap")
                return None
            
            # Save data
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
            
            # Add values
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
    
    # ==================== PLOT 6: COLLABORATION VS CITATIONS (LINEAR) ====================
    def plot_6_collaboration_vs_citations_linear(self):
        """6. Collaboration vs Citations relationship (linear scale)"""
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
                
                # Linear regression (if enabled)
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
            
            # Save data
            self.plot_data['6_collaboration_vs_citations_linear'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_6_collaboration_vs_citations_linear: {str(e)}")
            return None
    
    # ==================== PLOT 7: COLLABORATION VS CITATIONS (LOGARITHMIC) ====================
    def plot_7_collaboration_vs_citations_log(self):
        """7. Collaboration vs Citations relationship (logarithmic scale for Y)"""
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
                
                # Exponential regression (if enabled)
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
            
            # Save data
            self.plot_data['7_collaboration_vs_citations_log'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_7_collaboration_vs_citations_log: {str(e)}")
            return None
    
    # ==================== PLOT 8: REFERENCES VS CITATIONS (LINEAR) ====================
    def plot_8_references_vs_citations_linear(self):
        """8. Bubble chart: References vs Citations (linear scale)"""
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
            
            # Linear regression (if enabled)
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
            
            # Legend for sizes
            sizes = [5, 10, 15]
            labels = ['5 authors', '10 authors', '15 authors']
            legend_elements = [plt.scatter([], [], s=size*40, c='gray', alpha=0.7,
                                          edgecolors='black', label=label)
                              for size, label in zip(sizes, labels)]
            
            ax.legend(handles=legend_elements, title='Team Size', loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Save data
            self.plot_data['8_references_vs_citations_linear'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_8_references_vs_citations_linear: {str(e)}")
            return None
    
    # ==================== PLOT 9: REFERENCES VS CITATIONS (LOGARITHMIC) ====================
    def plot_9_references_vs_citations_log(self):
        """9. Bubble chart: References vs Citations (logarithmic scale for Y)"""
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
            
            # Exponential regression (if enabled)
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
            
            # Legend for sizes
            sizes = [5, 10, 15]
            labels = ['5 authors', '10 authors', '15 authors']
            legend_elements = [plt.scatter([], [], s=size*40, c='gray', alpha=0.7,
                                          edgecolors='black', label=label)
                              for size, label in zip(sizes, labels)]
            
            ax.legend(handles=legend_elements, title='Team Size', loc='upper left')
            ax.grid(True, alpha=0.3, which='both')
            
            # Save data
            self.plot_data['9_references_vs_citations_log'] = valid_data[required_cols].to_dict('records')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_9_references_vs_citations_log: {str(e)}")
            return None
    
    # ==================== PLOT 10: CONCEPTS ANALYSIS ====================
    def plot_10_concepts_analysis(self, top_n=30):
        """10. Concepts analysis (top 30 concepts)"""
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
            self.plot_data['10_concepts_analysis'] = {
                'top_concepts': top_concepts.to_dict(),
                'total_concepts': len(concept_counts)
            }
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
            
            # Graph 1: Bar chart
            y_pos = np.arange(len(top_concepts))
            colors = plt.cm.PuBu(np.linspace(0.3, 0.9, len(top_concepts)))
            
            bars = ax1.barh(y_pos, top_concepts.values, color=colors, edgecolor='black')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(top_concepts.index, fontsize=9)
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
                                colormap='viridis', max_words=100).generate_from_frequencies(concept_counts.to_dict())
            
            ax2.imshow(wordcloud, interpolation='bilinear')
            ax2.axis('off')
            ax2.set_title('Concept Word Cloud', fontweight='bold')
            
            # Set same Y axis limits
            ax1.set_ylim(-0.5, len(top_concepts) - 0.5)
            
            plt.suptitle('Research Concepts Analysis', fontweight='bold', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_10_concepts_analysis: {str(e)}")
            return None
    
    # ==================== PLOT 11: CONCEPT CO-OCCURRENCE MATRIX ====================
    def plot_11_concept_cooccurrence(self, top_n=15):
        """11. Concept co-occurrence matrix"""
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
                        fontweight='bold', fontsize=16)
            
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Co-occurrence Frequency', rotation=90, fontsize=12)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_11_concept_cooccurrence: {str(e)}")
            return None
    
    # ==================== PLOT 12: CONCEPT INFLUENCE ====================
    def plot_12_concept_influence(self):
        """12. Key concept influence (using max_citations)"""
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
            self.plot_data['12_concept_influence'] = concept_stats.reset_index().to_dict('records')
            
            fig, axes = plt.subplots(1, 2, figsize=(18, 10))
            
            # Graph 1: Mean citations
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
            
            # Graph 2: Bubble chart
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
            
            # Add annotations for top 5
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
    
    # ==================== PLOT 13: TEMPORAL EVOLUTION ====================
    def plot_13_temporal_evolution(self):
        """13. Evolution of publication activity and impact over time (using max_citations)"""
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
            self.plot_data['13_temporal_evolution'] = year_stats.reset_index().to_dict('records')
            
            fig, ax1 = plt.subplots(figsize=(14, 8))
            
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
                           label='Total Citations (max)')
            ax2.set_ylabel('Total Citations (max(CR, OA))', fontweight='bold', color='darkorange')
            ax2.tick_params(axis='y', labelcolor='darkorange')
            
            # Line: mean citations (additional)
            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('outward', 60))
            line2 = ax3.plot(year_stats.index, year_stats['mean_citations'], 
                           's-', color='darkgreen', linewidth=2, markersize=5,
                           label='Mean Citations per Paper (max)')
            ax3.set_ylabel('Mean Citations per Paper (max(CR, OA))', fontweight='bold', color='darkgreen')
            ax3.tick_params(axis='y', labelcolor='darkgreen')
            
            # Combine legends
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
    
    # ==================== PLOT 14: TEMPORAL HEATMAP ====================
    def plot_14_temporal_heatmap(self):
        """14. Heatmap: Publication year vs Article age (using max_annual_citations)"""
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
    
    # ==================== PLOT 15: TEAM SIZE ANALYSIS ====================
    def plot_15_team_size_analysis(self):
        """15. Team size analysis (using max_citations)"""
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
                'max_citations': 'mean',
                'references_count': 'mean'
            }).round(2)
            
            group_stats.columns = ['mean_attention', 'median_attention', 'std_attention',
                                 'num_papers', 'mean_citations', 'mean_references']
            
            # Order by increasing number of authors
            custom_order = ['Single author', '2 authors', '3 authors', '4-5 authors', 
                          '6-8 authors', '9-12 authors', '13+ authors', 'Unknown']
            
            # Filter only existing categories
            existing_categories = [cat for cat in custom_order if cat in group_stats.index]
            group_stats = group_stats.loc[existing_categories]
            
            # Save data
            self.plot_data['15_team_size_analysis'] = group_stats.reset_index().to_dict('records')
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            # Graph 1: Distribution of team sizes
            team_size_counts = self.df_processed['team_size_group'].value_counts()
            team_size_counts = team_size_counts.reindex(existing_categories, fill_value=0)
            axes[0].bar(team_size_counts.index, team_size_counts.values,
                       alpha=0.7, color='steelblue', edgecolor='black')
            axes[0].set_xlabel('Team Size', fontweight='bold')
            axes[0].set_ylabel('Number of Papers', fontweight='bold')
            axes[0].set_title('Distribution of Team Sizes', fontweight='bold')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # Graph 2: Mean attention by team size
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
            
            # Graph 3: Mean citations
            axes[2].bar(group_stats.index, group_stats['mean_citations'],
                       alpha=0.7, color='darkgreen', edgecolor='black')
            axes[2].set_xlabel('Team Size', fontweight='bold')
            axes[2].set_ylabel('Mean Citations (max(CR, OA))', fontweight='bold')
            axes[2].set_title('Mean Citations by Team Size', fontweight='bold')
            axes[2].tick_params(axis='x', rotation=45)
            axes[2].grid(True, alpha=0.3, axis='y')
            
            # Graph 4: Mean references
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
    
    # ==================== PLOT 16: CORRELATION MATRIX ====================
    def plot_16_correlation_matrix(self):
        """16. Correlation matrix with key parameters highlighted (using max_citations)"""
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
            
            # Reorder matrix: key parameters first
            key_params = ['count', 'max_citations', 'max_annual_citations']
            
            # Filter only those present in data
            existing_key_params = [p for p in key_params if p in corr_matrix.columns]
            other_params = [p for p in corr_matrix.columns if p not in existing_key_params]
            
            # New order
            new_order = existing_key_params + other_params
            corr_matrix = corr_matrix.reindex(index=new_order, columns=new_order)
            
            # Save data
            self.plot_data['16_correlation_matrix'] = {
                'correlation_matrix': corr_matrix.to_dict(),
                'columns': available_cols,
                'method': 'spearman',
                'key_parameters': existing_key_params
            }
            
            fig, ax = plt.subplots(figsize=(14, 12))
            
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Draw heatmap
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                       cmap='coolwarm', center=0, square=True,
                       linewidths=0.5, cbar_kws={'shrink': 0.8},
                       ax=ax, annot_kws={'fontsize': 9})
            
            # Add highlighting for key parameters
            key_param_indices = [i for i, col in enumerate(corr_matrix.columns) if col in existing_key_params]
            for idx in key_param_indices:
                # Highlight rows
                ax.add_patch(plt.Rectangle((0, idx), len(corr_matrix), 1, 
                                          fill=False, edgecolor='red', linewidth=2, 
                                          alpha=0.7))
                # Highlight columns
                ax.add_patch(plt.Rectangle((idx, 0), 1, len(corr_matrix), 
                                          fill=False, edgecolor='red', linewidth=2, 
                                          alpha=0.7))
            
            # Add legend for highlighting
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
    
    # ==================== PLOT 17: CR VS OA COMPARISON ====================
    def plot_17_citation_sources_comparison(self):
        """17. Comparison of CR vs OA citations (keeping for source comparison)"""
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
            
            # Graph 1: Scatter plot
            max_val = max(valid_data['Citation counts (CR)'].max(),
                         valid_data['Citation counts (OA)'].max())
            
            ax1.scatter(valid_data['Citation counts (CR)'],
                       valid_data['Citation counts (OA)'],
                       alpha=0.6, c='steelblue', edgecolors='black', linewidth=0.5)
            
            # Line y=x with regression (if enabled)
            ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y = x')
            
            # Linear regression (if enabled)
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
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.suptitle('Comparison of Citation Sources', fontweight='bold', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_17_citation_sources_comparison: {str(e)}")
            return None
    
    # ==================== PLOT 18: CITATIONS BY DOMAIN ====================
    def plot_18_citation_by_domain(self):
        """18. Citation impact by scientific domain (using max_annual_citations)"""
        try:
            required_cols = ['Domain', 'max_annual_citations']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) == 0:
                return None
            
            # Aggregate by domain
            domain_stats = valid_data.groupby('Domain').agg({
                'max_annual_citations': ['median', 'mean', 'std', 'count'],
                'count': 'mean'
            }).round(2)
            
            domain_stats.columns = ['median_citations', 'mean_citations', 'std_citations',
                                  'num_papers', 'mean_attention']
            domain_stats = domain_stats.sort_values('median_citations', ascending=False)
            
            # Save data
            self.plot_data['18_citation_by_domain'] = domain_stats.reset_index().to_dict('records')
            
            # Select top domains
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
            
            # Add mean values
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
    
    # ==================== PLOT 19: CUMULATIVE INFLUENCE ====================
    def plot_19_cumulative_influence(self):
        """19. Cumulative influence curve"""
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
            self.plot_data['19_cumulative_influence'] = {
                'sorted_counts': sorted_counts.tolist(),
                'cumulative_percentage': cumulative_percentage.tolist(),
                'article_percentage': article_percentage.tolist(),
                'total_citations': float(total_citations),
                'gini_coefficient': self._calculate_gini(sorted_counts.values)
            }
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
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
            ax1.legend(loc='lower right')
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
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.suptitle('Analysis of Local Influence Within Dataset', fontweight='bold', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_19_cumulative_influence: {str(e)}")
            return None
    
    def _calculate_gini(self, x):
        """Calculate Gini coefficient"""
        x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(x, dtype=float)
        if cumx[-1] == 0:
            return 0
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    
    # ==================== PLOT 20: REFERENCES VS ATTENTION ====================
    def plot_20_references_vs_attention(self):
        """20. References count vs attention (using max_citations)"""
        try:
            required_cols = ['references_count', 'count', 'max_annual_citations']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) < 10:
                return None
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # Graph 1: References vs Attention
            scatter1 = ax1.scatter(valid_data['references_count'],
                                 valid_data['count'],
                                 c=valid_data['max_annual_citations'],
                                 cmap='viridis', alpha=0.6, s=30,
                                 edgecolors='black', linewidth=0.5)
            
            # Linear regression (if enabled)
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
            
            # Graph 2: References vs Citations
            scatter2 = ax2.scatter(valid_data['references_count'],
                                 valid_data['max_annual_citations'],
                                 c=valid_data['count'],
                                 cmap='plasma', alpha=0.6, s=30,
                                 edgecolors='black', linewidth=0.5)
            
            # Linear regression (if enabled)
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
    
    # ==================== PLOT 21: JOURNAL IMPACT ====================
    def plot_21_journal_impact(self):
        """21. Journal impact analysis (using max_citations)"""
        try:
            required_cols = ['Full journal Name', 'count', 'max_annual_citations', 'max_citations']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            if len(valid_data) == 0:
                return None
            
            # Aggregate by journal
            journal_stats = valid_data.groupby('Full journal Name').agg({
                'count': ['mean', 'median', 'std', 'size'],
                'max_annual_citations': 'mean',
                'references_count': 'mean',
                'max_citations': 'mean'
            }).round(2)
            
            journal_stats.columns = ['mean_attention', 'median_attention', 'std_attention',
                                   'num_papers', 'mean_annual_citations', 'mean_references', 'mean_citations']
            
            # Filter journals with sufficient papers
            journal_stats = journal_stats[journal_stats['num_papers'] >= 3]
            journal_stats = journal_stats.sort_values('mean_attention', ascending=False)
            
            # Save data
            self.plot_data['21_journal_impact'] = journal_stats.reset_index().to_dict('records')
            
            # Select top journals
            top_journals = journal_stats.head(15)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Graph 1: Mean attention
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
            
            # Add values
            for bar, (_, row) in zip(bars1, top_journals.iterrows()):
                width = bar.get_width()
                info_text = f"n={int(row['num_papers'])}"
                ax1.text(width * 1.01, bar.get_y() + bar.get_height()/2,
                        info_text, va='center', fontsize=8)
            
            # Graph 2: Bubble chart
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
            
            # Add annotations
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
    
    # ==================== PLOT 22: HIERARCHICAL SANKEY DIAGRAM ====================
    def plot_22_hierarchical_sankey(self):
        """22. Hierarchical Sankey diagram: Domain → Field → Subfield → Topic (with field limits, using max_citations)"""
        try:
            required_cols = ['Domain', 'Field', 'Subfield', 'Topic', 'max_citations']
            available_cols = [col for col in required_cols if col in self.df_processed.columns]
            
            if len(available_cols) < 3:
                return None
            
            valid_data = self.df_processed.dropna(subset=available_cols)
            if len(valid_data) < 10:
                return None
            
            # Limit number of fields (Field) for readability
            # First determine top fields by total citations
            field_citations = valid_data.groupby('Field')['max_citations'].sum().sort_values(ascending=False)
            top_fields = field_citations.head(self.top_fields_sankey).index.tolist()
            
            # Filter data only for top fields
            filtered_data = valid_data[valid_data['Field'].isin(top_fields)]
            
            if len(filtered_data) < 5:
                self.log_warning(f"Insufficient data after filtering to top {self.top_fields_sankey} fields")
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
            
            # Aggregate weights (sum citations)
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
                
                # Add links
                domain_idx = add_node(domain)
                field_idx = add_node(field)
                subfield_idx = add_node(subfield)
                topic_idx = add_node(topic)
                
                links.append({'source': domain_idx, 'target': field_idx, 'value': weight})
                links.append({'source': field_idx, 'target': subfield_idx, 'value': weight})
                links.append({'source': subfield_idx, 'target': topic_idx, 'value': weight})
            
            # Save data
            self.plot_data['22_hierarchical_sankey'] = {
                'nodes': nodes,
                'links': links,
                'total_weight': sum([l['value'] for l in links]),
                'top_fields_used': top_fields,
                'fields_limit': self.top_fields_sankey
            }
            
            # Create Sankey diagram with plotly
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
            
            # Additional settings for node text
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
    
    # ==================== PLOT 23: MULTIDIMENSIONAL SCALING ====================
    def plot_23_multidimensional_scaling(self):
        """23. Multidimensional scaling of important predictors (using max_citations)"""
        try:
            # Select key predictors
            predictors = ['author count', 'references_count', 'num_countries',
                         'max_annual_citations', 'article_age', 'normalized_attention']
            
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
            self.plot_data['23_mds_analysis'] = {
                'explained_variance': pca.explained_variance_ratio_.tolist(),
                'components': pca.components_.tolist(),
                'pca_coordinates': pca_result.tolist(),
                'predictors': available_predictors
            }
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # Graph 1: PCA scatter plot
            scatter = ax1.scatter(pca_result[:, 0], pca_result[:, 1],
                                 c=analysis_data['count'], cmap='viridis',
                                 alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
            
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontweight='bold')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontweight='bold')
            ax1.set_title('PCA: Multidimensional Scaling of Predictors', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Local Mentions (count)', fontweight='bold')
            
            # Graph 2: Feature importance
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            
            for i, predictor in enumerate(available_predictors):
                ax2.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                         color='red', alpha=0.5, head_width=0.05)
                ax2.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15,
                        predictor, color='red', fontsize=10, fontweight='bold')
            
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
            
            plt.suptitle('Multidimensional Analysis of Research Predictors', fontweight='bold', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_23_multidimensional_scaling: {str(e)}")
            return None
    
    # ==================== PLOT 24: WEIGHTED CONCEPT NETWORK ====================
    def plot_24_concept_network_weighted(self):
        """24. Concept network with weights by influence (using max_citations)"""
        try:
            if 'concepts_list' not in self.df_processed.columns or 'max_citations' not in self.df_processed.columns:
                return None
            
            # Collect top concepts by occurrence
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
            
            # Add nodes with weights by citations
            for concept in top_concepts:
                # Find all papers with this concept
                concept_papers = []
                for idx, row in self.df_processed.iterrows():
                    if isinstance(row['concepts_list'], list) and concept in row['concepts_list']:
                        concept_papers.append(row.get('max_citations', 0))
                
                total_citations = sum(concept_papers)
                G.add_node(concept, citations=total_citations, papers=len(concept_papers))
            
            # Add edges with weights by co-occurrence
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
            self.plot_data['24_concept_network_weighted'] = {
                'nodes': [{'concept': node, 'citations': G.nodes[node]['citations'],
                          'papers': G.nodes[node]['papers']} for node in G.nodes()],
                'edges': [{'concept1': u, 'concept2': v, 'weight': G[u][v]['weight']} 
                         for u, v in G.edges()]
            }
            
            # Visualization
            fig, ax = plt.subplots(figsize=(16, 12))
            
            pos = nx.spring_layout(G, k=2, seed=42)
            
            # Node sizes by citations
            node_sizes = [G.nodes[n]['citations'] * 0.2 + 500 for n in G.nodes()]
            node_colors = [G.nodes[n]['papers'] for n in G.nodes()]
            
            nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                          node_color=node_colors, cmap='RdYlGn',
                                          alpha=0.8, edgecolors='black', linewidths=1.5, ax=ax)
            
            # Edges with thickness by weight
            if G.edges():
                edge_weights = [G[u][v]['weight'] * 0.01 for u, v in G.edges()]
                edges = nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5,
                                              edge_color='gray', style='solid', ax=ax)
            
            # Labels
            nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
            
            ax.set_title('Concept Network with Citation Impact', fontweight='bold', fontsize=16)
            ax.axis('off')
            
            # Color bar
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
    
    # ==================== NEW PLOT 25: TEMPORAL TERM ACTIVITY (VIOLIN PLOT) ====================
    def plot_25_term_temporal_density(self, hierarchy_level='Topic', top_percent=15, metric='total_attention'):
        """
        25. Temporal activity analysis for hierarchical terms (Topic/Subfield/Field/Domain)
        Shows first appearance, peak density, and last appearance as violin plots
        
        Parameters:
        - hierarchy_level: 'Topic', 'Subfield', 'Field', 'Domain', 'Concepts'
        - top_percent: percentage of top terms to show (default 15%)
        - metric: 'total_attention', 'activity_span', 'peak_density' for sorting
        """
        try:
            # Determine which column to use
            if hierarchy_level == 'Concepts':
                if 'concepts_list' not in self.df_processed.columns:
                    self.log_warning(f"Concepts data not available for plot 25")
                    return None
                # Explode concepts list to individual rows
                exploded_data = []
                for idx, row in self.df_processed.iterrows():
                    if isinstance(row['concepts_list'], list) and len(row['concepts_list']) > 0:
                        for concept in row['concepts_list']:
                            if concept and str(concept).strip():
                                exploded_data.append({
                                    'term': concept.strip(),
                                    'year': row.get('year', None),
                                    'count': row.get('count', 1),
                                    'max_citations': row.get('max_citations', 0),
                                    'max_annual_citations': row.get('max_annual_citations', 0)
                                })
                if not exploded_data:
                    self.log_warning("No concept data available")
                    return None
                term_df = pd.DataFrame(exploded_data)
                term_df = term_df.dropna(subset=['year', 'term'])
                term_column = 'term'
            else:
                # Check if the hierarchy level exists
                if hierarchy_level not in self.df_processed.columns:
                    self.log_warning(f"Column '{hierarchy_level}' not found for plot 25")
                    return None
                
                term_df = self.df_processed[[hierarchy_level, 'year', 'count', 'max_citations', 'max_annual_citations']].copy()
                term_df = term_df.dropna(subset=[hierarchy_level, 'year'])
                term_df.rename(columns={hierarchy_level: 'term'}, inplace=True)
                term_column = 'term'
            
            if len(term_df) == 0:
                self.log_warning(f"No valid data for hierarchy level '{hierarchy_level}'")
                return None
            
            # Filter years to reasonable range
            current_year = datetime.now().year
            term_df = term_df[term_df['year'] <= current_year]
            term_df = term_df[term_df['year'] >= 1900]  # Reasonable lower bound
            
            if len(term_df) == 0:
                self.log_warning("No valid years in data")
                return None
            
            # Calculate term statistics
            term_stats = []
            
            for term in term_df['term'].unique():
                term_data = term_df[term_df['term'] == term]
                years = term_data['year'].values
                if len(years) < 2:
                    continue
                
                first_year = int(years.min())
                last_year = int(years.max())
                activity_span = last_year - first_year
                
                if activity_span < 1:
                    continue
                
                # Calculate peak density (year with maximum count)
                year_counts = term_data.groupby('year')['count'].sum()
                peak_year = int(year_counts.idxmax())
                peak_density = float(year_counts.max())
                
                total_attention = float(term_data['count'].sum())
                total_citations = float(term_data['max_citations'].sum())
                total_annual_citations = float(term_data['max_annual_citations'].sum())
                num_papers = len(term_data)
                
                # Collect all years for violin plot (repeat years by count/weight)
                weighted_years = []
                for _, row in term_data.iterrows():
                    weight = int(row['count']) if metric == 'total_attention' else 1
                    weighted_years.extend([int(row['year'])] * max(1, weight))
                
                term_stats.append({
                    'term': term,
                    'first_year': first_year,
                    'peak_year': peak_year,
                    'last_year': last_year,
                    'activity_span': activity_span,
                    'peak_density': peak_density,
                    'total_attention': total_attention,
                    'total_citations': total_citations,
                    'total_annual_citations': total_annual_citations,
                    'num_papers': num_papers,
                    'years_list': weighted_years
                })
            
            if len(term_stats) == 0:
                self.log_warning("No terms with sufficient temporal data")
                return None
            
            term_stats_df = pd.DataFrame(term_stats)
            
            # Select top N% terms based on chosen metric
            if metric == 'total_attention':
                term_stats_df = term_stats_df.sort_values('total_attention', ascending=False)
            elif metric == 'activity_span':
                term_stats_df = term_stats_df.sort_values('activity_span', ascending=False)
            elif metric == 'peak_density':
                term_stats_df = term_stats_df.sort_values('peak_density', ascending=False)
            else:
                term_stats_df = term_stats_df.sort_values('total_attention', ascending=False)
            
            top_n = max(5, int(len(term_stats_df) * top_percent / 100))
            top_terms = term_stats_df.head(top_n)
            
            # Save data
            self.plot_data['25_term_temporal_density'] = {
                'hierarchy_level': hierarchy_level,
                'top_percent': top_percent,
                'sorting_metric': metric,
                'terms_analyzed': len(term_stats_df),
                'top_terms': top_terms[['term', 'first_year', 'peak_year', 'last_year', 
                                        'activity_span', 'total_attention', 'num_papers']].to_dict('records'),
                'analysis_year': self.analysis_year,
                'years_lookback': self.years_lookback
            }
            
            # Create figure with violin plots
            fig = go.Figure()
            
            # Sort terms by first_year for better visualization
            top_terms = top_terms.sort_values('first_year')
            
            # Create violin plots for each term
            for idx, row in top_terms.iterrows():
                term_name = row['term']
                if len(term_name) > 30:
                    term_name = term_name[:27] + '...'
                
                # Add violin trace
                fig.add_trace(go.Violin(
                    y=row['years_list'],
                    x=[term_name] * len(row['years_list']),
                    name=term_name,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor='lightblue',
                    opacity=0.7,
                    line_color='black',
                    line_width=1.5,
                    points=False,
                    side='positive',
                    spanmode='hard',
                    bandwidth=1.0
                ))
                
                # Add markers for first_year, peak_year, last_year
                fig.add_trace(go.Scatter(
                    x=[term_name],
                    y=[row['first_year']],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=1, color='black')),
                    name=f'First appearance',
                    showlegend=(idx == 0),
                    hovertext=f"{row['term']}<br>First year: {row['first_year']}",
                    hoverinfo='text'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[term_name],
                    y=[row['peak_year']],
                    mode='markers',
                    marker=dict(symbol='circle', size=14, color='red', line=dict(width=2, color='black')),
                    name=f'Peak density',
                    showlegend=(idx == 0),
                    hovertext=f"{row['term']}<br>Peak year: {row['peak_year']}<br>Peak density: {row['peak_density']:.0f}",
                    hoverinfo='text'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[term_name],
                    y=[row['last_year']],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='blue', line=dict(width=1, color='black')),
                    name=f'Last appearance',
                    showlegend=(idx == 0),
                    hovertext=f"{row['term']}<br>Last year: {row['last_year']}",
                    hoverinfo='text'
                ))
            
            # Add trend line for first appearances
            first_years = top_terms['first_year'].values
            x_positions = list(range(len(top_terms)))
            fig.add_trace(go.Scatter(
                x=x_positions,
                y=first_years,
                mode='lines+markers',
                line=dict(color='green', width=2, dash='dash'),
                marker=dict(size=6, color='green'),
                name='First appearance trend',
                showlegend=True
            ))
            
            # Add trend line for peak years
            peak_years = top_terms['peak_year'].values
            fig.add_trace(go.Scatter(
                x=x_positions,
                y=peak_years,
                mode='lines+markers',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=6, color='red'),
                name='Peak year trend',
                showlegend=True
            ))
            
            fig.update_layout(
                title=dict(
                    text=f"Temporal Activity of Top {top_percent}% {hierarchy_level}s<br>"
                         f"Sorted by {metric.replace('_', ' ').title()} | Violin shows year distribution",
                    font=dict(size=16, weight='bold')
                ),
                xaxis=dict(
                    title=f"{hierarchy_level} (Top {top_percent}%)",
                    tickangle=45,
                    tickfont=dict(size=10)
                ),
                yaxis=dict(
                    title="Publication Year",
                    range=[first_years.min() - 2, self.analysis_year + 2],
                    dtick=2,
                    gridcolor='lightgray',
                    gridwidth=0.5
                ),
                width=1400,
                height=800,
                plot_bgcolor='white',
                hovermode='closest',
                legend=dict(
                    x=1.02,
                    y=0.98,
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='black',
                    borderwidth=1
                )
            )
            
            # Add horizontal line for current analysis year
            fig.add_hline(y=self.analysis_year, line_dash="dot", line_color="gray",
                         annotation_text=f"Analysis year: {self.analysis_year}",
                         annotation_position="bottom right")
            
            # Add horizontal line for 5 years lookback
            lookback_year = self.analysis_year - self.years_lookback
            fig.add_hline(y=lookback_year, line_dash="dot", line_color="orange",
                         annotation_text=f"5 years lookback: {lookback_year}",
                         annotation_position="bottom left")
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_25_term_temporal_density: {str(e)}")
            return None
    
    # ==================== PLOT 25 SECONDARY: 5-YEAR ACTIVE TERMS ====================
    def plot_25b_active_terms_last_5_years(self, hierarchy_level='Topic', top_n=20):
        """
        25b. Active terms analysis for last 5 years
        Shows terms with highest activity in the most recent 5-year period
        """
        try:
            # Determine which column to use
            if hierarchy_level == 'Concepts':
                if 'concepts_list' not in self.df_processed.columns:
                    self.log_warning(f"Concepts data not available for plot 25b")
                    return None
                # Explode concepts list to individual rows
                exploded_data = []
                for idx, row in self.df_processed.iterrows():
                    if isinstance(row['concepts_list'], list) and len(row['concepts_list']) > 0:
                        for concept in row['concepts_list']:
                            if concept and str(concept).strip():
                                exploded_data.append({
                                    'term': concept.strip(),
                                    'year': row.get('year', None),
                                    'count': row.get('count', 1),
                                    'max_citations': row.get('max_citations', 0)
                                })
                if not exploded_data:
                    self.log_warning("No concept data available")
                    return None
                term_df = pd.DataFrame(exploded_data)
                term_df = term_df.dropna(subset=['year', 'term'])
                term_column = 'term'
            else:
                # Check if the hierarchy level exists
                if hierarchy_level not in self.df_processed.columns:
                    self.log_warning(f"Column '{hierarchy_level}' not found for plot 25b")
                    return None
                
                term_df = self.df_processed[[hierarchy_level, 'year', 'count', 'max_citations']].copy()
                term_df = term_df.dropna(subset=[hierarchy_level, 'year'])
                term_df.rename(columns={hierarchy_level: 'term'}, inplace=True)
                term_column = 'term'
            
            if len(term_df) == 0:
                self.log_warning(f"No valid data for hierarchy level '{hierarchy_level}'")
                return None
            
            # Filter last 5 years
            current_year = self.analysis_year
            lookback_year = current_year - self.years_lookback
            
            recent_df = term_df[term_df['year'] >= lookback_year]
            historical_df = term_df[term_df['year'] < lookback_year]
            
            if len(recent_df) == 0:
                self.log_warning(f"No data in last {self.years_lookback} years")
                return None
            
            # Calculate term statistics for recent period
            recent_stats = recent_df.groupby('term').agg({
                'count': ['sum', 'mean', 'size'],
                'max_citations': 'mean'
            }).round(2)
            recent_stats.columns = ['total_attention', 'mean_attention', 'num_papers', 'mean_citations']
            
            # Calculate historical attention for comparison
            if len(historical_df) > 0:
                historical_attention = historical_df.groupby('term')['count'].sum().to_dict()
                recent_stats['historical_attention'] = recent_stats.index.map(
                    lambda x: historical_attention.get(x, 0)
                )
                recent_stats['attention_growth'] = recent_stats['total_attention'] - recent_stats['historical_attention']
                recent_stats['growth_factor'] = recent_stats['total_attention'] / recent_stats['historical_attention'].replace(0, 1)
            else:
                recent_stats['historical_attention'] = 0
                recent_stats['attention_growth'] = recent_stats['total_attention']
                recent_stats['growth_factor'] = recent_stats['total_attention']
            
            # Select top terms by total attention
            recent_stats = recent_stats.sort_values('total_attention', ascending=False).head(top_n)
            recent_stats = recent_stats.reset_index()
            
            # Save data
            self.plot_data['25b_active_terms_5_years'] = {
                'hierarchy_level': hierarchy_level,
                'lookback_years': self.years_lookback,
                'analysis_year': self.analysis_year,
                'top_terms': recent_stats.to_dict('records')
            }
            
            # Create visualization
            fig = go.Figure()
            
            # Add bars for recent attention
            fig.add_trace(go.Bar(
                x=recent_stats['term'],
                y=recent_stats['total_attention'],
                name=f'Last {self.years_lookback} years',
                marker_color='#2E86AB',
                text=recent_stats['total_attention'].round(0).astype(int),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>' +
                              f'Last {self.years_lookback} years: %{{y:.0f}}<br>' +
                              'Papers: %{customdata[0]}<br>' +
                              'Mean citations: %{customdata[1]:.1f}<extra></extra>',
                customdata=np.column_stack([recent_stats['num_papers'], recent_stats['mean_citations']])
            ))
            
            # Add bars for historical attention (if any)
            if (recent_stats['historical_attention'] > 0).any():
                fig.add_trace(go.Bar(
                    x=recent_stats['term'],
                    y=recent_stats['historical_attention'],
                    name=f'Before {lookback_year}',
                    marker_color='lightgray',
                    text=recent_stats['historical_attention'].round(0).astype(int),
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>' +
                                  f'Before {lookback_year}: %{{y:.0f}}<extra></extra>'
                ))
            
            # Add line for growth factor on secondary axis
            fig.add_trace(go.Scatter(
                x=recent_stats['term'],
                y=recent_stats['growth_factor'],
                name='Growth factor (recent / historical)',
                yaxis='y2',
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=8, color='red', symbol='diamond'),
                text=recent_stats['growth_factor'].round(1),
                textposition='top center',
                hovertemplate='<b>%{x}</b><br>Growth factor: %{y:.1f}x<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(
                    text=f"Most Active {hierarchy_level}s in Last {self.years_lookback} Years ({lookback_year}-{current_year})<br>"
                         f"Top {top_n} by Total Attention | Growth factor = Recent / Historical",
                    font=dict(size=16, weight='bold')
                ),
                xaxis=dict(
                    title=hierarchy_level,
                    tickangle=45,
                    tickfont=dict(size=10)
                ),
                yaxis=dict(
                    title="Total Attention (sum of count)",
                    gridcolor='lightgray',
                    gridwidth=0.5
                ),
                yaxis2=dict(
                    title="Growth Factor",
                    overlaying='y',
                    side='right',
                    showgrid=False,
                    rangemode='tozero'
                ),
                width=1400,
                height=700,
                plot_bgcolor='white',
                barmode='group',
                legend=dict(
                    x=1.02,
                    y=0.98,
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='black',
                    borderwidth=1
                )
            )
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_25b_active_terms_last_5_years: {str(e)}")
            return None
    
    # ==================== NEW PLOT 26: ANIMATED JOURNAL HEATMAP ====================
    def plot_26_animated_journal_heatmap(self, top_journals=15):
        """
        26. Animated heatmap: Journal performance over years
        Shows how journal citation impact evolves with animation frame by year
        """
        try:
            required_cols = ['Full journal Name', 'year', 'max_annual_citations']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            # Select top journals by number of papers
            journal_counts = self.df_processed['Full journal Name'].value_counts()
            top_journals_list = journal_counts.head(top_journals).index.tolist()
            
            # Prepare data for animation
            heatmap_data = self.df_processed[
                self.df_processed['Full journal Name'].isin(top_journals_list)
            ].copy()
            
            if len(heatmap_data) == 0:
                return None
            
            # Aggregate by year and journal
            aggregated = heatmap_data.groupby(['year', 'Full journal Name']).agg({
                'max_annual_citations': 'mean',
                'doi': 'count'
            }).reset_index()
            aggregated.columns = ['year', 'journal', 'mean_annual_citations', 'num_papers']
            
            # Filter years with sufficient data
            year_counts = aggregated.groupby('year')['num_papers'].sum()
            valid_years = year_counts[year_counts >= len(top_journals_list) * 0.3].index.tolist()
            aggregated = aggregated[aggregated['year'].isin(valid_years)]
            
            if len(aggregated) == 0:
                self.log_warning("Insufficient data for animated journal heatmap")
                return None
            
            # Create animated heatmap
            fig = px.density_heatmap(
                aggregated,
                x='journal',
                y='year',
                z='mean_annual_citations',
                animation_frame='year',
                range_color=[0, aggregated['mean_annual_citations'].quantile(0.95)],
                color_continuous_scale='Viridis',
                title=f"Animated Journal Heatmap: Annual Citation Rate Evolution<br>Top {top_journals} Journals",
                labels={
                    'journal': 'Journal',
                    'year': 'Publication Year',
                    'mean_annual_citations': 'Mean Annual Citations (max(CR, OA))'
                },
                text_auto='.1f'
            )
            
            fig.update_layout(
                width=1200,
                height=800,
                xaxis=dict(tickangle=45, tickfont=dict(size=9)),
                yaxis=dict(title="Publication Year", autorange='reversed'),
                hovermode='closest'
            )
            
            # Save data
            self.plot_data['26_animated_journal_heatmap'] = {
                'top_journals': top_journals_list,
                'data': aggregated.to_dict('records'),
                'years': valid_years
            }
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_26_animated_journal_heatmap: {str(e)}")
            return None
    
    # ==================== NEW PLOT 27: ANIMATED COUNTRY CHORD (PERIODS) ====================
    def plot_27_animated_country_chord_periods(self, periods=4):
        """
        27. Animated chord diagram showing country collaboration evolution across time periods
        Creates separate chord diagrams for different time periods and animates between them
        """
        try:
            if 'countries_list' not in self.df_processed.columns or 'year' not in self.df_processed.columns:
                return None
            
            # Determine time periods
            min_year = int(self.df_processed['year'].min())
            max_year = int(self.df_processed['year'].max())
            
            if max_year - min_year < periods:
                period_years = max_year - min_year + 1
                periods = max(2, period_years)
            
            years_per_period = max(2, (max_year - min_year) // periods)
            periods_list = []
            
            for i in range(periods):
                start_year = min_year + i * years_per_period
                end_year = min(start_year + years_per_period - 1, max_year)
                if start_year <= end_year:
                    periods_list.append((start_year, end_year, f"{start_year}-{end_year}"))
            
            if len(periods_list) < 2:
                self.log_warning("Insufficient year range for animated chord diagram")
                return None
            
            # Collect collaboration data for each period
            period_frames = []
            
            for period_idx, (start_year, end_year, period_label) in enumerate(periods_list):
                period_data = self.df_processed[
                    (self.df_processed['year'] >= start_year) & 
                    (self.df_processed['year'] <= end_year)
                ]
                
                if len(period_data) < 5:
                    continue
                
                # Collect country pairs for this period
                country_pairs = []
                country_weights = defaultdict(float)
                
                for idx, row in period_data.iterrows():
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
                    continue
                
                # Select top countries for this period
                top_countries = sorted(country_weights.items(), key=lambda x: x[1], reverse=True)[:self.top_countries_chord]
                top_country_names = [c[0] for c in top_countries]
                
                # Create adjacency matrix
                n = len(top_country_names)
                country_to_idx = {name: i for i, name in enumerate(top_country_names)}
                adjacency_matrix = np.zeros((n, n))
                
                for pair_data in country_pairs:
                    if pair_data['country1'] in country_to_idx and pair_data['country2'] in country_to_idx:
                        i = country_to_idx[pair_data['country1']]
                        j = country_to_idx[pair_data['country2']]
                        adjacency_matrix[i, j] += pair_data['weight']
                        adjacency_matrix[j, i] += pair_data['weight']
                
                period_frames.append({
                    'period': period_label,
                    'countries': top_country_names,
                    'matrix': adjacency_matrix,
                    'weights': dict(top_countries)
                })
            
            if len(period_frames) < 2:
                self.log_warning("Insufficient data across time periods for animated chord diagram")
                return None
            
            # Create figure with frames for animation
            fig = go.Figure()
            
            # Add initial frame (first period)
            initial_frame = period_frames[0]
            n = len(initial_frame['countries'])
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            radius = 1.0
            x_positions = radius * np.cos(angles)
            y_positions = radius * np.sin(angles)
            
            # Colors for countries
            colors = []
            for i in range(n):
                hue = i / n
                rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                color_hex = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
                colors.append(color_hex)
            
            # Add node traces (will be updated in frames)
            fig.add_trace(go.Scatter(
                x=x_positions,
                y=y_positions,
                mode='markers+text',
                marker=dict(size=25, color=colors, line=dict(color='black', width=1.5)),
                text=initial_frame['countries'],
                textposition='middle center',
                textfont=dict(size=10),
                name='Countries',
                hoverinfo='text',
                hovertext=[f"<b>{c}</b><br>Total weight: {initial_frame['weights'][c]:.1f}" for c in initial_frame['countries']]
            ))
            
            # Add chord traces (will be updated in frames)
            max_weight = max(initial_frame['matrix'].flatten()) if initial_frame['matrix'].size > 0 else 1
            
            for i in range(n):
                for j in range(i+1, n):
                    weight = initial_frame['matrix'][i, j]
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
                        
                        line_width = 3 + 15 * weight / max_weight
                        
                        fig.add_trace(go.Scatter(
                            x=curve_x,
                            y=curve_y,
                            mode='lines',
                            line=dict(width=line_width, color='gray'),
                            hoverinfo='none',
                            showlegend=False
                        ))
            
            # Create frames for each period
            frames = []
            for frame_data in period_frames[1:]:
                frame_n = len(frame_data['countries'])
                frame_angles = np.linspace(0, 2 * np.pi, frame_n, endpoint=False)
                frame_x = radius * np.cos(frame_angles)
                frame_y = radius * np.sin(frame_angles)
                
                frame_colors = []
                for i in range(frame_n):
                    hue = i / frame_n
                    rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                    color_hex = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
                    frame_colors.append(color_hex)
                
                frame_max_weight = max(frame_data['matrix'].flatten()) if frame_data['matrix'].size > 0 else 1
                
                frame_traces = []
                
                # Node trace for this frame
                frame_traces.append(go.Scatter(
                    x=frame_x,
                    y=frame_y,
                    mode='markers+text',
                    marker=dict(size=25, color=frame_colors, line=dict(color='black', width=1.5)),
                    text=frame_data['countries'],
                    textposition='middle center',
                    textfont=dict(size=10),
                    hovertext=[f"<b>{c}</b><br>Total weight: {frame_data['weights'][c]:.1f}" for c in frame_data['countries']]
                ))
                
                # Chord traces for this frame
                for i in range(frame_n):
                    for j in range(i+1, frame_n):
                        weight = frame_data['matrix'][i, j]
                        if weight > 0:
                            t = np.linspace(0, 1, 100)
                            p0 = np.array([frame_x[i], frame_y[i]])
                            p3 = np.array([frame_x[j], frame_y[j]])
                            
                            mid_angle = (frame_angles[i] + frame_angles[j]) / 2
                            if abs(frame_angles[i] - frame_angles[j]) > np.pi:
                                mid_angle += np.pi
                            control_offset = 0.4 * (1 + weight / frame_max_weight)
                            ctrl_point = np.array([control_offset * np.cos(mid_angle), control_offset * np.sin(mid_angle)])
                            
                            curve_x = (1-t)**3 * p0[0] + 3*(1-t)**2*t * ctrl_point[0] + 3*(1-t)*t**2 * ctrl_point[0] + t**3 * p3[0]
                            curve_y = (1-t)**3 * p0[1] + 3*(1-t)**2*t * ctrl_point[1] + 3*(1-t)*t**2 * ctrl_point[1] + t**3 * p3[1]
                            
                            line_width = 3 + 15 * weight / frame_max_weight
                            
                            frame_traces.append(go.Scatter(
                                x=curve_x,
                                y=curve_y,
                                mode='lines',
                                line=dict(width=line_width, color='gray'),
                                hoverinfo='none',
                                showlegend=False
                            ))
                
                frames.append(go.Frame(
                    data=frame_traces,
                    name=frame_data['period'],
                    layout=go.Layout(
                        title_text=f"Country Collaborations: {frame_data['period']}"
                    )
                ))
            
            fig.frames = frames
            
            # Add play/pause buttons
            fig.update_layout(
                title=dict(
                    text=f"Evolution of Country Collaborations Across Time Periods<br>Top {self.top_countries_chord} Countries",
                    font=dict(size=16, weight='bold')
                ),
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    buttons=[dict(label="Play",
                                method="animate",
                                args=[None, {"frame": {"duration": 2000, "redraw": True},
                                            "fromcurrent": True}]),
                             dict(label="Pause",
                                method="animate",
                                args=[[None], {"frame": {"duration": 0, "redraw": False},
                                              "mode": "immediate"}]),
                             dict(label="Reset",
                                method="animate",
                                args=[[frames[0].name] if frames else [None],
                                      {"frame": {"duration": 0, "redraw": True},
                                       "mode": "immediate"}])]
                )],
                width=1000,
                height=1000,
                xaxis=dict(visible=False, range=[-1.5, 1.5], scaleanchor="y", scaleratio=1),
                yaxis=dict(visible=False, range=[-1.5, 1.5]),
                plot_bgcolor='white',
                paper_bgcolor='white',
                hovermode='closest'
            )
            
            # Save data
            self.plot_data['27_animated_country_chord'] = {
                'periods': [f['period'] for f in period_frames],
                'countries_per_period': [len(f['countries']) for f in period_frames],
                'total_collaborations': [np.sum(f['matrix'])/2 for f in period_frames]
            }
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_27_animated_country_chord_periods: {str(e)}")
            return None
    
    # ==================== NEW PLOT 28: ANIMATED WORLD MAP ====================
    def plot_28_animated_world_map(self):
        """
        28. Animated world map showing research activity by country over time
        Bubble size represents attention (count), color represents citations
        """
        try:
            if 'countries_list' not in self.df_processed.columns or 'year' not in self.df_processed.columns:
                return None
            
            # Prepare country-year data
            country_year_data = []
            
            for idx, row in self.df_processed.iterrows():
                if isinstance(row['countries_list'], list) and len(row['countries_list']) > 0:
                    year = row.get('year', None)
                    if pd.isna(year):
                        continue
                    
                    for country in row['countries_list']:
                        country_code = self._get_country_code(country.strip().upper())
                        if country_code:
                            country_year_data.append({
                                'country': country.strip(),
                                'country_code': country_code,
                                'year': int(year),
                                'attention': row.get('count', 1),
                                'citations': row.get('max_citations', 0),
                                'annual_citations': row.get('max_annual_citations', 0)
                            })
            
            if len(country_year_data) == 0:
                self.log_warning("No country data available for world map")
                return None
            
            country_df = pd.DataFrame(country_year_data)
            
            # Aggregate by country and year
            aggregated = country_df.groupby(['country_code', 'country', 'year']).agg({
                'attention': 'sum',
                'citations': 'mean',
                'annual_citations': 'mean'
            }).reset_index()
            
            # Filter years
            min_year = int(aggregated['year'].min())
            max_year = int(aggregated['year'].max())
            years_range = range(min_year, max_year + 1)
            
            # Create animated choropleth map
            fig = px.choropleth(
                aggregated,
                locations='country_code',
                color='attention',
                hover_name='country',
                animation_frame='year',
                range_color=[0, aggregated['attention'].quantile(0.95)],
                color_continuous_scale='Viridis',
                title=f"Animated World Map: Research Activity by Country ({min_year}-{max_year})<br>Bubble color = Attention (sum of count)",
                labels={
                    'attention': 'Total Attention',
                    'country_code': 'Country',
                    'year': 'Year'
                },
                projection='natural earth'
            )
            
            # Add hover data
            fig.update_traces(
                hovertemplate='<b>%{hovertext}</b><br>' +
                              'Year: %{frame}<br>' +
                              'Attention: %{z:.0f}<br>' +
                              'Citations: %{customdata[0]:.1f}<extra></extra>',
                customdata=aggregated[['citations', 'annual_citations']].values
            )
            
            fig.update_layout(
                width=1200,
                height=800,
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    coastlinecolor='gray',
                    showland=True,
                    landcolor='lightgray',
                    showocean=True,
                    oceancolor='lightblue',
                    showcountries=True,
                    countrycolor='white'
                ),
                title_font=dict(size=16, weight='bold')
            )
            
            # Save data
            self.plot_data['28_animated_world_map'] = {
                'years': years_range,
                'countries_analyzed': aggregated['country'].nunique(),
                'total_attention_by_year': aggregated.groupby('year')['attention'].sum().to_dict()
            }
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_28_animated_world_map: {str(e)}")
            return None
    
    def _get_country_code(self, country_name):
        """Convert country name to ISO 3166-1 alpha-3 code for mapping"""
        # Simple mapping for common countries
        country_mapping = {
            'US': 'USA', 'USA': 'USA', 'UNITED STATES': 'USA',
            'GB': 'GBR', 'UK': 'GBR', 'UNITED KINGDOM': 'GBR',
            'CN': 'CHN', 'CHINA': 'CHN', 'CHN': 'CHN',
            'DE': 'DEU', 'GERMANY': 'DEU', 'GERMAN': 'DEU',
            'FR': 'FRA', 'FRANCE': 'FRA',
            'JP': 'JPN', 'JAPAN': 'JPN',
            'IN': 'IND', 'INDIA': 'IND',
            'CA': 'CAN', 'CANADA': 'CAN',
            'AU': 'AUS', 'AUSTRALIA': 'AUS',
            'BR': 'BRA', 'BRAZIL': 'BRA',
            'RU': 'RUS', 'RUSSIA': 'RUS',
            'KR': 'KOR', 'SOUTH KOREA': 'KOR', 'KOREA': 'KOR',
            'IT': 'ITA', 'ITALY': 'ITA',
            'ES': 'ESP', 'SPAIN': 'ESP',
            'NL': 'NLD', 'NETHERLANDS': 'NLD',
            'SE': 'SWE', 'SWEDEN': 'SWE',
            'CH': 'CHE', 'SWITZERLAND': 'CHE',
            'IL': 'ISR', 'ISRAEL': 'ISR'
        }
        
        country_upper = country_name.upper().strip()
        return country_mapping.get(country_upper, country_upper[:3] if len(country_upper) >= 3 else None)
    
    # ==================== NEW PLOT 29: ANIMATED TOP 10 THEMES ====================
    def plot_29_animated_top_10_themes(self):
        """
        29. Animated bar chart race showing top 10 themes (Topic/Subfield/Field)
        Evolution of theme popularity over time
        """
        try:
            # Try different hierarchy levels in order of specificity
            hierarchy_levels = ['Topic', 'Subfield', 'Field']
            selected_level = None
            term_df = None
            
            for level in hierarchy_levels:
                if level in self.df_processed.columns:
                    temp_df = self.df_processed[[level, 'year', 'count']].copy()
                    temp_df = temp_df.dropna(subset=[level, 'year'])
                    if len(temp_df) > 10:
                        selected_level = level
                        term_df = temp_df
                        break
            
            if selected_level is None or term_df is None:
                self.log_warning("No suitable hierarchy level found for animated top themes")
                return None
            
            term_df.rename(columns={selected_level: 'theme'}, inplace=True)
            
            # Aggregate by year and theme
            theme_year_data = term_df.groupby(['year', 'theme']).agg({
                'count': 'sum'
            }).reset_index()
            
            # Get top themes overall
            total_by_theme = theme_year_data.groupby('theme')['count'].sum().sort_values(ascending=False)
            top_themes = total_by_theme.head(15).index.tolist()
            
            # Filter for top themes
            theme_year_data = theme_year_data[theme_year_data['theme'].isin(top_themes)]
            
            # Prepare data for animated bar chart
            years = sorted(theme_year_data['year'].unique())
            
            # Create cumulative or per-year data
            frames_data = []
            for year in years:
                year_data = theme_year_data[theme_year_data['year'] <= year].groupby('theme')['count'].sum().reset_index()
                year_data['year'] = year
                year_data = year_data.sort_values('count', ascending=False).head(10)
                frames_data.append(year_data)
            
            combined_data = pd.concat(frames_data, ignore_index=True)
            
            # Create animated bar chart
            fig = px.bar(
                combined_data,
                x='count',
                y='theme',
                orientation='h',
                animation_frame='year',
                color='count',
                color_continuous_scale='Viridis',
                range_x=[0, combined_data['count'].max() * 1.1],
                title=f"Top 10 {selected_level}s Evolution Over Time<br>Bar length = Cumulative Attention (sum of count)",
                labels={
                    'count': 'Total Attention (cumulative)',
                    'theme': selected_level,
                    'year': 'Year'
                }
            )
            
            fig.update_layout(
                width=1000,
                height=700,
                xaxis=dict(title="Total Attention", gridcolor='lightgray'),
                yaxis=dict(title=selected_level, categoryorder='total ascending'),
                hovermode='closest'
            )
            
            # Improve animation
            fig.update_traces(
                texttemplate='%{x:.0f}',
                textposition='outside',
                cliponaxis=False,
                marker=dict(line=dict(width=1, color='black'))
            )
            
            # Save data
            self.plot_data['29_animated_top_themes'] = {
                'hierarchy_level': selected_level,
                'top_themes': top_themes[:10],
                'years': years,
                'data': combined_data.to_dict('records')
            }
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_29_animated_top_10_themes: {str(e)}")
            return None
    
    # ==================== ORIGINAL PLOT 29 RENAMED TO PLOT 30 (KEEPING ALL ORIGINAL) ====================
    # Original plot_29_animated_bubble_chart renamed to plot_30_animated_bubble_chart
    def plot_30_animated_bubble_chart(self):
        """30. Animated bubble chart over years: count vs max_citations (originally plot 29)"""
        try:
            required_cols = ['year', 'count', 'max_citations', 'author count', 'num_countries']
            if not all(col in self.df_processed.columns for col in required_cols):
                return None
            
            valid_data = self.df_processed.dropna(subset=required_cols)
            valid_data = valid_data[valid_data['max_citations'] > 0]
            valid_data = valid_data[valid_data['year'].notna()]
            
            if len(valid_data) < 10:
                return None
            
            # Convert year to int
            valid_data['year'] = valid_data['year'].astype(int)
            
            # Create animated chart
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
            
            # Save data
            self.plot_data['30_animated_bubble_chart'] = valid_data[required_cols].to_dict('records')
            
            return fig
            
        except Exception as e:
            self.log_error(f"Error in plot_30_animated_bubble_chart: {str(e)}")
            return None
    
    # ==================== ORIGINAL PLOT 30 RENAMED TO PLOT 31 ====================
    def plot_31_topic_chord_diagram(self):
        """31. Chord diagram for topic co-occurrence (originally plot 30)"""
        try:
            if 'Topic' not in self.df_processed.columns:
                return None
            
            # Collect topic co-occurrence data
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
            
            # Select top N topics
            top_topics = sorted(topic_weights.items(), key=lambda x: x[1], reverse=True)[:15]
            top_topic_names = [t[0] for t in top_topics]
            
            # Build connections through shared Subfield
            if 'Subfield' in self.df_processed.columns:
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
                
                # Create color scheme
                colors = []
                for i in range(n):
                    hue = i / n
                    rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                    color_hex = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
                    colors.append(color_hex)
                
                # Arrange nodes in a circle
                angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
                radius = 1.0
                x_positions = radius * np.cos(angles)
                y_positions = radius * np.sin(angles)
                
                # Text positions outside
                text_radius = 1.25
                text_x = text_radius * np.cos(angles)
                text_y = text_radius * np.sin(angles)
                
                # Create figure
                fig = go.Figure()
                
                # Add external labels
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
                
                # Add nodes
                fig.add_trace(go.Scatter(
                    x=x_positions,
                    y=y_positions,
                    mode='markers',
                    marker=dict(size=20, color=colors, line=dict(color='black', width=1.5)),
                    hovertext=[f"<b>{name}</b><br>Total weight: {weight:.1f}" for name, weight in top_topics],
                    hoverinfo='text',
                    showlegend=False
                ))
                
                # Add chords
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
                
                # Add inner circle
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
                
                # Save data
                self.plot_data['31_topic_chord_diagram'] = {
                    'topics': top_topic_names,
                    'adjacency_matrix': adjacency_matrix.tolist()
                }
                
                return fig
            else:
                self.log_warning("Subfield column not found for topic chord diagram")
                return None
            
        except Exception as e:
            self.log_error(f"Error in plot_31_topic_chord_diagram: {str(e)}")
            return None
    
    def generate_all_plots(self, selected_plots=None):
        """Generate all plots with progress bar (31 plots total after changes)"""
        self.all_figures = {}
        self.plot_data = {}
        self.errors = []
        self.warnings = []
        
        # Updated list of all plot functions (31 plots after adding new ones)
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
            ("25_term_temporal_density", "25. Term Temporal Density (Violin Plot)", lambda: self.plot_25_term_temporal_density(
                hierarchy_level=st.session_state.get('term_hierarchy_level', 'Topic'),
                top_percent=st.session_state.get('term_top_percent', 15),
                metric=st.session_state.get('term_sort_metric', 'total_attention')
            )),
            ("25b_active_terms_5years", "25b. Active Terms Last 5 Years", lambda: self.plot_25b_active_terms_last_5_years(
                hierarchy_level=st.session_state.get('term_hierarchy_level', 'Topic'),
                top_n=st.session_state.get('term_top_n_5years', 20)
            )),
            ("26_animated_journal_heatmap", "26. Animated Journal Heatmap", lambda: self.plot_26_animated_journal_heatmap(15)),
            ("27_animated_country_chord", "27. Animated Country Chord (Periods)", lambda: self.plot_27_animated_country_chord_periods(4)),
            ("28_animated_world_map", "28. Animated World Map", self.plot_28_animated_world_map),
            ("29_animated_top_themes", "29. Animated Top 10 Themes", self.plot_29_animated_top_10_themes),
            ("30_animated_bubble", "30. Animated Bubble Chart", self.plot_30_animated_bubble_chart),
            ("31_topic_chord", "31. Topic Chord Diagram", self.plot_31_topic_chord_diagram)
        ]
        
        # Filter if specific plots selected
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
            
            # 4. Terminology sheet
            self._add_terminology_sheet(writer)
        
        excel_buffer.seek(0)
        return excel_buffer
    
    def _add_terminology_sheet(self, writer):
        """Add sheet with terminology and formula explanations"""
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
        """Save all plots and reports to ZIP archive"""
        if not self.all_figures:
            st.error("❌ No plots to save!")
            return None
        
        # Create ZIP archive
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 1. Save plots
            for i, (name, fig) in enumerate(self.all_figures.items()):
                # Check if plot should be exported as HTML (for animated plots)
                animated_plot_names = ['26_animated_journal_heatmap', '27_animated_country_chord', 
                                      '28_animated_world_map', '29_animated_top_themes', '30_animated_bubble']
                
                if name in animated_plot_names and hasattr(fig, 'update_layout'):
                    # Save as HTML for animated plots
                    html_str = pio.to_html(fig, include_plotlyjs='cdn', full_html=True)
                    filename = f"plot_{i+1:02d}_{name}.html"
                    zip_file.writestr(filename, html_str)
                else:
                    # Save as PNG for static plots
                    img_buffer = io.BytesIO()
                    if hasattr(fig, 'update_layout'):
                        # Plotly figure
                        pio.write_image(fig, img_buffer, format='png', width=1200, height=800, scale=2)
                    else:
                        # Matplotlib figure
                        fig.savefig(img_buffer, format='png', dpi=300,
                                  bbox_inches='tight', facecolor='white',
                                  edgecolor='black')
                    img_buffer.seek(0)
                    filename = f"plot_{i+1:02d}_{name}.png"
                    zip_file.writestr(filename, img_buffer.read())
                    if not hasattr(fig, 'update_layout'):
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
                'visualization_settings': {
                    'show_regression_trends': self.show_regression_trends,
                    'top_countries_chord': self.top_countries_chord,
                    'top_fields_sankey': self.top_fields_sankey,
                    'analysis_year': self.analysis_year,
                    'years_lookback': self.years_lookback
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
# STREAMLIT INTERFACE FUNCTIONS
# ============================================================================

def main():
    """Main Streamlit application function"""
    
    # Title
    st.title("📊 Scientific Data Visualization Dashboard")
    st.markdown("---")
    
    # Define ALL_PLOTS at the beginning of the function
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
        ("25_term_temporal_density", "25. Term Temporal Density (Violin Plot)"),
        ("25b_active_terms_5years", "25b. Active Terms Last 5 Years"),
        ("26_animated_journal_heatmap", "26. Animated Journal Heatmap"),
        ("27_animated_country_chord", "27. Animated Country Chord (Periods)"),
        ("28_animated_world_map", "28. Animated World Map"),
        ("29_animated_top_themes", "29. Animated Top 10 Themes"),
        ("30_animated_bubble", "30. Animated Bubble Chart"),
        ("31_topic_chord", "31. Topic Chord Diagram")
    ]
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ScientificDataAnalyzer()
    
    if 'plots_generated' not in st.session_state:
        st.session_state.plots_generated = False
    
    if 'selected_plots' not in st.session_state:
        st.session_state.selected_plots = [plot[0] for plot in ALL_PLOTS]
    
    # Term analysis settings
    if 'term_hierarchy_level' not in st.session_state:
        st.session_state.term_hierarchy_level = 'Topic'
    if 'term_top_percent' not in st.session_state:
        st.session_state.term_top_percent = 15
    if 'term_sort_metric' not in st.session_state:
        st.session_state.term_sort_metric = 'total_attention'
    if 'term_top_n_5years' not in st.session_state:
        st.session_state.term_top_n_5years = 20
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Navigation menu
        selected = option_menu(
            menu_title="Menu",
            options=["📋 Data Loading", "📊 Visualization", "📥 Download"],
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
        
        # Visualization settings (applied in real-time during generation)
        st.subheader("🎨 Visualization Settings")
        
        show_regression = st.checkbox(
            "📈 Show regression trends",
            value=st.session_state.analyzer.show_regression_trends if hasattr(st.session_state.analyzer, 'show_regression_trends') else True,
            help="Display regression lines on applicable plots"
        )
        
        top_countries = st.slider(
            "🌍 Countries in chord diagram",
            min_value=10,
            max_value=50,
            value=st.session_state.analyzer.top_countries_chord,
            step=5,
            help="Select number of top countries to display in collaboration chord diagram"
        )
        
        top_fields = st.slider(
            "📚 Fields count in Sankey diagram",
            min_value=5,
            max_value=20,
            value=st.session_state.analyzer.top_fields_sankey,
            step=1,
            help="Limit number of fields for Sankey diagram readability"
        )
        
        st.markdown("---")
        
        # Term analysis specific settings
        st.subheader("🏷️ Term Analysis Settings")
        
        st.session_state.term_hierarchy_level = st.selectbox(
            "Hierarchy level for term analysis",
            options=['Topic', 'Subfield', 'Field', 'Domain', 'Concepts'],
            index=['Topic', 'Subfield', 'Field', 'Domain', 'Concepts'].index(st.session_state.term_hierarchy_level)
        )
        
        st.session_state.term_top_percent = st.slider(
            "Top % of terms to display (Violin plot)",
            min_value=5,
            max_value=30,
            value=st.session_state.term_top_percent,
            step=5,
            help="Show only top X% of terms by selected metric"
        )
        
        st.session_state.term_sort_metric = st.selectbox(
            "Sort by metric",
            options=['total_attention', 'activity_span', 'peak_density'],
            format_func=lambda x: x.replace('_', ' ').title(),
            index=['total_attention', 'activity_span', 'peak_density'].index(st.session_state.term_sort_metric)
        )
        
        st.session_state.term_top_n_5years = st.slider(
            "Top N terms for 5-year activity",
            min_value=10,
            max_value=30,
            value=st.session_state.term_top_n_5years,
            step=5,
            help="Number of top active terms to display in 5-year analysis"
        )
        
        # Analysis year settings
        current_year = datetime.now().year
        st.session_state.analyzer.analysis_year = st.number_input(
            "Analysis reference year",
            min_value=2000,
            max_value=current_year,
            value=current_year,
            step=1,
            help="Year used as reference for 'Last 5 years' calculations"
        )
        
        st.session_state.analyzer.years_lookback = st.slider(
            "Lookback period (years)",
            min_value=2,
            max_value=10,
            value=5,
            step=1,
            help="Number of years to look back for recent activity analysis"
        )
        
        # Apply settings to analyzer
        st.session_state.analyzer.update_visualization_settings(
            show_regression_trends=show_regression,
            top_countries_chord=top_countries,
            top_fields_sankey=top_fields,
            analysis_year=st.session_state.analyzer.analysis_year,
            years_lookback=st.session_state.analyzer.years_lookback
        )
        
        st.markdown("---")
        
        # Sample data
        st.subheader("📚 Sample Data")
        sample_data = """doi	publication_date	Title	authors	ORCID ID 1; ORCID ID 2... ORCID ID last	author count	affiliations {aff 1; aff 2... aff last}	countries {country 1; ... country last}	Full journal Name	year	Volume	Pages (or article number)	Citation counts (CR)	Citation counts (OA)	Annual cit counts (CR)	Annual cit counts (OA)	references_count	count	Topic	Subfield	Field	Domain	Concepts
10.1021/acs.chemrev.6b00284	2016-11-09	Strategies for Carbon and Sulfur Tolerant Solid Oxide Fuel Cell Materials, Incorporating Lessons from Heterogeneous Catalysis	Paul Boldrin; Enrique Ruiz-Trejo; Joshua Mermelstein; José Miguel Bermúdez Menéndez; Tomás Ramı́rez Reina; Nigel P. Brandon	https://orcid.org/0000-0003-0058-6876; https://orcid.org/0000-0001-5560-5750; https://orcid.org/0000-0001-7211-2958; https://orcid.org/0000-0001-9693-5107; https://orcid.org/0000-0003-2230-8666	6	University of Surrey; Imperial College London; Boeing (United States)	US; GB	Chemical Reviews	2016	116	13633-13684	289	296	26.27	26.91	465	5	Advancements in Solid Oxide Fuel Cells	Chemistry	Carbon fibers	Catalysis	Sulfur; Chemistry; Carbon fibers; Catalysis; Oxide; Solid oxide fuel cell; Fuel cells; Nanotechnology; Environmental chemistry; Chemical engineering; Organic chemistry; Materials science; Engineering; Composite number; Physical chemistry; Composite material; Anode; Electrode
10.1126/science.aab3987	2015-07-23	Readily processed protonic ceramic fuel cells with high performance at low temperatures	Chuancheng Duan; Jianhua Tong; Meng Shang; Stefan Nikodemski; Michael Sanders; Sandrine Ricote; Ali Almansoori; Ryan O'Hayre	https://orcid.org/0000-0002-1826-1415; https://orcid.org/0000-0002-0684-1658; https://orcid.org/0000-0001-6366-5219; https://orcid.org/0000-0001-7565-0284; https://orcid.org/0000-0002-0789-5105; https://orcid.org/0000-0003-3762-3052	8	American Petroleum Institute; Colorado School of Mines	US	Science	2015	349	1321-1326	1325	1352	110.42	112.67	91	5	Advancements in Solid Oxide Fuel Cells	Oxide	Fuel cells	Materials science	Ceramic; Oxide; Fuel cells; Materials science; Methane; Electrolyte; Chemical engineering; Cathode; Ion; Solid oxide fuel cell; Chemistry; Composite material; Electrode; Metallurgy; Organic chemistry; Engineering; Physical chemistry"""
        
        if st.button("📋 Load sample data", use_container_width=True):
            st.session_state.sample_data_loaded = sample_data
            st.rerun()
        
        st.markdown("---")
        st.info("""
        **Instructions:**
        1. Paste TSV formatted data
        2. Click 'Load Data'
        3. Select plots to generate
        4. Download results
        """)
    
    # Main content
    if selected == "📋 Data Loading":
        st.header("📋 Data Loading")
        
        # Data input field
        data_input = st.text_area(
            "Paste TSV formatted data (tab-separated columns)",
            value=st.session_state.get('sample_data_loaded', ''),
            height=300,
            help="Copy and paste data from Excel/Google Sheets. First row must contain column headers."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Load Data", type="primary", use_container_width=True):
                if data_input.strip():
                    with st.spinner("Processing data..."):
                        st.session_state.analyzer.parse_data(data_input)
                        st.success("✅ Data successfully loaded!")
                        st.session_state.plots_generated = False
                else:
                    st.error("❌ Please enter data")
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.sample_data_loaded = ''
                st.rerun()
        
        # Show data information
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
                    st.metric("Total mentions", f"{total_mentions:,}")
            with col4:
                if 'max_citations' in st.session_state.analyzer.df_processed.columns:
                    mean_max_cit = st.session_state.analyzer.df_processed['max_citations'].mean()
                    st.metric("Mean max citations", f"{mean_max_cit:.1f}")
            
            # Show table
            with st.expander("👁️ View data"):
                st.dataframe(st.session_state.analyzer.df_processed.head(10))
    
    elif selected == "📊 Visualization":
        st.header("📊 Data Visualization")
        
        if st.session_state.analyzer.df_processed is None:
            st.warning("⚠️ Please load data first in the 'Data Loading' section")
            return
        
        # Plot selection
        st.subheader("🎯 Select plots to generate")
        
        # Select all / clear all buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Select all", use_container_width=True):
                st.session_state.selected_plots = [plot[0] for plot in ALL_PLOTS]
                st.rerun()
        with col2:
            if st.button("❌ Clear selection", use_container_width=True):
                st.session_state.selected_plots = []
                st.rerun()
        
        st.markdown("### Available Plots")
        
        # Group plots by category
        categories = {
            "📈 Basic Distributions": ["1_distribution", "19_cumulative_influence"],
            "🌍 International Collaboration": ["2_country_chord", "3_internationality_linear", "4_internationality_log", "6_collab_linear", "7_collab_log"],
            "📚 Journals and Publications": ["5_journal_heatmap", "21_journal_impact"],
            "🔗 References and Citations": ["8_references_linear", "9_references_log", "17_cr_vs_oa", "20_references_attention"],
            "🏷️ Concepts and Topics": ["10_concepts", "11_concept_cooccurrence", "12_concept_influence", "24_concept_network", "31_topic_chord"],
            "⏰ Temporal Analysis": ["13_temporal_evolution", "14_temporal_heatmap"],
            "🗓️ Term Temporal Activity (NEW)": ["25_term_temporal_density", "25b_active_terms_5years"],
            "👥 Teams and Organizations": ["15_team_size"],
            "📊 Metrics Analysis": ["16_correlation", "18_domain_citations", "23_mds"],
            "🏛️ Hierarchical Structure": ["22_hierarchical_sankey"],
            "🎬 Animated Visualizations (NEW)": ["26_animated_journal_heatmap", "27_animated_country_chord", "28_animated_world_map", "29_animated_top_themes", "30_animated_bubble"],
            "🎯 Multi-dimensional": []
        }
        
        for category, plot_ids in categories.items():
            if plot_ids:  # Only show categories with plots
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
        
        # Generation buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Generate selected plots", type="primary", use_container_width=True):
                if not st.session_state.selected_plots:
                    st.error("❌ Please select at least one plot")
                else:
                    with st.spinner("Generating plots..."):
                        st.session_state.analyzer.generate_all_plots(st.session_state.selected_plots)
                        st.session_state.plots_generated = True
                        st.success(f"✅ Generated {len(st.session_state.analyzer.all_figures)} plots!")
        
        with col2:
            if st.button("🎯 Generate all plots", use_container_width=True):
                st.session_state.selected_plots = [plot[0] for plot in ALL_PLOTS]
                with st.spinner("Generating all plots..."):
                    st.session_state.analyzer.generate_all_plots()
                    st.session_state.plots_generated = True
                    st.success(f"✅ Generated {len(st.session_state.analyzer.all_figures)} plots!")
        
        # Display generated plots
        if st.session_state.plots_generated and st.session_state.analyzer.all_figures:
            st.markdown("---")
            st.subheader("📈 Visualization Results")
            
            # Navigation through plots
            plot_names = list(st.session_state.analyzer.all_figures.keys())
            
            if len(plot_names) > 0:
                # Selector for choosing plot
                selected_plot = st.selectbox(
                    "Select plot to view",
                    options=plot_names,
                    format_func=lambda x: next(name for pid, name in ALL_PLOTS if pid == x)
                )
                
                # Display selected plot
                if selected_plot in st.session_state.analyzer.all_figures:
                    fig = st.session_state.analyzer.all_figures[selected_plot]
                    
                    # Check plot type (plotly or matplotlib)
                    if hasattr(fig, 'update_layout'):
                        # This is a plotly figure
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add HTML download button for animated plots
                        animated_plots = ['26_animated_journal_heatmap', '27_animated_country_chord', 
                                        '28_animated_world_map', '29_animated_top_themes', '30_animated_bubble']
                        if selected_plot in animated_plots:
                            html_str = pio.to_html(fig, include_plotlyjs='cdn', full_html=True)
                            st.download_button(
                                label="📥 Download as HTML (Interactive Animation)",
                                data=html_str,
                                file_name=f"{selected_plot}_animation.html",
                                mime="text/html",
                                use_container_width=True
                            )
                    else:
                        # This is a matplotlib figure
                        st.pyplot(fig)
                    
                    # Plot information
                    plot_name = next(name for pid, name in ALL_PLOTS if pid == selected_plot)
                    st.info(f"**{plot_name}**")
                    
                    # Navigation buttons
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
    
    elif selected == "📥 Download":
        st.header("📥 Download Results")
        
        if not st.session_state.plots_generated or not st.session_state.analyzer.all_figures:
            st.warning("⚠️ Please generate plots first in the 'Visualization' section")
            return
        
        st.success(f"✅ Available for download: {len(st.session_state.analyzer.all_figures)} plots")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download individual plots
            st.subheader("📸 Individual Plots")
            
            plot_options = {}
            for pid in st.session_state.analyzer.all_figures.keys():
                for plot_id, name in ALL_PLOTS:
                    if plot_id == pid:
                        plot_options[name] = pid
                        break
            
            selected_plot_name = st.selectbox("Select plot", options=list(plot_options.keys()))
            
            if selected_plot_name:
                plot_id = plot_options[selected_plot_name]
                fig = st.session_state.analyzer.all_figures[plot_id]
                
                # Save plot to buffer
                animated_plots = ['26_animated_journal_heatmap', '27_animated_country_chord', 
                                '28_animated_world_map', '29_animated_top_themes', '30_animated_bubble']
                
                if plot_id in animated_plots and hasattr(fig, 'update_layout'):
                    # Save as HTML for animated plots
                    html_str = pio.to_html(fig, include_plotlyjs='cdn', full_html=True)
                    st.download_button(
                        label=f"📥 Download {selected_plot_name} (HTML)",
                        data=html_str,
                        file_name=f"plot_{selected_plot_name[:20].replace(' ', '_')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                else:
                    # Save as PNG for static plots
                    img_buffer = io.BytesIO()
                    if hasattr(fig, 'update_layout'):
                        # Plotly figure
                        pio.write_image(fig, img_buffer, format='png', width=1200, height=800, scale=2)
                    else:
                        # Matplotlib figure
                        fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                    img_buffer.seek(0)
                    st.download_button(
                        label=f"📥 Download {selected_plot_name} (PNG)",
                        data=img_buffer,
                        file_name=f"plot_{selected_plot_name[:20].replace(' ', '_')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
        
        with col2:
            # Download all in ZIP
            st.subheader("📦 All Results")
            
            if st.button("📥 Download ZIP archive", type="primary", use_container_width=True):
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
        
        # Statistics
        st.markdown("---")
        st.subheader("📊 Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total plots", len(st.session_state.analyzer.all_figures))
        with col2:
            st.metric("Errors", len(st.session_state.analyzer.errors))
        with col3:
            st.metric("Warnings", len(st.session_state.analyzer.warnings))
        
        # Show errors and warnings
        if st.session_state.analyzer.errors:
            with st.expander("❌ Errors"):
                for error in st.session_state.analyzer.errors:
                    st.error(f"{error['timestamp']}: {error['message']}")
        
        if st.session_state.analyzer.warnings:
            with st.expander("⚠️ Warnings"):
                for warning in st.session_state.analyzer.warnings:
                    st.warning(f"{warning['timestamp']}: {warning['message']}")

# Run the application
if __name__ == "__main__":
    main()
