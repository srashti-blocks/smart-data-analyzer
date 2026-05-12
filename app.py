from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os, io, base64, json, warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs('uploads', exist_ok=True)

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='#0d0d0d')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded

def profile_dataframe(df):
    profile = {
        'shape': {'rows': int(df.shape[0]), 'cols': int(df.shape[1])},
        'columns': [],
        'missing_total': int(df.isnull().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
    }
    for col in df.columns:
        col_info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'missing': int(df[col].isnull().sum()),
            'missing_pct': round(df[col].isnull().mean() * 100, 2),
            'unique': int(df[col].nunique()),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info['stats'] = {
                'mean': round(float(df[col].mean()), 4),
                'median': round(float(df[col].median()), 4),
                'std': round(float(df[col].std()), 4),
                'min': round(float(df[col].min()), 4),
                'max': round(float(df[col].max()), 4),
                'skew': round(float(df[col].skew()), 4),
            }
        profile['columns'].append(col_info)
    return profile

def generate_charts(df):
    charts = {}
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    plt.style.use('dark_background')
    accent = '#00f5d4'
    accent2 = '#f72585'

    # 1. Distribution charts for numeric cols (up to 4)
    if numeric_cols:
        cols_to_plot = numeric_cols[:4]
        fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(5 * len(cols_to_plot), 4))
        fig.patch.set_facecolor('#0d0d0d')
        if len(cols_to_plot) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols_to_plot):
            ax.set_facecolor('#1a1a1a')
            data = df[col].dropna()
            ax.hist(data, bins=30, color=accent, alpha=0.85, edgecolor='none')
            ax.set_title(col, color='white', fontsize=11, fontweight='bold')
            ax.tick_params(colors='#888')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333')
            mean_val = data.mean()
            ax.axvline(mean_val, color=accent2, linestyle='--', linewidth=1.5, label=f'mean={mean_val:.2f}')
            ax.legend(fontsize=8, labelcolor='white', facecolor='#222')
        fig.suptitle('Distributions', color='white', fontsize=14, y=1.02)
        plt.tight_layout()
        charts['distributions'] = fig_to_base64(fig)

    # 2. Correlation heatmap
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols)), max(5, len(numeric_cols) - 1)))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#0d0d0d')
        cmap = cm.get_cmap('RdYlGn')
        im = ax.imshow(corr.values, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right', color='white', fontsize=9)
        ax.set_yticklabels(corr.columns, color='white', fontsize=9)
        for i in range(len(corr)):
            for j in range(len(corr.columns)):
                val = corr.values[i, j]
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color='black' if abs(val) > 0.5 else 'white', fontsize=8)
        plt.colorbar(im, ax=ax)
        ax.set_title('Correlation Heatmap', color='white', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        charts['correlation'] = fig_to_base64(fig)

    # 3. Missing values bar
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#1a1a1a')
        bars = ax.barh(missing.index, missing.values, color=accent2, alpha=0.85)
        ax.set_xlabel('Missing Count', color='#aaa')
        ax.set_title('Missing Values per Column', color='white', fontsize=13, fontweight='bold')
        ax.tick_params(colors='#888')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
        for bar, val in zip(bars, missing.values):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(val), va='center', color='white', fontsize=9)
        plt.tight_layout()
        charts['missing'] = fig_to_base64(fig)

    # 4. KMeans clustering with PCA (if enough numeric data)
    if len(numeric_cols) >= 2 and df.shape[0] >= 10:
        try:
            X = df[numeric_cols].dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            n_clusters = min(4, len(X) // 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_facecolor('#0d0d0d')
            ax.set_facecolor('#1a1a1a')
            colors = ['#00f5d4', '#f72585', '#fee440', '#4cc9f0']
            for i in range(n_clusters):
                mask = labels == i
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           c=colors[i % len(colors)], label=f'Cluster {i+1}',
                           alpha=0.75, s=50, edgecolors='none')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', color='#aaa')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', color='#aaa')
            ax.set_title(f'K-Means Clustering (k={n_clusters}) via PCA', color='white', fontsize=13, fontweight='bold')
            ax.tick_params(colors='#555')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333')
            ax.legend(labelcolor='white', facecolor='#222', fontsize=9)
            plt.tight_layout()
            charts['clustering'] = fig_to_base64(fig)
        except Exception:
            pass

    return charts

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are supported'}), 400

    try:
        df = pd.read_csv(file)
        if df.empty:
            return jsonify({'error': 'CSV file is empty'}), 400

        profile = profile_dataframe(df)
        charts = generate_charts(df)

        # Top correlations
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        top_corr = []
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().abs()
            np.fill_diagonal(corr.values, 0)
            pairs = corr.unstack().sort_values(ascending=False)
            seen = set()
            for (c1, c2), val in pairs.items():
                key = tuple(sorted([c1, c2]))
                if key not in seen and val > 0:
                    seen.add(key)
                    top_corr.append({'col1': c1, 'col2': c2, 'value': round(float(val), 3)})
                if len(top_corr) >= 5:
                    break

        # Smart insights
        insights = []
        if profile['missing_total'] > 0:
            insights.append(f"⚠️ {profile['missing_total']} missing values detected across {sum(1 for c in profile['columns'] if c['missing'] > 0)} columns.")
        if profile['duplicate_rows'] > 0:
            insights.append(f"🔁 {profile['duplicate_rows']} duplicate rows found — consider cleaning before modeling.")
        if top_corr:
            best = top_corr[0]
            insights.append(f"🔗 Strongest correlation: '{best['col1']}' & '{best['col2']}' at {best['value']}.")
        for col in profile['columns']:
            if col.get('stats') and abs(col['stats']['skew']) > 1.5:
                insights.append(f"📊 '{col['name']}' is highly skewed (skew={col['stats']['skew']}) — may need log transform.")

        return jsonify({
            'profile': profile,
            'charts': charts,
            'top_correlations': top_corr,
            'insights': insights,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
