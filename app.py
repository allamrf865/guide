import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# Title and Subtitle
st.title("📊 Analisis Ketercapaian Program Sovis - Guidebook Pengobatan Massal")
st.markdown("""
### 📖 Evaluasi Efisiensi, Dampak, dan Proyeksi Program
Selamat datang di dashboard interaktif untuk memantau kemajuan dan performa Program Guidebook Pengobatan Massal. 
Gunakan visualisasi ini untuk mendapatkan wawasan mendalam!
""")

# Simulasi Data
data_bab = pd.DataFrame({
    'Bab': ['Pendahuluan', 'Pemeriksaan Dewasa', 'Manajemen Farmasi', 'Alur Pengobatan', 'Penutup'],
    'Target (%)': [100, 100, 100, 100, 100],
    'Realisasi (%)': [100, 95, 90, 80, 100],
    'Efektivitas (%)': [90, 85, 80, 75, 90],
    'Waktu Target (Minggu)': [2, 4, 3, 3, 2],
    'Waktu Realisasi (Minggu)': [2, 5, 4, 5, 2]
})

# Rumus Efisiensi Operasional Relatif (EOR)
alpha = 1.2
beta = 1.5
data_bab['Efisiensi Operasional Relatif (EOR)'] = (
    (data_bab['Realisasi (%)'] * data_bab['Efektivitas (%)']) ** alpha
) / (
    (data_bab['Waktu Realisasi (Minggu)'] ** beta) * np.log(data_bab['Target (%)'])
)

# Rumus Dampak Keterlambatan (DK)
data_bab['Dampak Keterlambatan (DK) (%)'] = np.abs(
    data_bab['Waktu Realisasi (Minggu)'] - data_bab['Waktu Target (Minggu)']
) / data_bab['Waktu Target (Minggu)'] * 100

# Proyeksi Waktu Penyelesaian dengan Polynomial Ridge Regression
poly = PolynomialFeatures(degree=3)
X = data_bab[['Target (%)', 'Realisasi (%)', 'Efektivitas (%)']]
y = data_bab['Waktu Realisasi (Minggu)']
X_poly = poly.fit_transform(X)
ridge = Ridge(alpha=0.1).fit(X_poly, y)

# Prediksi Waktu Penyelesaian untuk Bab Baru
X_new = pd.DataFrame({
    'Target (%)': [100, 100],
    'Realisasi (%)': [95, 90],
    'Efektivitas (%)': [85, 80]
})
X_new_poly = poly.transform(X_new)
predicted_times = ridge.predict(X_new_poly)

# Data Proyeksi
proyeksi_data = pd.DataFrame({
    'Bab': ['Bab 6', 'Bab 7'],
    'Prediksi Waktu (Minggu)': predicted_times
})

# UI - Multi-layer Chart
st.subheader("📌 Ketercapaian Program dan Efisiensi")
fig = go.Figure()

# Layer 1: Bar Chart (Realisasi dan EOR)
fig.add_trace(go.Bar(
    x=data_bab['Bab'],
    y=data_bab['Realisasi (%)'],
    name='Realisasi (%)',
    marker=dict(color='rgb(26, 118, 255)')
))
fig.add_trace(go.Bar(
    x=data_bab['Bab'],
    y=data_bab['Efisiensi Operasional Relatif (EOR)'],
    name='Efisiensi Operasional Relatif (EOR)',
    marker=dict(color='rgb(255, 127, 14)')
))

# Layer 2: Line Chart (Dampak Keterlambatan)
fig.add_trace(go.Scatter(
    x=data_bab['Bab'],
    y=data_bab['Dampak Keterlambatan (DK) (%)'],
    name='Dampak Keterlambatan (%)',
    mode='lines+markers',
    line=dict(color='rgb(214, 39, 40)', dash='dash')
))

# Layer 3: Proyeksi Waktu Penyelesaian
fig.add_trace(go.Scatter(
    x=proyeksi_data['Bab'],
    y=proyeksi_data['Prediksi Waktu (Minggu)'],
    name='Prediksi Waktu Penyelesaian (Minggu)',
    mode='markers',
    marker=dict(size=12, color='rgb(44, 160, 44)')
))

# Layout
fig.update_layout(
    title='📊 Analisis Ketercapaian dan Proyeksi Guidebook',
    xaxis_title='Bab',
    yaxis_title='Skor atau Waktu',
    legend_title='Indikator',
    barmode='group',
    template='plotly_white',
    title_font=dict(size=20, color='rgb(56, 78, 108)')
)
st.plotly_chart(fig, use_container_width=True)

# Kesimpulan
st.markdown("""
### 📌 Kesimpulan Utama
1. **Efisiensi Operasional Relatif (EOR):** Bab "Pendahuluan" memiliki efisiensi terbaik dengan skor **{:.2f}**.
2. **Dampak Keterlambatan:** Bab "Alur Pengobatan" menunjukkan dampak keterlambatan tertinggi sebesar **{:.2f}%**.
3. **Proyeksi Penyelesaian:** Bab 6 dan Bab 7 diprediksi masing-masing memerlukan waktu **{:.2f} minggu** dan **{:.2f} minggu** untuk penyelesaian.
""".format(
    data_bab['Efisiensi Operasional Relatif (EOR)'].max(),
    data_bab['Dampak Keterlambatan (DK) (%)'].max(),
    predicted_times[0],
    predicted_times[1]
))
