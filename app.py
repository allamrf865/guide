import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

# Title and Subtitle
st.title("📊 Analisis Canggih Program Sovis - Guidebook Pengobatan Massal")
st.markdown("""
### 📖 Evaluasi Efisiensi, Kinerja, dan Prediksi
Dashboard ini dirancang untuk memberikan analisis mendalam terkait performa anggota tim. Dengan data dan visualisasi interaktif, 
Anda dapat memahami efisiensi, kontribusi, dan proyeksi dari setiap bab dalam Guidebook.
""")

# Sidebar for Input and Navigation
st.sidebar.header("📂 Navigasi dan Input")
st.sidebar.markdown("Gunakan opsi berikut untuk mengatur analisis.")

# Data Simulasi
data_bab = pd.DataFrame({
    'Bab': ['Pendahuluan', 'Pemeriksaan Dewasa', 'Manajemen Farmasi', 'Alur Pengobatan', 'Penutup'],
    'Target (%)': [100, 100, 100, 100, 100],
    'Realisasi (%)': [100, 95, 90, 85, 100],
    'Efektivitas (%)': [90, 85, 80, 75, 90],
    'Waktu Target (Minggu)': [2, 4, 3, 3, 2],
    'Waktu Realisasi (Minggu)': [2, 5, 4, 5, 2],
    'Kompleksitas (%)': [70, 80, 85, 90, 75],
    'Kontribusi (%)': [20, 25, 30, 15, 10],
    'Kepatuhan Prosedur (%)': [95, 90, 85, 80, 100],
    'Kedisiplinan (%)': [98, 93, 85, 78, 96]
})

# Calculations for Advanced Metrics
alpha = 1.3
beta = 1.7
data_bab['Efisiensi Operasional Relatif (EOR)'] = (
    (data_bab['Realisasi (%)'] * data_bab['Efektivitas (%)'] * data_bab['Kedisiplinan (%)']) ** alpha
) / (
    (data_bab['Waktu Realisasi (Minggu)'] ** beta) * np.log(data_bab['Kompleksitas (%)'] + 1)
)

data_bab['Dampak Keterlambatan (DK) (%)'] = np.abs(
    data_bab['Waktu Realisasi (Minggu)'] - data_bab['Waktu Target (Minggu)']
) / data_bab['Waktu Target (Minggu)'] * 100

data_bab['Index Kinerja Komposit (IKK)'] = (
    0.4 * data_bab['Realisasi (%)'] +
    0.3 * data_bab['Efektivitas (%)'] +
    0.2 * data_bab['Kepatuhan Prosedur (%)'] +
    0.1 * data_bab['Kontribusi (%)']
)

# Ridge Regression Model for Prediction
features = ['Target (%)', 'Realisasi (%)', 'Efektivitas (%)', 'Kompleksitas (%)', 'Kedisiplinan (%)']
X = data_bab[features]
y = data_bab['Waktu Realisasi (Minggu)']

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
ridge = Ridge(alpha=0.1).fit(X_poly, y)

# Sidebar Input for Prediction
st.sidebar.subheader("🔮 Prediksi Bab Baru")
target = st.sidebar.slider("Target (%)", min_value=80, max_value=100, value=100, step=5)
realisasi = st.sidebar.slider("Realisasi (%)", min_value=70, max_value=100, value=90, step=5)
efektivitas = st.sidebar.slider("Efektivitas (%)", min_value=70, max_value=100, value=85, step=5)
kompleksitas = st.sidebar.slider("Kompleksitas (%)", min_value=60, max_value=100, value=80, step=5)
kedisiplinan = st.sidebar.slider("Kedisiplinan (%)", min_value=70, max_value=100, value=90, step=5)

X_new = pd.DataFrame({
    'Target (%)': [target],
    'Realisasi (%)': [realisasi],
    'Efektivitas (%)': [efektivitas],
    'Kompleksitas (%)': [kompleksitas],
    'Kedisiplinan (%)': [kedisiplinan]
})
X_new_poly = poly.transform(X_new)
predicted_time = ridge.predict(X_new_poly)

# Visualization
st.subheader("📊 Analisis Data dan Visualisasi")
fig1 = px.bar(
    data_bab,
    x='Bab',
    y=['Realisasi (%)', 'Efisiensi Operasional Relatif (EOR)', 'Index Kinerja Komposit (IKK)'],
    barmode='group',
    title="Perbandingan Realisasi, Efisiensi, dan Kinerja",
    labels={'value': 'Skor', 'Bab': 'Bab'}
)
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.scatter(
    data_bab,
    x='Efisiensi Operasional Relatif (EOR)',
    y='Index Kinerja Komposit (IKK)',
    size='Dampak Keterlambatan (DK) (%)',
    color='Bab',
    hover_data=['Waktu Realisasi (Minggu)', 'Kedisiplinan (%)'],
    title="Analisis Kinerja vs Efisiensi",
    labels={'Efisiensi Operasional Relatif (EOR)': 'EOR', 'Index Kinerja Komposit (IKK)': 'IKK'}
)
st.plotly_chart(fig2, use_container_width=True)

# Predicted Result Display
st.subheader("🔮 Hasil Prediksi Bab Baru")
st.markdown(f"""
- **Target (%)**: {target}  
- **Realisasi (%)**: {realisasi}  
- **Efektivitas (%)**: {efektivitas}  
- **Kompleksitas (%)**: {kompleksitas}  
- **Kedisiplinan (%)**: {kedisiplinan}  
- **Prediksi Waktu Penyelesaian**: **{predicted_time[0]:.2f} Minggu**  
""")

# Insights Section
st.subheader("📋 Insight dan Interpretasi")
max_efficiency = data_bab['Efisiensi Operasional Relatif (EOR)'].idxmax()
max_delay = data_bab['Dampak Keterlambatan (DK) (%)'].idxmax()
st.markdown(f"""
1. **Bab dengan Efisiensi Operasional Relatif (EOR) tertinggi**: **{data_bab.iloc[max_efficiency]['Bab']}** dengan skor **{data_bab.iloc[max_efficiency]['Efisiensi Operasional Relatif (EOR)']:.2f}**.
2. **Bab dengan Dampak Keterlambatan tertinggi**: **{data_bab.iloc[max_delay]['Bab']}** sebesar **{data_bab.iloc[max_delay]['Dampak Keterlambatan (DK) (%)']:.2f}%**.
3. Prediksi waktu penyelesaian untuk bab baru menunjukkan potensi alokasi waktu yang lebih realistis.
""")

# Downloadable Data
st.download_button(
    label="📥 Download Data Analisis",
    data=data_bab.to_csv(index=False),
    file_name="analisis_program.csv",
    mime="text/csv"
)
