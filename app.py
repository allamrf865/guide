import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

# Title and Introduction
st.title("📊 Analisis Canggih Program Sovis - RCA dan Efisiensi Operasional")
st.markdown("""
### 📖 Evaluasi Efisiensi, RCA, dan Prediksi Kinerja
Dashboard ini mengintegrasikan analisis Root Cause Analysis (RCA), efisiensi operasional, dan prediksi kinerja anggota program. Dilengkapi dengan visualisasi canggih, analisis mendalam, dan prediksi berbasis Machine Learning.
""")

# Data Simulasi RCA
data_rca = {
    'CI': [1, 2, 3, 4, 5],
    'CV': [0.3, 0.4, 0.5, 0.2, 0.1],
    'IC': [1.5, 2.0, 1.8, 0.8, 1.2],
    'CP': 1.2,
    'PD': 0.9,
    'SD': 0.8,
    'ST': 0.85,
    'SI': 0.75,
    'MP': 0.7,
    'VF': 0.9
}

PI = 5  # Faktor Utama
time_decay = 0.1  # Lambda untuk eksponensial decay
time_periods = np.array([1, 2, 3, 4, 5])

# Rumus RCA
def calculate_rca_time(PI, CI, CV, IC, CP, PD, SD, ST, SI, MP, VF, t, decay):
    contributions = [
        (PI + np.log(1 + ci * cv) / np.sqrt(ic)) * np.exp(-decay * t[i])
        for i, (ci, cv, ic) in enumerate(zip(CI, CV, IC))
    ]
    RCA = sum(contributions) * CP * PD * SD * ST * SI * (MP + VF)
    return RCA, contributions

RCA_final_time, contributions_time = calculate_rca_time(
    PI, data_rca['CI'], data_rca['CV'], data_rca['IC'], data_rca['CP'], data_rca['PD'],
    data_rca['SD'], data_rca['ST'], data_rca['SI'], data_rca['MP'], data_rca['VF'],
    time_periods, time_decay
)

# Data Simulasi Bab
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

# Rumus Efisiensi Operasional Relatif (EOR)
alpha = 1.3
beta = 1.7
data_bab['Efisiensi Operasional Relatif (EOR)'] = (
    (data_bab['Realisasi (%)'] * data_bab['Efektivitas (%)'] * data_bab['Kedisiplinan (%)']) ** alpha
) / (
    (data_bab['Waktu Realisasi (Minggu)'] ** beta) * np.log(data_bab['Kompleksitas (%)'] + 1)
)

# Rumus Dampak Keterlambatan (DK)
data_bab['Dampak Keterlambatan (DK) (%)'] = np.abs(
    data_bab['Waktu Realisasi (Minggu)'] - data_bab['Waktu Target (Minggu)']
) / data_bab['Waktu Target (Minggu)'] * 100

# Index Kinerja Komposit (IKK)
data_bab['Index Kinerja Komposit (IKK)'] = (
    0.4 * data_bab['Realisasi (%)'] +
    0.3 * data_bab['Efektivitas (%)'] +
    0.2 * data_bab['Kepatuhan Prosedur (%)'] +
    0.1 * data_bab['Kontribusi (%)']
)

# Kontribusi Efisiensi (KE)
data_bab['Kontribusi Efisiensi (KE)'] = (
    data_bab['Efisiensi Operasional Relatif (EOR)'] * data_bab['Kontribusi (%)'] / 100
)

# Analisis PCA
features = ['Target (%)', 'Realisasi (%)', 'Efektivitas (%)', 'Kompleksitas (%)', 
            'Kepatuhan Prosedur (%)', 'Kedisiplinan (%)', 'Index Kinerja Komposit (IKK)']
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_bab[features])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)
data_bab['PCA1'] = pca_result[:, 0]
data_bab['PCA2'] = pca_result[:, 1]

# Visualisasi Kontribusi RCA
st.subheader("📌 Visualisasi Kontribusi RCA dan Efisiensi Operasional")
df_contributions = pd.DataFrame({
    'CI': data_rca['CI'],
    'Contribution': contributions_time,
    'Time Period': time_periods
})
fig_contributions = px.bar(
    df_contributions,
    x='CI',
    y='Contribution',
    title='Kontribusi Faktor CI terhadap RCA',
    labels={'Contribution': 'Kontribusi'},
    template='plotly_white'
)
st.plotly_chart(fig_contributions, use_container_width=True)

# Visualisasi RCA Over Time
fig_time = px.line(
    df_contributions,
    x='Time Period',
    y='Contribution',
    title='RCA Over Time',
    labels={'Contribution': 'RCA Score'},
    template='plotly_white'
)
st.plotly_chart(fig_time, use_container_width=True)

# Visualisasi PCA
fig2 = px.scatter(
    data_bab,
    x='PCA1', y='PCA2', color='Bab',
    size='Efisiensi Operasional Relatif (EOR)',
    hover_data=['Index Kinerja Komposit (IKK)', 'Dampak Keterlambatan (DK) (%)'],
    title='Analisis Dimensi Kinerja (PCA)'
)
st.plotly_chart(fig2, use_container_width=True)

# Kesimpulan
st.subheader("📋 Kesimpulan dan Interpretasi")
st.markdown(f"""
- **RCA Final (Time):** {RCA_final_time:.2f}  
- **Efisiensi Operasional Relatif (EOR)** tertinggi ditemukan pada Bab **{data_bab.iloc[data_bab['Efisiensi Operasional Relatif (EOR)'].idxmax()]['Bab']}**.  
- **Dampak Keterlambatan (DK)** tertinggi terjadi pada Bab **{data_bab.iloc[data_bab['Dampak Keterlambatan (DK) (%)'].idxmax()]['Bab']}**.  
""")
