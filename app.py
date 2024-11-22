import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# Title and Subtitle
st.title("📊 Analisis Program Sovis - Guidebook Pengobatan Massal")
st.markdown("""
### 📖 Evaluasi Efisiensi, Dampak, dan Proyeksi Program
Selamat datang di dashboard interaktif untuk memantau kemajuan dan performa Program Guidebook Pengobatan Massal. 
Gunakan visualisasi dan analisis interaktif ini untuk mendapatkan wawasan mendalam!
""")

# Sidebar untuk filter dan opsi
st.sidebar.header("📂 Filter dan Opsi Analisis")
selected_bab = st.sidebar.multiselect(
    "Pilih Bab untuk Analisis:",
    options=["Pendahuluan", "Pemeriksaan Dewasa", "Manajemen Farmasi", "Alur Pengobatan", "Penutup"],
    default=["Pendahuluan", "Pemeriksaan Dewasa", "Manajemen Farmasi", "Alur Pengobatan", "Penutup"]
)

# Data Simulasi
data_bab = pd.DataFrame({
    'Bab': ['Pendahuluan', 'Pemeriksaan Dewasa', 'Manajemen Farmasi', 'Alur Pengobatan', 'Penutup'],
    'Target (%)': [100, 100, 100, 100, 100],
    'Realisasi (%)': [100, 95, 90, 80, 100],
    'Efektivitas (%)': [90, 85, 80, 75, 90],
    'Waktu Target (Minggu)': [2, 4, 3, 3, 2],
    'Waktu Realisasi (Minggu)': [2, 5, 4, 5, 2]
})

# Filter berdasarkan Bab
filtered_data = data_bab[data_bab["Bab"].isin(selected_bab)]

# Perhitungan Efisiensi Operasional Relatif (EOR)
alpha = 1.2
beta = 1.5
filtered_data['Efisiensi Operasional Relatif (EOR)'] = (
    (filtered_data['Realisasi (%)'] * filtered_data['Efektivitas (%)']) ** alpha
) / (
    (filtered_data['Waktu Realisasi (Minggu)'] ** beta) * np.log(filtered_data['Target (%)'])
)

# Perhitungan Dampak Keterlambatan (DK)
filtered_data['Dampak Keterlambatan (DK) (%)'] = np.abs(
    filtered_data['Waktu Realisasi (Minggu)'] - filtered_data['Waktu Target (Minggu)']
) / filtered_data['Waktu Target (Minggu)'] * 100

# Model Ridge Regression untuk prediksi waktu penyelesaian
poly = PolynomialFeatures(degree=3)
X = filtered_data[['Target (%)', 'Realisasi (%)', 'Efektivitas (%)']]
y = filtered_data['Waktu Realisasi (Minggu)']
X_poly = poly.fit_transform(X)
ridge = Ridge(alpha=0.1).fit(X_poly, y)

# Input data untuk prediksi baru
st.sidebar.header("Prediksi Bab Baru")
target = st.sidebar.slider("Target (%)", min_value=80, max_value=100, value=100)
realisasi = st.sidebar.slider("Realisasi (%)", min_value=70, max_value=100, value=90)
efektivitas = st.sidebar.slider("Efektivitas (%)", min_value=70, max_value=100, value=85)

X_new = pd.DataFrame({
    'Target (%)': [target],
    'Realisasi (%)': [realisasi],
    'Efektivitas (%)': [efektivitas]
})
X_new_poly = poly.transform(X_new)
predicted_time = ridge.predict(X_new_poly)

# Visualisasi Data
st.subheader("📌 Analisis Data dan Proyeksi")
col1, col2 = st.columns(2)

with col1:
    fig1 = px.bar(
        filtered_data,
        x="Bab",
        y=["Realisasi (%)", "Efisiensi Operasional Relatif (EOR)"],
        barmode="group",
        labels={"value": "Skor", "Bab": "Bab"},
        title="Realisasi dan Efisiensi Operasional"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.line(
        filtered_data,
        x="Bab",
        y="Dampak Keterlambatan (DK) (%)",
        labels={"Dampak Keterlambatan (DK) (%)": "Dampak Keterlambatan (%)", "Bab": "Bab"},
        title="Dampak Keterlambatan per Bab",
        markers=True
    )
    st.plotly_chart(fig2, use_container_width=True)

# Proyeksi Waktu Penyelesaian
st.subheader("🔮 Prediksi Waktu Penyelesaian Bab Baru")
st.markdown(f"""
- **Target (%)**: {target}  
- **Realisasi (%)**: {realisasi}  
- **Efektivitas (%)**: {efektivitas}  
- **Prediksi Waktu Penyelesaian**: **{predicted_time[0]:.2f} Minggu**  
""")

# Ringkasan Interaktif
st.subheader("📋 Ringkasan")
max_efficiency = filtered_data['Efisiensi Operasional Relatif (EOR)'].idxmax()
max_delay = filtered_data['Dampak Keterlambatan (DK) (%)'].idxmax()
st.markdown(f"""
1. Bab dengan **Efisiensi Operasional Relatif (EOR)** tertinggi: **{filtered_data.iloc[max_efficiency]['Bab']}** 
   dengan skor **{filtered_data.iloc[max_efficiency]['Efisiensi Operasional Relatif (EOR)']:.2f}**.
2. Bab dengan **Dampak Keterlambatan (DK)** tertinggi: **{filtered_data.iloc[max_delay]['Bab']}** 
   sebesar **{filtered_data.iloc[max_delay]['Dampak Keterlambatan (DK) (%)']:.2f}%**.
""")

# Interaktif Tambahan: Tabel Data
st.subheader("📊 Data Lengkap")
st.dataframe(filtered_data)

# Export Data
st.download_button(
    label="📥 Download Data sebagai CSV",
    data=filtered_data.to_csv(index=False),
    file_name="analisis_program.csv",
    mime="text/csv"
)

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# Perbaikan Data Simulasi
data = {
    'CI': [1, 2, 3, 4, 5],  # Konversi CI menjadi nilai numerik
    'CV': [0.3, 0.4, 0.5, 0.2, 0.1],  # Contributing Value
    'IC': [1.5, 2.0, 1.8, 0.8, 1.2],  # Intensity Coefficient
    'CP': 1.2,  # Correction Priority
    'PD': 0.9,  # Probability of Damage
    'SD': 0.8,  # Solution Determinant
    'ST': 0.85, # Solution Threshold
    'SI': 0.75, # Solution Implementation
    'MP': 0.7,  # Mitigation Plan
    'VF': 0.9   # Verification Factor
}

# Faktor utama (PI)
PI = 5  # Contoh nilai faktor utama masalah

# Faktor waktu
time_decay = 0.1  # Lambda untuk eksponensial decay
time_periods = np.array([1, 2, 3, 4, 5])  # Simulasi waktu untuk setiap CI

# Rumus RCA dengan Waktu
def calculate_rca_time(PI, CI, CV, IC, CP, PD, SD, ST, SI, MP, VF, t, decay):
    contributions = [
        (PI + np.log(1 + ci * cv) / np.sqrt(ic)) * np.exp(-decay * t[i])
        for i, (ci, cv, ic) in enumerate(zip(CI, CV, IC))
    ]
    RCA = sum(contributions) * CP * PD * SD * ST * SI * (MP + VF)
    return RCA, contributions

RCA_final_time, contributions_time = calculate_rca_time(
    PI, data['CI'], data['CV'], data['IC'], data['CP'], data['PD'],
    data['SD'], data['ST'], data['SI'], data['MP'], data['VF'],
    time_periods, time_decay
)

# Analisis Solusi: Expected Impact Analysis
solutions = {'Solution 1': 2.0, 'Solution 2': 1.5}  # Penurunan RCA oleh solusi
costs = {'Solution 1': 3.0, 'Solution 2': 2.5}  # Biaya implementasi solusi
weights = {'Solution 1': 0.8, 'Solution 2': 0.6}  # Bobot prioritas solusi

EIA_results = {sol: (solutions[sol] / costs[sol]) * weights[sol] for sol in solutions}

# Validasi dengan PCA
variables = np.array([data['CV'], data['IC']]).T
scaler = StandardScaler()
variables_scaled = scaler.fit_transform(variables)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(variables_scaled)

explained_variance = pca.explained_variance_ratio_

# Dataframe untuk Kontribusi RCA dan Waktu
df_contributions = pd.DataFrame({
    'CI': data['CI'],
    'Contribution': contributions_time,
    'Time Period': time_periods
})

# Visualisasi: Bar Chart Kontribusi
fig_contributions = px.bar(
    df_contributions,
    x='CI',
    y='Contribution',
    title='Kontribusi Faktor CI terhadap RCA',
    labels={'Contribution': 'Kontribusi'},
    text='Contribution',
    template='plotly_white'
)
fig_contributions.update_traces(marker_color='rgb(58, 123, 189)', textposition='outside')

# Visualisasi: Heatmap Korelasi
correlation_matrix = pd.DataFrame(variables, columns=['CV', 'IC']).corr()
heatmap_fig = px.imshow(
    correlation_matrix,
    text_auto=True,
    color_continuous_scale='Blues',
    title='Korelasi Antar Faktor RCA'
)

# Visualisasi: PCA Komponen Dominan
fig_pca = go.Figure(data=[
    go.Bar(name='Komponen 1', x=['CV', 'IC'], y=pca.components_[0]),
    go.Bar(name='Komponen 2', x=['CV', 'IC'], y=pca.components_[1])
])
fig_pca.update_layout(barmode='group', title='PCA: Dominasi Variabel RCA')

# Visualisasi: RCA Periode Waktu
fig_time = px.line(
    df_contributions,
    x='Time Period',
    y='Contribution',
    title='RCA Over Time',
    labels={'Contribution': 'RCA Score'},
    template='plotly_white'
)

# Tampilkan Hasil
print(f"RCA Final (Time): {RCA_final_time:.2f}")
print("Explained Variance by PCA:")
print(f"Komponen 1: {explained_variance[0]:.2f}")
print(f"Komponen 2: {explained_variance[1]:.2f}")
print("EIA Results:")
print(EIA_results)

# Tampilkan Visualisasi
fig_contributions.show()
heatmap_fig.show()
fig_pca.show()
fig_time.show()

