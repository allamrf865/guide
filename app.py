import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

# Title and Introduction
st.title("📊 Analisis Canggih Program Sovis - Guidebook Pengobatan Massal")
st.markdown("""
### 📖 Evaluasi Efisiensi, Kinerja, dan Prediksi
Selamat datang di dashboard interaktif untuk memantau kinerja anggota tim Program Guidebook Pengobatan Massal. 
Dashboard ini dilengkapi dengan analisis mendalam menggunakan algoritma canggih untuk membantu Anda memahami 
efisiensi, kontribusi, dan proyeksi kerja dari tiap bab guidebook. 
""")

# Data Simulasi (Variabel Analisis Lebih Banyak)
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

# Analisis Dimensi dengan PCA
features = ['Target (%)', 'Realisasi (%)', 'Efektivitas (%)', 'Kompleksitas (%)', 
            'Kepatuhan Prosedur (%)', 'Kedisiplinan (%)', 'Index Kinerja Komposit (IKK)']
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_bab[features])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)
data_bab['PCA1'] = pca_result[:, 0]
data_bab['PCA2'] = pca_result[:, 1]

# Visualisasi Data Utama
st.subheader("📌 Visualisasi Data Kinerja dan Efisiensi")
fig1 = px.bar(
    data_bab, 
    x='Bab', 
    y=['Realisasi (%)', 'Efisiensi Operasional Relatif (EOR)', 'Index Kinerja Komposit (IKK)'], 
    barmode='group',
    title='Realisasi, Efisiensi, dan Kinerja Komposit'
)
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.scatter(
    data_bab,
    x='PCA1', y='PCA2', color='Bab',
    size='Efisiensi Operasional Relatif (EOR)',
    hover_data=['Index Kinerja Komposit (IKK)', 'Dampak Keterlambatan (DK) (%)'],
    title='Analisis Dimensi Kinerja (PCA)'
)
st.plotly_chart(fig2, use_container_width=True)

# Prediksi Waktu Penyelesaian
st.sidebar.header("Prediksi Bab Baru")
target = st.sidebar.slider("Target (%)", min_value=80, max_value=100, value=100)
realisasi = st.sidebar.slider("Realisasi (%)", min_value=70, max_value=100, value=90)
efektivitas = st.sidebar.slider("Efektivitas (%)", min_value=70, max_value=100, value=85)
kompleksitas = st.sidebar.slider("Kompleksitas (%)", min_value=60, max_value=100, value=80)
kedisiplinan = st.sidebar.slider("Kedisiplinan (%)", min_value=70, max_value=100, value=90)

X_new = pd.DataFrame({
    'Target (%)': [target],
    'Realisasi (%)': [realisasi],
    'Efektivitas (%)': [efektivitas],
    'Kompleksitas (%)': [kompleksitas],
    'Kedisiplinan (%)': [kedisiplinan]
})
poly = PolynomialFeatures(degree=3)
ridge = Ridge(alpha=0.1).fit(poly.fit_transform(data_bab[features]), data_bab['Waktu Realisasi (Minggu)'])
predicted_time = ridge.predict(poly.transform(X_new))

# Tampilkan Prediksi
st.subheader("🔮 Prediksi Waktu Penyelesaian Bab Baru")
st.markdown(f"""
- **Target (%)**: {target}  
- **Realisasi (%)**: {realisasi}  
- **Efektivitas (%)**: {efektivitas}  
- **Kompleksitas (%)**: {kompleksitas}  
- **Kedisiplinan (%)**: {kedisiplinan}  
- **Prediksi Waktu Penyelesaian**: **{predicted_time[0]:.2f} Minggu**  
""")

# Kesimpulan
st.subheader("📋 Interpretasi Hasil")
max_efficiency = data_bab['Efisiensi Operasional Relatif (EOR)'].idxmax()
max_delay = data_bab['Dampak Keterlambatan (DK) (%)'].idxmax()
st.markdown(f"""
1. **Efisiensi Operasional Relatif (EOR)** tertinggi ditemukan pada Bab **{data_bab.iloc[max_efficiency]['Bab']}** 
   dengan skor **{data_bab.iloc[max_efficiency]['Efisiensi Operasional Relatif (EOR)']:.2f}**.
2. Bab dengan **Dampak Keterlambatan (DK)** tertinggi adalah **{data_bab.iloc[max_delay]['Bab']}** 
   sebesar **{data_bab.iloc[max_delay]['Dampak Keterlambatan (DK) (%)']:.2f}%**.
3. Index Kinerja Komposit (IKK) memberikan pandangan menyeluruh tentang kinerja tiap bab berdasarkan kontribusi, efektivitas, dan kepatuhan prosedur.
4. Prediksi waktu penyelesaian untuk bab baru membantu merencanakan alokasi waktu lebih baik.
""")

# Download Data
st.download_button(
    label="📥 Download Data Lengkap sebagai CSV",
    data=data_bab.to_csv(index=False),
    file_name="analisis_program_advanced.csv",
    mime="text/csv"
)
