import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# 1. SETUP HALAMAN
st.set_page_config(
    page_title="IDM Dashboard",
    page_icon="🇮🇩",
    layout="wide"
)

st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    </style>
""", unsafe_allow_html=True)

# 2. LOAD RESOURCES
@st.cache_resource
def load_resources():
    try:
        with open('model_knn_idm.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        
        df = pd.read_csv('idmFinal23.csv') 
        
        # Mapping Label Status
        if 'STATUS_ENCODED' in df.columns:
            map_rev = {0: 'SGT TERTINGGAL', 1: 'TERTINGGAL', 2: 'BERKEMBANG', 3: 'MAJU', 4: 'MANDIRI'}
            df['Label'] = df['STATUS_ENCODED'].map(map_rev)
            
        return model, df
    except FileNotFoundError:
        return None, None

model, df_dataset = load_resources()

# 3. UTILITIES
def hybrid_input(label, key, default_val, help_text):
    if key not in st.session_state:
        st.session_state[key] = default_val

    def update_from_slider():
        val = st.session_state[f"{key}_slider"]
        st.session_state[key] = val
        st.session_state[f"{key}_input"] = val

    def update_from_input():
        val = st.session_state[f"{key}_input"]
        st.session_state[key] = val
        st.session_state[f"{key}_slider"] = val

    st.sidebar.markdown(f"**{label}**", help=help_text)
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        st.slider("S", 0.0, 1.0, key=f"{key}_slider", value=st.session_state[key], step=0.01, label_visibility="collapsed", on_change=update_from_slider)
    with col2:
        st.number_input("I", 0.0, 1.0, key=f"{key}_input", value=st.session_state[key], step=0.01, label_visibility="collapsed", on_change=update_from_input)
    return st.session_state[key]

# 4. HEADER & SIDEBAR
st.title("🇮🇩 Executive IDM Dashboard")
st.caption("Implementasi Algoritma K-Nearest Neighbor (Modul 5) untuk Klasifikasi Status Desa")

if model is None or df_dataset is None:
    st.error("FILE MISSING: Pastikan 'model_knn_idm.pkl' dan 'idmFinal23.csv' ada di folder!")
    st.stop()

with st.sidebar:
    st.header("Panel Input")
    st.info("Sesuaikan parameter di bawah untuk simulasi prediksi.")
    
    val_iks = hybrid_input("Sosial (IKS)", "iks", 0.60, "Kesehatan, Pendidikan, Modal Sosial")
    val_ike = hybrid_input("Ekonomi (IKE)", "ike", 0.50, "Perdagangan, Logistik, Pasar")
    val_ikl = hybrid_input("Lingkungan (IKL)", "ikl", 0.70, "Ekologi & Mitigasi Bencana")

# 5. PREDICTION LOGIC
input_data = np.array([[val_iks, val_ike, val_ikl]])
prediction = model.predict(input_data)[0]
probs = model.predict_proba(input_data)[0]
confidence = max(probs) * 100

status_map = {0: 'SANGAT TERTINGGAL', 1: 'TERTINGGAL', 2: 'BERKEMBANG', 3: 'MAJU', 4: 'MANDIRI'}
result_text = status_map.get(prediction, "Unknown")

# 6. MAIN CONTENT (TABS)
tab_pred, tab_vis_modul = st.tabs(["Dashboard Prediksi", "Visualisasi Modul 5"])

# --- TAB 1: PREDIKSI & DASHBOARD UTAMA ---
with tab_pred:
    col_left, col_right = st.columns([1, 1.5])
    
    with col_left:
        st.subheader("Hasil Analisis AI")
        
        # Status Card
        color_map = {'MANDIRI': 'green', 'MAJU': 'blue', 'BERKEMBANG': 'orange', 'TERTINGGAL': 'red', 'SGT TERTINGGAL': 'darkred'}
        st.markdown(f"""
        <div style="background-color:{color_map.get(result_text, 'gray')};padding:20px;border-radius:10px;text-align:center;color:white;">
            <h4 style='margin:0'>STATUS DESA:</h4>
            <h1 style='margin:0;font-size:36px'>{result_text}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        c1, c2 = st.columns(2)
        c1.metric("Confidence", f"{confidence:.0f}%")
        c2.metric("Nilai IDM", f"{(val_iks+val_ike+val_ikl)/3:.3f}")
        
        categories = ['Sosial (IKS)', 'Ekonomi (IKE)', 'Lingkungan (IKL)']
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[val_iks, val_ike, val_ikl], theta=categories, fill='toself', name='Input'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), margin=dict(t=30, b=30), height=300)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_right:
        st.subheader("Benchmark 3D")
        df_sample = df_dataset.sample(800)
        fig_3d = px.scatter_3d(
            df_sample, x='IKS_2023', y='IKE_2023', z='IKL_2023',
            color='Label', color_discrete_sequence=px.colors.qualitative.Bold, opacity=0.4,
            title="Sebaran Data Desa (3D Plot)"
        )
        fig_3d.add_scatter3d(
            x=[val_iks], y=[val_ike], z=[val_ikl], mode='markers',
            marker=dict(size=20, color='red', symbol='diamond'), name='POSISI ANDA'
        )
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=500)
        st.plotly_chart(fig_3d, use_container_width=True)

# --- TAB 2: VISUALISASI  ---
with tab_vis_modul:
    st.header("Implementasi Plot Modul 5")
    st.caption("Visualisasi di bawah ini diadaptasi dari Modul 5 menggunakan dataset IDM 2023.")
    
    # Ambil sampel data biar enteng (Seaborn berat kalau 70rb data)
    df_viz = df_dataset.sample(2000)
    
    # Row 1: Histogram & Countplot
    col1, col2 = st.columns(2)
    
    with col1:
        # Plot 1: Distribution (Histogram) 
        st.subheader("1. Distribution Plot")
        target_col = st.selectbox("Pilih Indeks:", ['IKS_2023', 'IKE_2023', 'IKL_2023'])
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.histplot(df_viz[target_col], kde=True, color='blue', ax=ax1)
        ax1.set_title(f'Distribusi Skor {target_col}')
        st.pyplot(fig1)
        
    with col2:
        # Plot 2: Countplot (Status) 
        st.subheader("2. Count Plot Status")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        # Urutkan label biar rapi
        order_list = ['SGT TERTINGGAL', 'TERTINGGAL', 'BERKEMBANG', 'MAJU', 'MANDIRI']
        sns.countplot(y='Label', data=df_viz, order=order_list, palette='Set2', ax=ax2)
        ax2.set_title('Jumlah Desa per Status')
        st.pyplot(fig2)

    st.divider()

    # Row 2: Boxplot & Violinplot
    col3, col4 = st.columns(2)
    
    with col3:
        # Plot 3: Boxplot (Status vs Ekonomi) 
        st.subheader("3. Boxplot (Ekonomi vs Status)")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='IKE_2023', y='Label', data=df_viz, order=order_list, palette='Set1', ax=ax3)
        ax3.set_title('Sebaran Ekonomi per Status')
        st.pyplot(fig3)
        
    with col4:
        # Plot 4: Violin Plot (Status vs Sosial) 
        st.subheader("4. Violin Plot (Sosial vs Status)")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.violinplot(x='IKS_2023', y='Label', data=df_viz, order=order_list, palette='Set3', ax=ax4)
        ax4.set_title('Distribusi Sosial per Status')
        st.pyplot(fig4)

    st.divider()
    
    # Row 3: Pairplot 
    st.subheader("5. Pairplot (Hubungan Antar Indeks)")
    st.caption("Menampilkan korelasi antara IKS, IKE, dan IKL berdasarkan Status.")
    
    fig5 = sns.pairplot(df_viz[['IKS_2023', 'IKE_2023', 'IKL_2023', 'Label']], hue='Label', palette='coolwarm')
    st.pyplot(fig5)

# Footer
st.markdown("---")
st.caption("Praktikum Machine Learning 1")