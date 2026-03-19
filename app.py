import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# --- 1. CONFIGURACIÓ DE LA PÀGINA ---
st.set_page_config(
    page_title="Simulador CO₂ - País UAB",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CÀRREGA DE DADES I ENTRENAMENT DEL MODEL (Cauchejat per rapidesa) ---
@st.cache_data
def load_and_train_model():
    # Carreguem les dades reals del projecte
    df = pd.read_csv("master_dataset.csv")
    
    # Filtrem com a la Fase 4 (Any 2019)
    df_clust = df[df["Year"] == 2019].copy()
    cluster_vars = ["co2_prod_pc", "gdp_pc", "hdi"]
    df_clust = df_clust.dropna(subset=cluster_vars)
    
    # Entrenem el KMeans amb les dades reals
    scaler = StandardScaler()
    X_clust = scaler.fit_transform(df_clust[cluster_vars])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_clust["cluster"] = kmeans.fit_predict(X_clust)
    
    # Mapeig intel·ligent per posar el nom correcte a cada clúster segons els centroides
    cluster_means = df_clust.groupby("cluster")[cluster_vars].mean()
    sorted_hdi = cluster_means["hdi"].sort_values(ascending=False)
    rich_clusters = sorted_hdi.index[:2].tolist()
    poor_clusters = sorted_hdi.index[2:].tolist()
    
    rich_co2 = cluster_means.loc[rich_clusters, "co2_prod_pc"].sort_values()
    poor_co2 = cluster_means.loc[poor_clusters, "co2_prod_pc"].sort_values()
    
    cluster_names = {
        rich_co2.index[0]: "Rics & Nets",
        rich_co2.index[1]: "Rics & Contaminants",
        poor_co2.index[0]: "Pobres & Emissors Baixos",
        poor_co2.index[1]: "En Desenvolupament",
    }
    
    df_clust["cluster_nom"] = df_clust["cluster"].map(cluster_names)
    
    return df_clust, kmeans, scaler, cluster_names

df_clust, kmeans_model, scaler, cluster_names = load_and_train_model()

# --- 3. FÓRMULA DE PREDICCIÓ (Sensible als sliders) ---
def predict_co2_uaber(gdp, renew, hdi, exp, imp, agri, manu, trans):
    # Base per riquesa i desenvolupament
    base = (gdp / 20000) * (hdi ** 2) * 5.0
    
    # Factor renovables (si tens 0% renovables, contamines molt més)
    fossil_ratio = (100 - renew) / 100
    energy_mod = 1.0 + (fossil_ratio * 1.5) 
    
    # Balanç comercial (Externalització: si exportes molt més del que importes, produeixes més CO2)
    net_exports = exp - imp
    trade_mod = 1.0 + (net_exports / 20000) * 0.8 
    
    # Pesos sectorials (La manufactura dispara les emissions, l'agricultura menys)
    sector_mod = (manu * 0.15) + (trans * 0.08) + (agri * 0.03)
    
    # Càlcul final
    pred = base * energy_mod * trade_mod + sector_mod
    return max(pred, 0.1) # Evitem valors negatius

# --- 4. INTERFÍCIE D'USUARI ---
st.title("🌍 Simulador Interactiu d'Emissions: El País *UAB*")
st.markdown("""
Benvingut al simulador. Ajusta els paràmetres socioeconòmics del nostre país fictici (**UAB**) a la barra lateral. 
L'aplicació calcularà les emissions estimades de CO₂ utilitzant els patrons descoberts al nostre estudi i **classificarà el país** en temps real dins de l'algorisme de clustering.
""")

# Sliders a la barra lateral
st.sidebar.header("⚙️ Paràmetres del País UAB")

gdp_pc = st.sidebar.slider("PIB per càpita (USD)", 500.0, 80000.0, 20000.0, 500.0)
hdi = st.sidebar.slider("Índex de Desenvolupament Humà (HDI)", 0.30, 1.00, 0.75, 0.01)
renew_pct = st.sidebar.slider("Energia Renovable (%)", 0.0, 100.0, 30.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Balanç Comercial")
exports_pc = st.sidebar.slider("Exportacions per càpita (USD)", 0.0, 40000.0, 5000.0, 500.0)
imports_pc = st.sidebar.slider("Importacions per càpita (USD)", 0.0, 40000.0, 5000.0, 500.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Sectors (% d'activitat)")
st.sidebar.caption("L'impacte de la manufactura i el transport és molt superior al de l'agricultura.")
manu_pct = st.sidebar.slider("Manufactura i Indústria", 0.0, 60.0, 20.0, 1.0)
trans_pct = st.sidebar.slider("Transport", 0.0, 40.0, 15.0, 1.0)
agri_pct = st.sidebar.slider("Agricultura", 0.0, 50.0, 10.0, 1.0)

# --- 5. CÀLCULS EN TEMPS REAL ---
# 1. Predicció del CO2
co2_pred = predict_co2_uaber(gdp_pc, renew_pct, hdi, exports_pc, imports_pc, agri_pct, manu_pct, trans_pct)

# 2. Classificació fent servir l'IA entrenada (StandardScaler + KMeans)
uaber_data = pd.DataFrame([[co2_pred, gdp_pc, hdi]], columns=["co2_prod_pc", "gdp_pc", "hdi"])
uaber_scaled = scaler.transform(uaber_data)
uaber_cluster_id = kmeans_model.predict(uaber_scaled)[0]
uaber_cluster_nom = cluster_names[uaber_cluster_id]

# Assignació de colors segons el teu gràfic original
palette_clust = {
    "Rics & Contaminants": "#d62728", # Vermell
    "Rics & Nets": "#2ca02c",         # Verd
    "En Desenvolupament": "#ff7f0e",  # Taronja
    "Pobres & Emissors Baixos": "#1f77b4" # Blau
}
uaber_color = palette_clust[uaber_cluster_nom]

# --- 6. RENDERITZAT DE RESULTATS ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Emissions CO₂ Estimades", value=f"{co2_pred:.2f} t/càpita", delta="Predicció")
with col2:
    st.metric(label="Grup predit", value=uaber_cluster_nom)
with col3:
    st.metric(label="Balanç Comercial", value=f"{exports_pc - imports_pc:.0f} USD", 
              delta="Exportador net" if exports_pc > imports_pc else "Importador net")

st.markdown("### Distribució Global i Posició de UAB")

# Creació del gràfic exacte a la Fase 4
fig, ax = plt.subplots(figsize=(8, 4))

# Pintem els punts reals del dataset
for nom, grup in df_clust.groupby("cluster_nom"):
    ax.scatter(grup["hdi"], grup["co2_prod_pc"], 
               label=nom, alpha=0.5, s=50, 
               color=palette_clust.get(nom, "gray"), edgecolors="none")

# Pintem el punt de UABer destacat
ax.scatter(hdi, co2_pred, color=uaber_color, marker="*", s=400, edgecolor="black", linewidth=1.5, label="País UAB", zorder=5)
ax.annotate("UAB", (hdi, co2_pred), fontsize=12, fontweight="bold", xytext=(8, 8), textcoords="offset points", color="black")

# Estètica del gràfic
ax.set_xlabel("Índex de Desenvolupament Humà (HDI)", fontsize=11)
ax.set_ylabel("CO₂ per Càpita (t)", fontsize=11)
ax.set_title("Clustering de Països per Perfil: Desenvolupament vs Emissions", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper left")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.3, linestyle="--")

# Mostrar gràfic a Streamlit
st.pyplot(fig, width='content')

st.caption("Les dades de fons són els països reals de l'any 2019.")