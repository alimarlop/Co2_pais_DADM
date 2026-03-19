import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="CO2 Simulator", layout="wide")

st.title("🌍 Simulador d'Emissions de CO₂ - País UABERS")
st.write("Modifica les variables per veure com canvien les emissions")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("master_dataset.csv")  # <-- vuestro dataset final
    return df

df = load_data()

# -----------------------------
# VARIABLES
# -----------------------------
features = [
    "gdp_pc",
    "hdi",
    "gini",
    "exports_share_gdp",
    "pct_lowcarbon"
]

target = "co2_prod_pc"

df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# -----------------------------
# MODEL
# -----------------------------
model = LinearRegression()
model.fit(X, y)

# -----------------------------
# SIDEBAR (INPUTS)
# -----------------------------
st.sidebar.header("Paràmetres del país UABERS")

gdp = st.sidebar.slider("PIB per càpita ($)", 500, 80000, 20000)
hdi = st.sidebar.slider("HDI", 0.4, 1.0, 0.8)
gini = st.sidebar.slider("Gini (desigualtat)", 20, 60, 35)
exports = st.sidebar.slider("Exportacions (% PIB)", 0.1, 100, 25)
renewables = st.sidebar.slider("% Energies baixes en carboni", 0, 100, 30)

# -----------------------------
# PREDICTION
# -----------------------------
input_data = np.array([[gdp, hdi, gini, exports, renewables]])
prediction = model.predict(input_data)[0]

# -----------------------------
# OUTPUT
# -----------------------------
st.subheader("📊 Resultat")

st.metric(
    label="Emissions CO₂ per càpita (UABERS)",
    value=f"{prediction:.2f} tones/persona"
)

# -----------------------------
# COMPARISON GRAPH
# -----------------------------
st.subheader("Comparació amb països reals")

sample = df.sample(200)

fig, ax = plt.subplots()

ax.scatter(sample["gdp_pc"], sample["co2_prod_pc"], alpha=0.5)

# punto UABERS
ax.scatter(gdp, prediction, s=200, marker="X")

ax.set_xlabel("PIB per càpita")
ax.set_ylabel("CO₂ produït per càpita")
ax.set_title("Relació PIB vs CO₂")

st.pyplot(fig)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("Importància de variables")

coef_df = pd.DataFrame({
    "Variable": features,
    "Coeficient": model.coef_
})

st.dataframe(coef_df)

# -----------------------------
# INTERPRETATION
# -----------------------------
st.subheader("Interpretació")

st.write("""
Aquest simulador es basa en un model de regressió lineal entrenat amb dades globals.
Permet observar com canvis en variables econòmiques i energètiques afecten les emissions.

Tendències generals:
- PIB i exportacions augmenten emissions
- Energies renovables redueixen emissions
- La desigualtat (Gini) pot influir en les emissions
""")