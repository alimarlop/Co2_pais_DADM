"""
Streamlit app for predicting CO₂ emissions per capita and classifying a fictional
country called UABer.  The app allows the user to adjust economic and energy
parameters with sliders, computes an estimated CO₂ per capita value based on
simple heuristic relationships, and classifies the country into one of four
profiles derived from the project analysis: ‘Rics & Contaminants’, ‘Rics & Nets’,
‘En Desenvolupament’ and ‘Pobres & Emissors Baixos’.  It also displays the
country on a synthetic scatter plot of Human Development Index (HDI) versus
predicted CO₂ per capita alongside representative sample points for each
profile.

This application does not rely on the external datasets used in the project,
making it self‑contained and easy to run.  The classification logic and
emissions estimation are based on the patterns uncovered during the project:
higher GDP per capita and lower renewable energy share tend to increase
emissions, while higher HDI correlates with higher consumption.  Sectoral
contributions (agriculture, manufacturing and transport) add further
adjustments, reflecting the relative emissions intensity of those activities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Prediction and classification logic
#
# The following functions implement heuristic relationships between the
# socioeconomic inputs and the estimated emissions per capita.  These rules are
# not trained from real data but are crafted to reflect the qualitative
# relationships observed in the coursework.  You can adjust the coefficients
# within these functions to tune the behaviour of the model.

def predict_co2_per_capita(
    gdp_pc: float,
    renew_pct: float,
    hdi: float,
    agriculture_pct: float,
    manufacturing_pct: float,
    transport_pct: float,
) -> float:
    """Estimate CO₂ emissions per capita (tonnes) from input parameters.

    Parameters
    ----------
    gdp_pc : float
        Gross domestic product per capita (USD).
    renew_pct : float
        Share of renewable energy in the electricity mix (%) on a 0–100 scale.
    hdi : float
        Human Development Index on a 0–1 scale.
    agriculture_pct : float
        Estimated contribution of the agricultural sector to national emissions (%)
        on a 0–100 scale.  Used as a relative weight in the emissions model.
    manufacturing_pct : float
        Estimated contribution of the manufacturing/construction sector (%) on a
        0–100 scale.
    transport_pct : float
        Estimated contribution of the transport sector (%) on a 0–100 scale.

    Returns
    -------
    float
        Estimated tonnes of CO₂ emissions per capita.

    Notes
    -----
    The formula used here combines several terms:
    - A base term proportional to GDP per capita, reflecting the link between
      wealth and consumption.
    - A renewable factor, where a lower renewable share increases emissions.
    - An HDI factor, since higher HDI is typically associated with higher
      consumption and energy use.
    - Sectoral adjustments, where manufacturing and transport carry larger
      weights than agriculture due to their higher emissions intensity.
    """
    # Convert GDP to a base indicator (higher GDP -> higher potential emissions)
    base = gdp_pc / 20_000  # scaled so that 20 k USD yields base=1

    # Factor capturing how non‑renewable the energy mix is (0 = fully renewable,
    # 3 = fully fossil).  Emissions increase with the share of fossil energy.
    nonrenewable_factor = (1 - renew_pct / 100) * 3

    # HDI factor: 0.5 for very low HDI and up to 1.5 for very high HDI
    hdi_factor = 0.5 + hdi

    # Core emissions estimate.  The constant multiplier (3) tunes the order of
    # magnitude to yield values similar to typical per‑capita emissions (0–30 t).
    predicted = base * nonrenewable_factor * hdi_factor * 3

    # Sectoral adjustments.  Manufacturing and transport are given larger
    # coefficients because they tend to be more emissions intensive than
    # agriculture.  The units here add directly to the tonne estimate.
    sector_adjust = (
        0.01 * agriculture_pct
        + 0.05 * manufacturing_pct
        + 0.03 * transport_pct
    )

    return max(predicted + sector_adjust, 0.0)


def classify_country(
    gdp_pc: float,
    co2_pc: float,
    hdi: float,
    renew_pct: float,
) -> str:
    """Classify a country into one of four profiles based on GDP and emissions.

    Parameters
    ----------
    gdp_pc : float
        GDP per capita in USD.
    co2_pc : float
        Estimated CO₂ emissions per capita (tonnes).
    hdi : float
        Human Development Index (0–1).  Currently not used but kept for
        extensibility.
    renew_pct : float
        Percentage of renewable energy in the electricity mix.  Currently not
        used in the classifier but retained for consistency.

    Returns
    -------
    str
        One of the cluster names: ‘Rics & Contaminants’, ‘Rics & Nets’,
        ‘En Desenvolupament’ or ‘Pobres & Emissors Baixos’.

    Notes
    -----
    This classifier uses heuristic thresholds inspired by the clustering
    discovered in the coursework.  Rich countries are defined as those with
    GDP per capita above 20 000 USD.  Among them, if per‑capita emissions
    exceed 10 t, they are labelled as “Rics & Contaminants”; otherwise they
    are “Rics & Nets”.  Countries with GDP below 5 000 USD are considered
    poorer; if their emissions are very low (≤3 t), they fall into the
    “Pobres & Emissors Baixos” cluster, otherwise they are treated as
    “En Desenvolupament”.  All others are classified as “En Desenvolupament”.
    """
    # Rich countries: GDP per capita above 20 k USD
    if gdp_pc >= 20_000:
        return (
            "Rics & Contaminants"
            if co2_pc > 10.0
            else "Rics & Nets"
        )

    # Poor countries: GDP per capita below 5 k USD
    if gdp_pc < 5_000:
        return (
            "Pobres & Emissors Baixos"
            if co2_pc <= 3.0
            else "En Desenvolupament"
        )

    # Intermediate GDP countries are considered in development
    return "En Desenvolupament"


def generate_synthetic_dataset(n: int = 200) -> pd.DataFrame:
    """Generate a synthetic dataset of countries for the scatter plot.

    The synthetic data is used purely for visualisation.  It samples GDP per
    capita, HDI, renewable share and sector contributions from plausible ranges
    and computes the corresponding emissions and cluster using the same
    functions as the user input.  This makes the scatter plot representative of
    the classification logic without requiring the original datasets.

    Parameters
    ----------
    n : int, default 200
        Number of synthetic countries to generate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: gdp_pc, hdi, co2_pc, cluster.
    """
    rng = np.random.default_rng(0)
    records = []
    for _ in range(n):
        gdp_pc = rng.uniform(500, 60_000)
        renew_pct = rng.uniform(10, 80)
        hdi = rng.uniform(0.4, 1.0)
        agriculture_pct = rng.uniform(0, 40)
        manufacturing_pct = rng.uniform(0, 60)
        transport_pct = rng.uniform(0, 40)
        co2_pc = predict_co2_per_capita(
            gdp_pc,
            renew_pct,
            hdi,
            agriculture_pct,
            manufacturing_pct,
            transport_pct,
        )
        cluster = classify_country(
            gdp_pc,
            co2_pc,
            hdi,
            renew_pct,
        )
        records.append(
            {
                "gdp_pc": gdp_pc,
                "hdi": hdi,
                "co2_pc": co2_pc,
                "cluster": cluster,
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="Predicció de CO₂ per càpita per al país UABer",
        page_icon="🟢",
        layout="wide",
    )

    st.title("Predicció de CO₂ per càpita per al país fictici 'UABer'")

    st.markdown(
        """
        Aquest simulador permet definir un país fictici ajustant diverses
        variables socioeconòmiques i veure quina seria la seva emissió de CO₂ per
        habitant.  També classifica el país en un dels quatre perfils descoberts
        en el projecte: **Rics & Contaminants**, **Rics & Nets**, **En
        Desenvolupament** o **Pobres & Emissors Baixos**.  A la dreta es mostra
        el punt corresponent a la predicció sobre un diagrama de dispersió amb
        dades sintètiques representatives.
        """
    )

    # Sidebar with input sliders
    st.sidebar.header("Paràmetres del país UABer")
    gdp_pc = st.sidebar.slider(
        "PIB per càpita (USD)",
        min_value=500.0,
        max_value=100_000.0,
        value=20_000.0,
        step=500.0,
    )
    renew_pct = st.sidebar.slider(
        "Percentatge d’energia renovable (%)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=1.0,
    )
    hdi = st.sidebar.slider(
        "Índex de Desenvolupament Humà (HDI)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.01,
    )
    exports_pc = st.sidebar.slider(
        "Exportacions per càpita (USD)",
        min_value=0.0,
        max_value=50_000.0,
        value=5_000.0,
        step=500.0,
    )
    imports_pc = st.sidebar.slider(
        "Importacions per càpita (USD)",
        min_value=0.0,
        max_value=50_000.0,
        value=5_000.0,
        step=500.0,
    )
    # Sector sliders.  These allow the user to assign relative weights to
    # different sectors.  Their sum is not enforced to equal 100% but a
    # normalisation step is applied below if the sum exceeds 100.
    st.sidebar.markdown("**Pes relatiu d’emissions per sector (%)**")
    agriculture_pct = st.sidebar.slider(
        "Agricultura",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
    )
    manufacturing_pct = st.sidebar.slider(
        "Manufactura i construcció",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=1.0,
    )
    transport_pct = st.sidebar.slider(
        "Transport",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0,
    )

    # Normalise sectors if their sum exceeds 100
    total_sector = agriculture_pct + manufacturing_pct + transport_pct
    if total_sector > 100:
        scale = 100 / total_sector
        agriculture_pct *= scale
        manufacturing_pct *= scale
        transport_pct *= scale
        st.sidebar.info(
            "Els percentatges sectorials s’han ajustat per sumar 100%."
        )

    # Compute predicted CO₂ per capita
    co2_pc = predict_co2_per_capita(
        gdp_pc,
        renew_pct,
        hdi,
        agriculture_pct,
        manufacturing_pct,
        transport_pct,
    )

    # Classify country
    cluster = classify_country(
        gdp_pc,
        co2_pc,
        hdi,
        renew_pct,
    )

    # Layout: results on the left, scatter plot on the right
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Resultats de la predicció")
        st.write(
            f"**Emissions de CO₂ per càpita estimades:** {co2_pc:.2f} t/persona"
        )
        st.write(f"**Classificació del país:** {cluster}")
        st.write("**Paràmetres del model:**")
        st.write(f"PIB per càpita: {gdp_pc:.0f} USD")
        st.write(f"Energia renovable: {renew_pct:.1f}%")
        st.write(f"HDI: {hdi:.2f}")
        st.write(
            f"Pes sectorial (agricultura / manufactura / transport): "
            f"{agriculture_pct:.1f}% / {manufacturing_pct:.1f}% / {transport_pct:.1f}%"
        )

    with col2:
        # Generate synthetic data for scatter plot
        synthetic_df = generate_synthetic_dataset()
        fig, ax = plt.subplots(figsize=(8, 5))
        # Colour map for clusters
        colours = {
            "Rics & Contaminants": "#d62728",
            "Rics & Nets": "#2ca02c",
            "En Desenvolupament": "#ff7f0e",
            "Pobres & Emissors Baixos": "#1f77b4",
        }
        for cl, grp in synthetic_df.groupby("cluster"):
            ax.scatter(
                grp["hdi"],
                grp["co2_pc"],
                s=40,
                alpha=0.7,
                label=cl,
                color=colours.get(cl, "grey"),
            )
        # Plot the new country's point
        ax.scatter(
            [hdi],
            [co2_pc],
            color="black",
            marker="x",
            s=100,
            label="UABer",
        )
        ax.set_xlabel("Índex de Desenvolupament Humà (HDI)")
        ax.set_ylabel("CO₂ per càpita (t/persona)")
        ax.set_title(
            "Distribució sintètica d'emissions vs. desenvolupament i punt UABer"
        )
        ax.legend(loc="best", fontsize=8)
        st.pyplot(fig)


if __name__ == "__main__":
    main()