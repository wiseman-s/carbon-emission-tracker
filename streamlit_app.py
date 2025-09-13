# streamlit_app.py
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from carbon_loader import human_equivalents
from sklearn.linear_model import LinearRegression
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# ------------------------
# Page config
# ------------------------
st.set_page_config(
    page_title="Carbon Emission Tracker",
    page_icon="🌍",
    layout="wide"
)

# ------------------------
# Heading
# ------------------------
st.title("🌍 Carbon Emission Tracker")
st.markdown("### Visualize, estimate, and explore emission scenarios for energy generation")

# ------------------------
# Sidebar
# ------------------------
st.sidebar.title("Carbon Emission Tracker")
st.sidebar.markdown("Powering People for a Better Tomorrow — sustainable, reliable, affordable energy.")
st.sidebar.markdown("---")

st.sidebar.markdown("### Dataset Guidelines")
st.sidebar.markdown("""
- Required columns: `year`, `source`, `generation_gwh`, `co2_tonnes`
- `year` should be numeric (e.g., 2023)
- `source` should be one of: Hydro, Solar, Wind, Geothermal, Thermal
- `generation_gwh` and `co2_tonnes` must be numeric
""")

# ------------------------
# Safe loader for uploaded file or demo data
# ------------------------
def load_energy_data_safe(uploaded_file=None):
    try:
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        else:
            # Fallback demo dataset
            df = pd.DataFrame({
                "year": [2023, 2023, 2023],
                "source": ["Hydro", "Solar", "Wind"],
                "generation_gwh": [500, 200, 150],
                "co2_tonnes": [0, 50, 30]
            })

        # Ensure required columns
        required_cols = ["year", "source", "generation_gwh", "co2_tonnes"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0 if col != "source" else "Unknown"

        # Clean columns
        df['source'] = df['source'].astype(str).str.title()
        for col in ["year", "generation_gwh", "co2_tonnes"]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.strip(), errors='coerce').fillna(0)

        # Drop invalid rows
        df = df[df['year'] > 0]
        df = df[df['source'].notna()]
        df = df.reset_index(drop=True)

        if df.empty:
            return None, "Dataset is empty after cleaning."
        return df, None
    except Exception as e:
        return None, f"Error loading dataset: {e}"

# ------------------------
# Upload file or load demo
# ------------------------
uploaded_file = st.sidebar.file_uploader("Upload Energy Data (CSV or Excel)", type=["csv", "xlsx"])
df, message = load_energy_data_safe(uploaded_file)

if df is None:
    st.warning(message or "No valid dataset.")
    st.stop()
if message:
    st.warning(message)

# ------------------------
# Preview
# ------------------------
st.subheader("Preview of cleaned data (first 20 rows)")
st.dataframe(df.head(20))

# ------------------------
# Aggregations
# ------------------------
gen_by_source = df.groupby("source", as_index=False)["generation_gwh"].sum()
em_by_source = df.groupby("source", as_index=False)["co2_tonnes"].sum()
annual = df.groupby("year", as_index=False).agg(
    total_generation_gwh=("generation_gwh", "sum"),
    total_emissions_tonnes=("co2_tonnes", "sum")
)
annual['year'] = annual['year'].astype(int)

# ------------------------
# Charts
# ------------------------
colors = {
    "Hydro": "#1f77b4",
    "Solar": "#ff7f0e",
    "Wind": "#2ca02c",
    "Geothermal": "#d62728",
    "Thermal": "#9467bd",
}

st.subheader("📊 Energy Generation by Source (Total)")
if not gen_by_source.empty:
    chart_gen_source = alt.Chart(gen_by_source).mark_bar().encode(
        x=alt.X("source:N", sort='-y', title="Energy Source"),
        y=alt.Y("generation_gwh:Q", title="Total Generation (GWh)"),
        color=alt.Color("source:N", scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values())), legend=None),
        tooltip=[alt.Tooltip("source:N"), alt.Tooltip("generation_gwh:Q", format=",.2f")]
    ).properties(height=400, width=700)
    st.altair_chart(chart_gen_source)

st.subheader("🌫️ CO₂ Emissions by Source (Total)")
if not em_by_source.empty:
    chart_em_source = alt.Chart(em_by_source).mark_bar().encode(
        x=alt.X("source:N", sort='-y', title="Energy Source"),
        y=alt.Y("co2_tonnes:Q", title="Total CO₂ Emissions (tonnes)"),
        color=alt.Color("source:N", scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values())), legend=None),
        tooltip=[alt.Tooltip("source:N"), alt.Tooltip("co2_tonnes:Q", format=",.0f")]
    ).properties(height=400, width=700)
    st.altair_chart(chart_em_source)

st.subheader("📈 Annual Trends")
if not annual.empty:
    chart_gen_annual = alt.Chart(annual).mark_line(point=True, color="#2E8B57").encode(
        x="year:Q",
        y="total_generation_gwh:Q",
        tooltip=[alt.Tooltip("year:Q"), alt.Tooltip("total_generation_gwh:Q", format=",.0f")]
    ).properties(height=300, width=600)
    chart_em_annual = alt.Chart(annual).mark_line(point=True, color="#FF8C00").encode(
        x="year:Q",
        y="total_emissions_tonnes:Q",
        tooltip=[alt.Tooltip("year:Q"), alt.Tooltip("total_emissions_tonnes:Q", format=",.0f")]
    ).properties(height=300, width=600)
    col1, col2 = st.columns(2)
    col1.altair_chart(chart_gen_annual)
    col2.altair_chart(chart_em_annual)

# ------------------------
# Key metrics
# ------------------------
total_gen = gen_by_source['generation_gwh'].sum() if not gen_by_source.empty else 0
total_emissions = em_by_source['co2_tonnes'].sum() if not em_by_source.empty else 0
equiv = human_equivalents(total_emissions) if total_emissions > 0 else {"trees": 0, "cars": 0, "homes": 0}

st.subheader("📌 Key Metrics")
c1, c2, c3 = st.columns(3)
c1.metric("⚡ Total Generation (GWh)", f"{total_gen:,.0f}")
c2.metric("🌫️ Total CO₂ (tonnes)", f"{total_emissions:,.0f}")
c3.metric("🌳 Tree Equivalent", f"{equiv['trees']:,} trees")
st.markdown(f"Other equivalents: {equiv['cars']:,} cars off the road per year • {equiv['homes']:,} homes powered per year")
