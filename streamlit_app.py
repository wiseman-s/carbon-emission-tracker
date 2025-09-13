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
from reportlab.lib.utils import ImageReader
from PIL import Image

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
- `generation_gwh` must be numeric
- If `co2_tonnes` is missing, the system will calculate it automatically
""")
st.sidebar.markdown("#### Sample Dataset")
sample_df = pd.DataFrame({
    "year": [2023, 2023, 2023],
    "source": ["Hydro", "Solar", "Wind"],
    "generation_gwh": [500, 200, 150],
    "co2_tonnes": [0, 50, 30]
})
st.sidebar.dataframe(sample_df)

# ------------------------
# Session state
# ------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.message = None

# ------------------------
# Default CO2 emission factors (tonnes per GWh)
# ------------------------
default_emission_factors = {
    "Hydro": 0,
    "Solar": 0,
    "Wind": 0,
    "Geothermal": 5,   # example value
    "Thermal": 900,    # example value for coal/gas
    "Unknown": 100     # fallback for unknown sources
}

# ------------------------
# Robust loader with CO2 calculation
# ------------------------
def load_energy_data_safe(uploaded_file=None):
    try:
        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        else:
            df = sample_df.copy()

        # Normalize column names
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

        # Required columns
        required_cols = ["year", "source", "generation_gwh", "co2_tonnes"]
        message = None
        for col in required_cols:
            if col not in df.columns:
                message = f"Column '{col}' is missing. Filling with default values."
                if col == "source":
                    df[col] = "Unknown"
                else:
                    df[col] = 0

        # Convert numeric columns
        df['year'] = pd.to_numeric(df['year'].astype(str).str.replace(",", "").str.strip(), errors='coerce')
        df['generation_gwh'] = pd.to_numeric(df['generation_gwh'].astype(str).str.replace(",", "").str.strip(), errors='coerce')

        # Compute CO2 if missing or zero
        df['co2_tonnes'] = pd.to_numeric(df.get('co2_tonnes', 0), errors='coerce')
        def compute_co2(row):
            try:
                if pd.isna(row['co2_tonnes']) or row['co2_tonnes'] == 0:
                    factor = default_emission_factors.get(row['source'], 100)
                    return row['generation_gwh'] * factor
                return row['co2_tonnes']
            except:
                return 0
        df['co2_tonnes'] = df.apply(compute_co2, axis=1)

        # Mark invalid rows
        df['valid'] = df['year'].notna() & df['source'].notna() & df['generation_gwh'].notna() & df['co2_tonnes'].notna()
        if df['valid'].sum() == 0:
            return None, "No valid rows found. Check column names and data formatting."

        if df['valid'].sum() < len(df):
            message = "Some rows have missing or invalid data and were ignored in analysis."

        df_clean = df[df['valid']].copy()
        df_clean = df_clean.reset_index(drop=True)
        return df_clean, message

    except Exception as e:
        return None, f"Error loading dataset: {e}"

# ------------------------
# Upload file
# ------------------------
uploaded_file = st.sidebar.file_uploader("Upload Energy Data (CSV or Excel)", type=["csv", "xlsx"])
df, message = load_energy_data_safe(uploaded_file)

st.session_state.df = df
st.session_state.message = message

if df is None:
    st.warning(message)
    st.stop()
if message:
    st.warning(message)

# ------------------------
# Data preview
# ------------------------
st.subheader("Preview of cleaned data (first 20 rows)")
st.dataframe(df.head(20))

# ------------------------
# Aggregations
# ------------------------
gen_by_source = df.groupby("source", as_index=False)["generation_gwh"].sum().sort_values("generation_gwh", ascending=False)
em_by_source = df.groupby("source", as_index=False)["co2_tonnes"].sum().sort_values("co2_tonnes", ascending=False)
annual = df.groupby("year", as_index=False).agg(
    total_generation_gwh=pd.NamedAgg(column="generation_gwh", aggfunc="sum"),
    total_emissions_tonnes=pd.NamedAgg(column="co2_tonnes", aggfunc="sum")
).sort_values("year")
annual['year'] = annual['year'].astype(int)

# ------------------------
# Charts
# ------------------------
colors = {"Hydro":"#1f77b4","Solar":"#ff7f0e","Wind":"#2ca02c","Geothermal":"#d62728","Thermal":"#9467bd"}

st.subheader("📊 Energy Generation by Source (Total)")
if not gen_by_source.empty:
    chart_gen_source = alt.Chart(gen_by_source).mark_bar().encode(
        x=alt.X("source:N", sort='-y', title="Energy Source"),
        y=alt.Y("generation_gwh:Q", title="Total Generation (GWh)"),
        color=alt.Color("source:N", scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values())), legend=None),
        tooltip=[alt.Tooltip("source:N"), alt.Tooltip("generation_gwh:Q", format=",.2f")]
    ).properties(height=400, width=700)
    st.altair_chart(chart_gen_source)
else:
    st.info("No generation data to display.")

st.subheader("🌫️ CO₂ Emissions by Source (Total)")
if not em_by_source.empty:
    chart_em_source = alt.Chart(em_by_source).mark_bar().encode(
        x=alt.X("source:N", sort='-y', title="Energy Source"),
        y=alt.Y("co2_tonnes:Q", title="Total CO₂ Emissions (tonnes)"),
        color=alt.Color("source:N", scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values())), legend=None),
        tooltip=[alt.Tooltip("source:N"), alt.Tooltip("co2_tonnes:Q", format=",.0f")]
    ).properties(height=400, width=700)
    st.altair_chart(chart_em_source)
else:
    st.info("No emissions data to display.")

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
# Metrics, forecast, insights, PDF generation
# ------------------------
# Keep your original code for metrics, forecast, insights, PDF here
