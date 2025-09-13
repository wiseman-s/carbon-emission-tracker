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

# ------------------------
# Dataset guidelines and sample
# ------------------------
st.sidebar.markdown("### Dataset Guidelines")
st.sidebar.markdown("""
- **Required columns**: `year`, `source`, `generation_gwh`, `co2_tonnes`
- `year` should be numeric (e.g., 2023)
- `source` should be one of: Hydro, Solar, Wind, Geothermal, Thermal
- `generation_gwh` and `co2_tonnes` must be numeric
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
# Persist dataset in session_state
# ------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.message = None

# ------------------------
# Safe data loader
# ------------------------
def load_energy_data_safe(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    required_cols = ["year", "source", "generation_gwh", "co2_tonnes"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    message = None
    
    if missing_cols:
        message = f"Warning: Missing columns: {', '.join(missing_cols)}. These will be filled with defaults."
        for col in missing_cols:
            if col == "source":
                df[col] = "Unknown"
            else:
                df[col] = 0

    # Clean 'source' column
    df['source'] = df['source'].astype(str).str.title()

    # Remove commas/spaces in numeric columns and force numeric type
    for col in ["year", "generation_gwh", "co2_tonnes"]:
        df[col] = df[col].astype(str).str.replace(",", "").str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Drop rows with invalid year or source
    df = df[df['year'] > 0]
    df = df[df['source'].notna()]

    # Reset index for Altair
    df = df.reset_index(drop=True)

    if df.empty:
        message = "Uploaded dataset is empty or invalid after cleaning."

    return df, message

# ------------------------
# Upload or load demo data
# ------------------------
uploaded_file = st.sidebar.file_uploader("Upload Energy Data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    df, message = load_energy_data_safe(uploaded_file)
    st.session_state.df = df
    st.session_state.message = message
else:
    if st.sidebar.button("Load demo sample data"):
        df, message = load_energy_data_safe("sample_data/sample_energy_data.csv")
        st.session_state.df = df
        st.session_state.message = message

df = st.session_state.df
message = st.session_state.message

if df is None or df.empty:
    st.warning("No valid dataset loaded. Upload a CSV/Excel or click 'Load demo sample data'.")
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
else:
    st.info("No annual data to display.")

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

