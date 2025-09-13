# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PIL import Image
from reportlab.lib.utils import ImageReader

# ------------------------
# Safe chart-to-PNG function (skip if altair_saver missing)
# ------------------------
def save_chart_image(chart):
    try:
        from io import BytesIO
        from altair_saver import save as alt_save
        buf = BytesIO()
        alt_save(chart, fp=buf, fmt="png", scale_factor=2)
        buf.seek(0)
        return buf
    except ModuleNotFoundError:
        return None
    except Exception:
        return None

# ------------------------
# Page config
# ------------------------
st.set_page_config(page_title="Kenya Carbon Emission Tracker", page_icon="🌍", layout="wide")

st.title("🌍 Kenya Carbon Emission Tracker")
st.markdown("### Visualize, estimate, and explore emission scenarios for energy generation in Kenya")

# ------------------------
# Sidebar
# ------------------------
st.sidebar.title("Kenya Carbon Emission Tracker")
st.sidebar.markdown("Sustainable, reliable, and affordable energy for Kenya.")
st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Guidelines")
st.sidebar.markdown("""
- Required columns: `year`, `source`, `generation_gwh`, `co2_tonnes` (optional)
- `year`: numeric (e.g., 2023)
- `source`: Hydro, Solar, Wind, Geothermal, Thermal
- `generation_gwh`: numeric
- Missing `co2_tonnes` will be calculated automatically
""")
st.sidebar.markdown("#### Sample Dataset")
sample_df = pd.DataFrame({
    "year":[2023,2023,2023],
    "source":["Hydro","Solar","Wind"],
    "generation_gwh":[500,200,150],
    "co2_tonnes":[0,0,0]
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
default_emission_factors = {"Hydro":0,"Solar":0,"Wind":0,"Geothermal":5,"Thermal":900,"Unknown":100}
baseline_emission_factor = 900  # tonnes/GWh if energy was from Thermal

# ------------------------
# Load and clean data
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

        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
        required_cols = ["year","source","generation_gwh","co2_tonnes"]
        message = None
        for col in required_cols:
            if col not in df.columns:
                message = f"Column '{col}' missing. Filling default."
                df[col] = "Unknown" if col=="source" else 0

        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['generation_gwh'] = pd.to_numeric(df['generation_gwh'], errors='coerce')
        df['co2_tonnes'] = pd.to_numeric(df.get('co2_tonnes',0), errors='coerce')

        # Calculate missing CO2
        df['co2_tonnes'] = df.apply(lambda row: row['generation_gwh']*default_emission_factors.get(row['source'],100)
                                    if pd.isna(row['co2_tonnes']) or row['co2_tonnes']==0 else row['co2_tonnes'], axis=1)

        # Calculate avoided CO2 for renewable sources
        df['avoided_co2'] = df.apply(lambda row: row['generation_gwh']*baseline_emission_factor
                                     if row['source'] in ['Hydro','Solar','Wind'] else 0, axis=1)

        df['valid'] = df['year'].notna() & df['source'].notna() & df['generation_gwh'].notna() & df['co2_tonnes'].notna()
        if df['valid'].sum()==0: return None, "No valid rows found."
        if df['valid'].sum()<len(df): message="Some rows invalid and ignored."
        df_clean = df[df['valid']].reset_index(drop=True)
        return df_clean, message
    except Exception as e:
        return None, f"Error: {e}"

uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv","xlsx"])
df, message = load_energy_data_safe(uploaded_file)
st.session_state.df = df
st.session_state.message = message

if df is None:
    st.warning(message)
    st.stop()
if message: st.warning(message)

# ------------------------
# Data preview
# ------------------------
st.subheader("Preview of cleaned data (first 20 rows)")
st.dataframe(df.head(20))

# ------------------------
# Metrics & insights
# ------------------------
total_gen = df['generation_gwh'].sum()
total_emissions = df['co2_tonnes'].sum()
total_avoided = df['avoided_co2'].sum()

trees = int(total_avoided/22)  # approx CO2 avoided per tree
cars = int(total_avoided/4.6)  # approx CO2 per car per year
homes = int(total_avoided/7.5) # approx CO2 per home per year

insights_list=[
    f"In Kenya, carbon saved this year is equivalent to planting {trees} trees.",
    f"This reduction is like taking {cars} cars off Kenyan roads.",
    f"Sustainable energy has powered approximately {homes} Kenyan homes.",
    "Renewable energy adoption in Kenya improves air quality and supports national energy goals."
]

st.subheader("💡 Key Metrics")
c1,c2,c3 = st.columns(3)
c1.metric("⚡ Total Generation (GWh)", f"{total_gen:,.0f}")
c2.metric("🌫️ Total CO₂ Emissions (tonnes)", f"{total_emissions:,.0f}")
c3.metric("🌱 Total CO₂ Avoided (tonnes)", f"{total_avoided:,.0f}")

st.subheader("💡 Insights")
for i,insight in enumerate(insights_list,1): st.markdown(f"{i}. {insight}")

# ------------------------
# Charts
# ------------------------
gen_by_source = df.groupby("source", as_index=False)["generation_gwh"].sum()
em_by_source = df.groupby("source", as_index=False)["co2_tonnes"].sum()
avoided_by_source = df.groupby("source", as_index=False)["avoided_co2"].sum()

st.subheader("📊 Energy Generation by Source")
chart_gen_source = alt.Chart(gen_by_source).mark_bar().encode(
    x="source:N", y="generation_gwh:Q", color="source:N",
    tooltip=["source","generation_gwh"]
).properties(height=400,width=700)
st.altair_chart(chart_gen_source)

st.subheader("🌫️ CO₂ Emissions by Source")
chart_em_source = alt.Chart(em_by_source).mark_bar().encode(
    x="source:N", y="co2_tonnes:Q", color="source:N",
    tooltip=["source","co2_tonnes"]
).properties(height=400,width=700)
st.altair_chart(chart_em_source)

st.subheader("🌱 CO₂ Avoided by Source")
chart_avoided_source = alt.Chart(avoided_by_source).mark_bar().encode(
    x="source:N", y="avoided_co2:Q", color="source:N",
    tooltip=["source","avoided_co2"]
).properties(height=400,width=700)
st.altair_chart(chart_avoided_source)

# ------------------------
# Annual trends + forecast
# ------------------------
annual = df.groupby("year", as_index=False).agg(
    total_generation_gwh=("generation_gwh","sum"),
    total_emissions_tonnes=("co2_tonnes","sum"),
    total_avoided_co2=("avoided_co2","sum")
).sort_values("year")

st.subheader("📈 Annual Trends")
chart_gen_annual = alt.Chart(annual).mark_line(point=True,color="#2E8B57").encode(
    x="year:Q", y="total_generation_gwh:Q", tooltip=["year","total_generation_gwh"]
)
chart_em_annual = alt.Chart(annual).mark_line(point=True,color="#FF8C00").encode(
    x="year:Q", y="total_emissions_tonnes:Q", tooltip=["year","total_emissions_tonnes"]
)
chart_avoided_annual = alt.Chart(annual).mark_line(point=True,color="#2E8B57").encode(
    x="year:Q", y="total_avoided_co2:Q", tooltip=["year","total_avoided_co2"]
)

col1,col2,col3 = st.columns(3)
col1.altair_chart(chart_gen_annual)
col2.altair_chart(chart_em_annual)
col3.altair_chart(chart_avoided_annual)

st.subheader("🔮 Quick Forecast (experimental)")
forecast_target = st.selectbox("Forecast target", options=["Total Generation (GWh)","Total CO₂ (tonnes)","Total CO₂ Avoided (tonnes)"])
n_years = st.slider("Forecast years ahead",1,10,3)

if len(annual)>=2:
    if forecast_target=="Total Generation (GWh)":
        X = annual['year'].values.reshape(-1,1); y = annual['total_generation_gwh'].values; y_label="Generation (GWh)"
    elif forecast_target=="Total CO₂ (tonnes)":
        X = annual['year'].values.reshape(-1,1); y = annual['total_emissions_tonnes'].values; y_label="CO₂ (tonnes)"
    else:
        X = annual['year'].values.reshape(-1,1); y = annual['total_avoided_co2'].values; y_label="Avoided CO₂ (tonnes)"
    model = LinearRegression(); model.fit(X,y)
    last_year = int(annual['year'].max())
    future_years = np.arange(last_year+1,last_year+1+n_years)
    preds = model.predict(future_years.reshape(-1,1))
    hist_df = pd.DataFrame({"year":annual['year'],"value":y})
    fut_df = pd.DataFrame({"year":future_years,"value":preds})
    comb = pd.concat([hist_df,fut_df],ignore_index=True)
    chart_forecast = alt.Chart(comb).mark_line(point=True,color="#6A5ACD").encode(
        x="year:Q", y=alt.Y("value:Q",title=y_label),
        tooltip=["year","value"]
    ).properties(height=350,width=700)
    st.altair_chart(chart_forecast)

# ------------------------
# PDF Generation (text only)
# ------------------------
def generate_pdf(metrics_dict, insights_list):
    buffer = BytesIO()
    c = canvas.Canvas(buffer,pagesize=letter)
    width,height = letter
    c.setFont("Helvetica-Bold",20); c.drawString(50,height-50,"Kenya Carbon Emission Tracker Report")
    y_pos = height-100
    c.setFont("Helvetica",12)
    for k,v in metrics_dict.items(): c.drawString(50,y_pos,f"{k}: {v}"); y_pos-=20
    y_pos-=10
    c.setFont("Helvetica-Bold",14); c.drawString(50,y_pos,"Insights:"); y_pos-=20
    c.setFont("Helvetica",12)
    for insight in insights_list: c.drawString(60,y_pos,f"- {insight}"); y_pos-=20
    y_pos-=10
    c.save(); buffer.seek(0)
    return buffer

metrics_dict = {
    "Total Generation (GWh)": f"{total_gen:,.0f}",
    "Total CO₂ Emissions (tonnes)": f"{total_emissions:,.0f}",
    "Total CO₂ Avoided (tonnes)": f"{total_avoided:,.0f}"
}

if st.button("📄 Generate & Download PDF Report"):
    pdf_buffer = generate_pdf(metrics_dict, insights_list)
    st.download_button(label="📥 Download PDF", data=pdf_buffer, file_name="kenya_carbon_report.pdf", mime="application/pdf")
