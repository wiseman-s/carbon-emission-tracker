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
from altair_saver import save as alt_save

# ------------------------
# Page config
# ------------------------
st.set_page_config(page_title="Kenya Carbon Emission Tracker", page_icon="🌍", layout="wide")

# ------------------------
# Heading
# ------------------------
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
# Per-row CO2 avoided with invalid highlight
# ------------------------
st.subheader("🌱 Impact per Entry (CO₂ Avoided)")

def highlight_invalid(row):
    return ['background-color: #ffcccc' if not row.valid else '' for _ in row]

df_display = df.copy()
df_display['avoided_co2'] = df_display['avoided_co2'].round(0)
st.dataframe(df_display[['year','source','generation_gwh','co2_tonnes','avoided_co2','valid']].style.apply(highlight_invalid, axis=1))

# ------------------------
# Aggregations
# ------------------------
gen_by_source = df.groupby("source", as_index=False)["generation_gwh"].sum().sort_values("generation_gwh", ascending=False)
em_by_source = df.groupby("source", as_index=False)["co2_tonnes"].sum().sort_values("co2_tonnes", ascending=False)
avoided_by_source = df.groupby("source", as_index=False)["avoided_co2"].sum().sort_values("avoided_co2", ascending=False)
annual = df.groupby("year", as_index=False).agg(
    total_generation_gwh=pd.NamedAgg(column="generation_gwh", aggfunc="sum"),
    total_emissions_tonnes=pd.NamedAgg(column="co2_tonnes", aggfunc="sum"),
    total_avoided_co2=pd.NamedAgg(column="avoided_co2", aggfunc="sum")
).sort_values("year")
annual['year'] = annual['year'].astype(int)

# ------------------------
# Charts
# ------------------------
colors = {"Hydro":"#1f77b4","Solar":"#ff7f0e","Wind":"#2ca02c","Geothermal":"#d62728","Thermal":"#9467bd"}

st.subheader("📊 Energy Generation by Source (Total)")
chart_gen_source = alt.Chart(gen_by_source).mark_bar().encode(
    x=alt.X("source:N", sort='-y', title="Energy Source"),
    y=alt.Y("generation_gwh:Q", title="Total Generation (GWh)"),
    color=alt.Color("source:N", scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values())), legend=None),
    tooltip=[alt.Tooltip("source:N"), alt.Tooltip("generation_gwh:Q", format=",.2f")]
).properties(height=400, width=700)
st.altair_chart(chart_gen_source)

st.subheader("🌫️ CO₂ Emissions by Source (Total)")
chart_em_source = alt.Chart(em_by_source).mark_bar().encode(
    x=alt.X("source:N", sort='-y', title="Energy Source"),
    y=alt.Y("co2_tonnes:Q", title="Total CO₂ Emissions (tonnes)"),
    color=alt.Color("source:N", scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values())), legend=None),
    tooltip=[alt.Tooltip("source:N"), alt.Tooltip("co2_tonnes:Q", format=",.0f")]
).properties(height=400, width=700)
st.altair_chart(chart_em_source)

st.subheader("🌱 Avoided CO₂ by Source (Total)")
chart_avoided_source = alt.Chart(avoided_by_source).mark_bar().encode(
    x=alt.X("source:N", sort='-y', title="Energy Source"),
    y=alt.Y("avoided_co2:Q", title="Avoided CO₂ (tonnes)"),
    color=alt.Color("source:N", scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values())), legend=None),
    tooltip=[alt.Tooltip("source:N"), alt.Tooltip("avoided_co2:Q", format=",.0f")]
).properties(height=400, width=700)
st.altair_chart(chart_avoided_source)

st.subheader("📈 Annual Trends")
chart_gen_annual = alt.Chart(annual).mark_line(point=True, color="#2E8B57").encode(
    x="year:Q", y="total_generation_gwh:Q",
    tooltip=[alt.Tooltip("year:Q"), alt.Tooltip("total_generation_gwh:Q", format=",.0f")]
).properties(height=300, width=600)
chart_em_annual = alt.Chart(annual).mark_line(point=True, color="#FF8C00").encode(
    x="year:Q", y="total_emissions_tonnes:Q",
    tooltip=[alt.Tooltip("year:Q"), alt.Tooltip("total_emissions_tonnes:Q", format=",.0f")]
).properties(height=300, width=600)
chart_avoided_annual = alt.Chart(annual).mark_line(point=True, color="#6A5ACD").encode(
    x="year:Q", y="total_avoided_co2:Q",
    tooltip=[alt.Tooltip("year:Q"), alt.Tooltip("total_avoided_co2:Q", format=",.0f")]
).properties(height=300, width=600)
col1, col2, col3 = st.columns(3)
col1.altair_chart(chart_gen_annual)
col2.altair_chart(chart_em_annual)
col3.altair_chart(chart_avoided_annual)

# ------------------------
# Key metrics
# ------------------------
st.subheader("📌 Key Metrics")
total_gen = gen_by_source['generation_gwh'].sum()
total_emissions = em_by_source['co2_tonnes'].sum()
total_avoided = avoided_by_source['avoided_co2'].sum()
c1, c2, c3 = st.columns(3)
c1.metric("⚡ Total Generation (GWh)", f"{total_gen:,.0f}")
c2.metric("🌫️ Total CO₂ Emissions (tonnes)", f"{total_emissions:,.0f}")
c3.metric("🌱 Total CO₂ Avoided (tonnes)", f"{total_avoided:,.0f}")

# ------------------------
# Quick forecast
# ------------------------
st.subheader("🔮 Quick Forecast (experimental)")
forecast_target = st.selectbox("Forecast target", options=["Total Generation (GWh)", "Total CO₂ (tonnes)","Total CO₂ Avoided (tonnes)"])
n_years = st.slider("Forecast years ahead", 1, 10, 3)

if len(annual) >= 2:
    if forecast_target=="Total Generation (GWh)":
        X = annual['year'].values.reshape(-1,1); y=annual['total_generation_gwh'].values; y_label="Generation (GWh)"
    elif forecast_target=="Total CO₂ (tonnes)":
        X = annual['year'].values.reshape(-1,1); y=annual['total_emissions_tonnes'].values; y_label="CO₂ (tonnes)"
    else:
        X = annual['year'].values.reshape(-1,1); y=annual['total_avoided_co2'].values; y_label="Avoided CO₂ (tonnes)"
    model = LinearRegression(); model.fit(X,y)
    last_year = int(annual['year'].max())
    future_years = np.arange(last_year+1,last_year+1+n_years)
    preds = model.predict(future_years.reshape(-1,1))
    hist_df = pd.DataFrame({"year":annual['year'],"value":y})
    fut_df = pd.DataFrame({"year":future_years,"value":preds})
    comb = pd.concat([hist_df,fut_df],ignore_index=True)
    chart_forecast = alt.Chart(comb).mark_line(point=True,color="#6A5ACD").encode(
        x="year:Q", y=alt.Y("value:Q", title=y_label),
        tooltip=[alt.Tooltip("year:Q"), alt.Tooltip("value:Q",format=",.0f")]
    ).properties(height=350,width=700)
    st.altair_chart(chart_forecast)

# ------------------------
# Kenya-focused insights
# ------------------------
trees = int(total_avoided/22)  # 22 tonnes/year per tree approx
cars = int(total_avoided/4.6)  # 4.6 tonnes CO2/year per car approx
homes = int(total_avoided/7.5) # 7.5 tonnes CO2/year per home approx
insights_list=[
    f"In Kenya, carbon saved this year is equivalent to planting {trees} trees.",
    f"This reduction is like taking {cars} cars off Kenyan roads.",
    f"Sustainable energy has powered approximately {homes} Kenyan homes.",
    "Renewable energy adoption in Kenya improves air quality and supports national energy goals."
]
st.subheader("💡 Insights")
for i,insight in enumerate(insights_list,1):
    st.markdown(f"{i}. {insight}")

# ------------------------
# PDF Generation
# ------------------------
def generate_pdf(metrics_dict, insights_list, charts):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold",20); c.drawString(50,height-50,"Kenya Carbon Emission Tracker Report")
    y_pos = height-100

    # Metrics
    c.setFont("Helvetica",12)
    for k,v in metrics_dict.items(): c.drawString(50,y_pos,f"{k}: {v}"); y_pos-=20
    y_pos-=10

    # Insights
    c.setFont("Helvetica-Bold",14); c.drawString(50,y_pos,"Insights:"); y_pos-=20
    c.setFont("Helvetica",12)
    for insight in insights_list: c.drawString(60,y_pos,f"- {insight}"); y_pos-=20
    y_pos-=10

    # Charts
    for chart in charts:
        img_buf = save_chart_image(chart)
        if img_buf is None: continue
        img = Image.open(img_buf)
        img_reader = ImageReader(img)
        if y_pos<250: c.showPage(); y_pos=height-50
        c.drawImage(img_reader,50,y_pos-250,width=500,height=250)
        y_pos-=270

    c.save(); buffer.seek(0)
    return buffer

def save_chart_image(chart):
    buf = BytesIO()
    try: alt_save(chart, fp=buf, fmt="png", scale_factor=2); buf.seek(0); return buf
    except Exception as e: st.error(f"Chart PNG generation failed: {e}"); return None

metrics_dict = {
    "Total Generation (GWh)":f"{total_gen:,.0f}",
    "Total CO₂ Emissions (tonnes)":f"{total_emissions:,.0f}",
    "Total CO₂ Avoided (tonnes)":f"{total_avoided:,.0f}"
}

charts_to_save = [chart_gen_source, chart_em_source, chart_avoided_source]
if 'chart_forecast' in locals(): charts_to_save.append(chart_forecast)

if st.button("📄 Generate & Download PDF Report"):
    pdf_buffer = generate_pdf(metrics_dict, insights_list, charts_to_save)
    st.download_button(label="📥 Download PDF", data=pdf_buffer, file_name="kenya_carbon_report.pdf", mime="application/pdf")
