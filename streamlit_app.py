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
st.set_page_config(page_title="Carbon Emission Tracker", page_icon="🌍", layout="wide")

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
- `year` numeric (e.g., 2023)
- `source` in Hydro, Solar, Wind, Geothermal, Thermal
- `generation_gwh` numeric
- Missing `co2_tonnes` will be calculated automatically
""")
st.sidebar.markdown("#### Sample Dataset")
sample_df = pd.DataFrame({
    "year":[2023,2023,2023],
    "source":["Hydro","Solar","Wind"],
    "generation_gwh":[500,200,150],
    "co2_tonnes":[0,50,30]
})
st.sidebar.dataframe(sample_df)

# ------------------------
# Session state
# ------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.message = None

# ------------------------
# Default CO2 emission factors
# ------------------------
default_emission_factors = {"Hydro":0,"Solar":0,"Wind":0,"Geothermal":5,"Thermal":900,"Unknown":100}

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

        df['co2_tonnes'] = df.apply(lambda row: row['generation_gwh']*default_emission_factors.get(row['source'],100)
                                    if pd.isna(row['co2_tonnes']) or row['co2_tonnes']==0 else row['co2_tonnes'], axis=1)
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

st.subheader("📈 Annual Trends")
chart_gen_annual = alt.Chart(annual).mark_line(point=True, color="#2E8B57").encode(
    x="year:Q", y="total_generation_gwh:Q",
    tooltip=[alt.Tooltip("year:Q"), alt.Tooltip("total_generation_gwh:Q", format=",.0f")]
).properties(height=300, width=600)
chart_em_annual = alt.Chart(annual).mark_line(point=True, color="#FF8C00").encode(
    x="year:Q", y="total_emissions_tonnes:Q",
    tooltip=[alt.Tooltip("year:Q"), alt.Tooltip("total_emissions_tonnes:Q", format=",.0f")]
).properties(height=300, width=600)
col1, col2 = st.columns(2)
col1.altair_chart(chart_gen_annual)
col2.altair_chart(chart_em_annual)

# ------------------------
# Key metrics
# ------------------------
st.subheader("📌 Key Metrics")
total_gen = gen_by_source['generation_gwh'].sum()
total_emissions = em_by_source['co2_tonnes'].sum()
equiv = human_equivalents(total_emissions)
c1, c2, c3 = st.columns(3)
c1.metric("⚡ Total Generation (GWh)", f"{total_gen:,.0f}")
c2.metric("🌫️ Total CO₂ (tonnes)", f"{total_emissions:,.0f}")
c3.metric("🌳 Tree Equivalent", f"{equiv['trees']:,} trees")

# ------------------------
# Forecast
# ------------------------
st.subheader("🔮 Quick Forecast (experimental)")
forecast_target = st.selectbox("Forecast target", options=["Total Generation (GWh)","Total CO₂ (tonnes)"])
n_years = st.slider("Forecast years ahead",1,10,3)
if len(annual)>=2:
    if forecast_target.startswith("Total Generation"):
        X = annual['year'].values.reshape(-1,1)
        y = annual['total_generation_gwh'].values
        y_label="Generation (GWh)"
    else:
        X = annual['year'].values.reshape(-1,1)
        y = annual['total_emissions_tonnes'].values
        y_label="CO₂ (tonnes)"
    model = LinearRegression()
    model.fit(X,y)
    last_year=int(annual['year'].max())
    future_years=np.arange(last_year+1,last_year+1+n_years)
    preds=model.predict(future_years.reshape(-1,1))
    hist_df=pd.DataFrame({"year":annual['year'],"value":y})
    fut_df=pd.DataFrame({"year":future_years,"value":preds})
    comb=pd.concat([hist_df,fut_df],ignore_index=True)
    chart_forecast=alt.Chart(comb).mark_line(point=True,color="#6A5ACD").encode(
        x="year:Q", y=alt.Y("value:Q",title=y_label),
        tooltip=[alt.Tooltip("year:Q"), alt.Tooltip("value:Q",format=",.0f")]
    ).properties(height=350,width=700)
    st.altair_chart(chart_forecast)

# ------------------------
# Insights
# ------------------------
insights_list=[
    f"Carbon saved this year is equivalent to planting {equiv['trees']} trees.",
    f"Emissions reduction is equivalent to taking {equiv['cars']} cars off the road.",
    f"Sustainable energy has powered {equiv['homes']} homes.",
    "Using renewable energy reduces emissions, improves air quality, and supports Kenya’s sustainable energy vision."
]
st.subheader("💡 Insights")
for i,insight in enumerate(insights_list,1):
    st.markdown(f"{i}. {insight}")

# ------------------------
# PDF generation (Streamlit Cloud compatible)
# ------------------------
def generate_pdf(metrics_dict, insights_list):
    buffer=BytesIO()
    c=canvas.Canvas(buffer,pagesize=letter)
    width,height=letter
    c.setFont("Helvetica-Bold",20)
    c.drawString(50,height-50,"Carbon Emission Tracker Report")
    y_pos=height-100
    # Metrics
    c.setFont("Helvetica",12)
    for k,v in metrics_dict.items():
        c.drawString(50,y_pos,f"{k}: {v}")
        y_pos-=20
    y_pos-=10
    # Insights
    c.setFont("Helvetica-Bold",14)
    c.drawString(50,y_pos,"Insights:")
    y_pos-=20
    c.setFont("Helvetica",12)
    for insight in insights_list:
        c.drawString(60,y_pos,f"- {insight}")
        y_pos-=20
    c.save()
    buffer.seek(0)
    return buffer

metrics_dict={
    "Total Generation (GWh)":f"{total_gen:,.0f}",
    "Total CO₂ (tonnes)":f"{total_emissions:,.0f}",
    "Tree Equivalent":f"{equiv['trees']:,} trees"
}

if st.button("📄 Generate & Download PDF Report"):
    pdf_buffer=generate_pdf(metrics_dict, insights_list)
    st.download_button(
        label="📥 Download PDF",
        data=pdf_buffer,
        file_name="carbon_emission_report.pdf",
        mime="application/pdf"
    )
