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
# Session state
# ------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.message = None

# ------------------------
# Safe loader
# ------------------------
def load_energy_data_safe(uploaded_file=None):
    try:
        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        else:
            # fallback sample data
            df = sample_df.copy()

        # Ensure required columns
        required_cols = ["year", "source", "generation_gwh", "co2_tonnes"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        message = None
        if missing_cols:
            message = f"Missing columns: {', '.join(missing_cols)}. These will be filled with defaults."
            for col in missing_cols:
                if col == "source":
                    df[col] = "Unknown"
                else:
                    df[col] = 0

        # Clean columns
        df['source'] = df['source'].astype(str).str.title()
        for col in ["year", "generation_gwh", "co2_tonnes"]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.strip(), errors='coerce').fillna(0)

        df = df[df['year'] > 0]
        df = df[df['source'].notna()]
        df = df.reset_index(drop=True)
        if df.empty:
            return None, "Dataset is empty or invalid after cleaning."
        return df, message
    except Exception as e:
        return None, f"Error loading dataset: {e}"

# ------------------------
# Upload file
# ------------------------
uploaded_file = st.sidebar.file_uploader("Upload Energy Data (CSV or Excel)", type=["csv", "xlsx"])
if uploaded_file:
    df, message = load_energy_data_safe(uploaded_file)
else:
    df, message = load_energy_data_safe()

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
# Metrics
# ------------------------
total_gen = gen_by_source['generation_gwh'].sum() if not gen_by_source.empty else 0
total_emissions = em_by_source['co2_tonnes'].sum() if not em_by_source.empty else 0
equiv = human_equivalents(total_emissions) if total_emissions > 0 else {"trees":0,"cars":0,"homes":0}

st.subheader("📌 Key Metrics")
c1, c2, c3 = st.columns(3)
c1.metric("⚡ Total Generation (GWh)", f"{total_gen:,.0f}")
c2.metric("🌫️ Total CO₂ (tonnes)", f"{total_emissions:,.0f}")
c3.metric("🌳 Tree Equivalent", f"{equiv['trees']:,} trees")
st.markdown(f"Other equivalents: {equiv['cars']:,} cars off the road per year • {equiv['homes']:,} homes powered per year")

# ------------------------
# Forecast
# ------------------------
st.subheader("🔮 Quick Forecast (experimental)")
forecast_target = st.selectbox("Forecast target", options=["Total Generation (GWh)", "Total CO₂ (tonnes)"])
n_years = st.slider("Forecast years ahead", 1, 10, 3)

chart_forecast = None
if len(annual) >= 2:
    if forecast_target.startswith("Total Generation"):
        X = annual['year'].values.reshape(-1,1)
        y = annual['total_generation_gwh'].values
        y_label = "Generation (GWh)"
    else:
        X = annual['year'].values.reshape(-1,1)
        y = annual['total_emissions_tonnes'].values
        y_label = "CO₂ (tonnes)"
    model = LinearRegression()
    model.fit(X, y)
    last_year = int(annual['year'].max())
    future_years = np.arange(last_year+1, last_year+1+n_years)
    preds = model.predict(future_years.reshape(-1,1))
    hist_df = pd.DataFrame({"year": annual['year'], "value": y})
    fut_df = pd.DataFrame({"year": future_years, "value": preds})
    comb = pd.concat([hist_df, fut_df], ignore_index=True)
    chart_forecast = alt.Chart(comb).mark_line(point=True, color="#6A5ACD").encode(
        x="year:Q",
        y=alt.Y("value:Q", title=y_label),
        tooltip=[alt.Tooltip("year:Q"), alt.Tooltip("value:Q", format=",.0f")]
    ).properties(height=350, width=700)
    st.altair_chart(chart_forecast)
else:
    st.info("Not enough data for forecast.")

# ------------------------
# Insights
# ------------------------
insights_list = [
    f"Carbon saved this year is equivalent to planting {equiv['trees']} trees.",
    f"Emissions reduction is equivalent to taking {equiv['cars']} cars off the road.",
    f"Sustainable energy has powered {equiv['homes']} homes.",
    "Using renewable energy reduces emissions, improves air quality, and supports Kenya’s sustainable energy vision."
]
st.subheader("💡 Insights")
for i, insight in enumerate(insights_list,1):
    st.markdown(f"{i}. {insight}")

# ------------------------
# PDF Generation
# ------------------------
def save_chart_image(chart):
    buf = BytesIO()
    chart.save(buf, format="png", scale_factor=2)
    buf.seek(0)
    return buf

def generate_pdf(metrics_dict, insights_list, charts):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height-50, "Carbon Emission Tracker Report")
    y_pos = height-100

    # Metrics
    c.setFont("Helvetica", 12)
    for k,v in metrics_dict.items():
        c.drawString(50, y_pos, f"{k}: {v}")
        y_pos -= 20
    y_pos -= 10

    # Insights
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, "Insights:")
    y_pos -= 20
    c.setFont("Helvetica", 12)
    for insight in insights_list:
        c.drawString(60, y_pos, f"- {insight}")
        y_pos -= 20
    y_pos -= 10

    # Charts
    for chart in charts:
        img_buf = save_chart_image(chart)
        img = Image.open(img_buf)
        img_reader = ImageReader(img)
        if y_pos < 250:
            c.showPage()
            y_pos = height - 50
        c.drawImage(img_reader, 50, y_pos-250, width=500, height=250)
        y_pos -= 270

    c.save()
    buffer.seek(0)
    return buffer

metrics_dict = {
    "Total Generation (GWh)": f"{total_gen:,.0f}",
    "Total CO₂ (tonnes)": f"{total_emissions:,.0f}",
    "Tree Equivalent": f"{equiv['trees']:,} trees"
}

if st.button("📄 Generate & Download PDF Report"):
    charts_to_save = [chart_gen_source, chart_em_source]
    if chart_forecast:
        charts_to_save.append(chart_forecast)
    pdf_buffer = generate_pdf(metrics_dict, insights_list, charts_to_save)
    st.download_button(
        label="📥 Download PDF",
        data=pdf_buffer,
        file_name="carbon_emission_report.pdf",
        mime="application/pdf"
    )
