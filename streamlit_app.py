import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PIL import Image

# ------------------------
# Helper functions
# ------------------------

def load_and_clean_data(uploaded_file=None):
    sample_data = pd.DataFrame({
        "year":[2023,2023,2023],
        "source":["Hydro","Solar","Wind"],
        "generation_gwh":[500,200,150],
        "co2_tonnes":[0,0,0]
    })

    try:
        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        else:
            df = sample_data.copy()

        # Standardize column names
        df.columns = [c.lower().strip().replace(" ","_") for c in df.columns]

        # Check required columns
        required_cols = ["year","source","generation_gwh"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            return None, f"Missing columns: {', '.join(missing_cols)}"

        # Ensure numeric
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['generation_gwh'] = pd.to_numeric(df['generation_gwh'], errors='coerce')
        df['co2_tonnes'] = pd.to_numeric(df.get('co2_tonnes',0), errors='coerce').fillna(0)

        # Remove invalid rows
        df['valid'] = df['year'].notna() & df['source'].notna() & df['generation_gwh'].notna()
        if df['valid'].sum() == 0:
            return None, "No valid rows found. Please correct your dataset."
        elif df['valid'].sum() < len(df):
            msg = f"{len(df)-df['valid'].sum()} invalid row(s) ignored."
        else:
            msg = None
        df = df[df['valid']].reset_index(drop=True)

        # Default CO2 emission factors (tonnes per GWh)
        factors = {"Hydro":0,"Solar":0,"Wind":0,"Geothermal":5,"Thermal":900,"Unknown":100}
        df['co2_tonnes'] = df.apply(lambda r: r['generation_gwh']*factors.get(r['source'],100)
                                    if r['co2_tonnes']==0 else r['co2_tonnes'], axis=1)

        # Avoided CO2 for renewables
        baseline_factor = 900
        df['avoided_co2'] = df.apply(lambda r: r['generation_gwh']*baseline_factor
                                     if r['source'] in ["Hydro","Solar","Wind"] else 0, axis=1)

        return df, msg

    except Exception as e:
        return None, f"Error loading data: {e}"


def generate_pdf_with_charts(metrics_dict, insights_list, chart_images):
    buffer = BytesIO()
    c = canvas.Canvas(buffer,pagesize=letter)
    width,height = letter

    c.setFont("Helvetica-Bold",20)
    c.drawString(50,height-50,"Kenya Carbon Emission Tracker Report")
    y_pos = height-100

    # Metrics
    c.setFont("Helvetica",12)
    for k,v in metrics_dict.items():
        c.drawString(50,y_pos,f"{k}: {v}")
        y_pos -= 20
    y_pos -= 10

    # Insights
    c.setFont("Helvetica-Bold",14)
    c.drawString(50,y_pos,"Insights:")
    y_pos -= 20
    c.setFont("Helvetica",12)
    for insight in insights_list:
        c.drawString(60,y_pos,f"- {insight}")
        y_pos -= 20
    y_pos -= 10

    # Charts
    for img in chart_images:
        if y_pos < 300:
            c.showPage()
            y_pos = height-50
        img_reader = ImageReader(img)
        c.drawImage(img_reader,50,y_pos-250,width=500,height=250)
        y_pos -= 270

    c.save()
    buffer.seek(0)
    return buffer

def altair_chart_to_image(chart):
    buf = BytesIO()
    chart.save(buf, format="png", scale_factor=2)
    buf.seek(0)
    return Image.open(buf)

# ------------------------
# Page config
# ------------------------
st.set_page_config(page_title="Kenya Carbon Emission Tracker", page_icon="🌍", layout="wide")
st.title("🌍 Kenya Carbon Emission Tracker")
st.markdown("### Visualize, estimate, and explore energy generation impact in Kenya")

# ------------------------
# Sidebar
# ------------------------
st.sidebar.title("Kenya Carbon Emission Tracker")
st.sidebar.markdown("Upload your CSV/Excel dataset or use the sample data below.")

uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv","xlsx"])
df, msg = load_and_clean_data(uploaded_file)

if df is None:
    st.error(msg)
    st.stop()
elif msg:
    st.warning(msg)

st.subheader("Preview of data")
st.dataframe(df.head(20))

# ------------------------
# Metrics
# ------------------------
total_gen = df['generation_gwh'].sum()
total_emissions = df['co2_tonnes'].sum()
total_avoided = df['avoided_co2'].sum()

trees = int(total_avoided/22)
cars = int(total_avoided/4.6)
homes = int(total_avoided/7.5)

insights = [
    f"In Kenya, carbon saved this year is equivalent to planting {trees} trees.",
    f"This reduction is like taking {cars} cars off Kenyan roads.",
    f"Sustainable energy has powered approx. {homes} homes.",
    "Renewable energy adoption in Kenya improves air quality and supports national energy goals."
]

st.subheader("💡 Key Metrics")
c1,c2,c3 = st.columns(3)
c1.metric("⚡ Total Generation (GWh)", f"{total_gen:,.0f}")
c2.metric("🌫️ Total CO₂ Emissions (tonnes)", f"{total_emissions:,.0f}")
c3.metric("🌱 Total CO₂ Avoided (tonnes)", f"{total_avoided:,.0f}")

st.subheader("💡 Insights")
for i,ins in enumerate(insights,1): st.markdown(f"{i}. {ins}")

# ------------------------
# Charts
# ------------------------
gen_by_source = df.groupby("source",as_index=False)["generation_gwh"].sum()
em_by_source = df.groupby("source",as_index=False)["co2_tonnes"].sum()
avoided_by_source = df.groupby("source",as_index=False)["avoided_co2"].sum()

st.subheader("📊 Energy Generation by Source")
chart_gen = alt.Chart(gen_by_source).mark_bar().encode(
    x="source:N", y="generation_gwh:Q", color="source:N",
    tooltip=["source","generation_gwh"]
).properties(height=400,width=700)
st.altair_chart(chart_gen)

st.subheader("🌫️ CO₂ Emissions by Source")
chart_em = alt.Chart(em_by_source).mark_bar().encode(
    x="source:N", y="co2_tonnes:Q", color="source:N",
    tooltip=["source","co2_tonnes"]
).properties(height=400,width=700)
st.altair_chart(chart_em)

st.subheader("🌱 CO₂ Avoided by Source")
chart_avoided = alt.Chart(avoided_by_source).mark_bar().encode(
    x="source:N", y="avoided_co2:Q", color="source:N",
    tooltip=["source","avoided_co2"]
).properties(height=400,width=700)
st.altair_chart(chart_avoided)

# ------------------------
# Annual trends + forecast
# ------------------------
annual = df.groupby("year",as_index=False).agg(
    total_generation_gwh=("generation_gwh","sum"),
    total_emissions_tonnes=("co2_tonnes","sum"),
    total_avoided_co2=("avoided_co2","sum")
).sort_values("year")

st.subheader("📈 Annual Trends")
chart_gen_year = alt.Chart(annual).mark_line(point=True,color="#2E8B57").encode(
    x="year:Q", y="total_generation_gwh:Q", tooltip=["year","total_generation_gwh"]
)
chart_em_year = alt.Chart(annual).mark_line(point=True,color="#FF8C00").encode(
    x="year:Q", y="total_emissions_tonnes:Q", tooltip=["year","total_emissions_tonnes"]
)
chart_avoided_year = alt.Chart(annual).mark_line(point=True,color="#2E8B57").encode(
    x="year:Q", y="total_avoided_co2:Q", tooltip=["year","total_avoided_co2"]
)

col1,col2,col3 = st.columns(3)
col1.altair_chart(chart_gen_year)
col2.altair_chart(chart_em_year)
col3.altair_chart(chart_avoided_year)

st.subheader("🔮 Quick Forecast (experimental)")
forecast_target = st.selectbox("Forecast target", options=["Total Generation (GWh)","Total CO₂ (tonnes)","Total CO₂ Avoided (tonnes)"])
n_years = st.slider("Forecast years ahead",1,10,3)

if len(annual) >= 2:
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
# Generate PDF with charts
# ------------------------
if st.button("📄 Generate & Download PDF"):
    chart_images = [altair_chart_to_image(c) for c in [chart_gen,chart_em,chart_avoided]]
    metrics_dict = {
        "Total Generation (GWh)":f"{total_gen:,.0f}",
        "Total CO₂ Emissions (tonnes)":f"{total_emissions:,.0f}",
        "Total CO₂ Avoided (tonnes)":f"{total_avoided:,.0f}"
    }
    pdf_buffer = generate_pdf_with_charts(metrics_dict, insights, chart_images)
    st.download_button("📥 Download PDF", pdf_buffer, "kenya_carbon_report.pdf","application/pdf")
