# carbon_loader.py
import pandas as pd

# Emission factors in tonnes CO₂ per GWh (IPCC-based averages, approximate)
EMISSION_FACTORS = {
    "Geothermal": 45,   # 45 tCO₂/GWh
    "Hydro": 5,         # near zero, but allow small value
    "Solar": 20,        # manufacturing + lifecycle
    "Wind": 15          # manufacturing + lifecycle
}

def load_energy_data(uploaded_file):
    """
    Loads and cleans uploaded energy data (CSV/Excel).
    Expects columns: Month, Plant, Geothermal, Hydro, Solar, Wind
    Returns a cleaned DataFrame and a message if issues exist.
    """
    try:
        if isinstance(uploaded_file, str):  # demo sample file
            if uploaded_file.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        else:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
    except Exception as e:
        return None, f"❌ Failed to read file: {e}"

    # Normalize column names (strip, lowercase)
    df.columns = [c.strip().lower() for c in df.columns]

    required = ["month", "plant", "geothermal", "hydro", "solar", "wind"]
    for col in required:
        if col not in df.columns:
            return None, f"❌ Missing required column: {col}. Please follow the upload guidelines."

    # Handle missing values
    for col in ["geothermal", "hydro", "solar", "wind"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Convert Month to datetime, add year
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    if df["month"].isnull().any():
        return None, "❌ Invalid date format in 'Month' column. Use YYYY-MM-DD."

    df["year"] = df["month"].dt.year

    # Reshape to long format (source-level rows)
    df_long = df.melt(
        id_vars=["month", "year", "plant"],
        value_vars=["geothermal", "hydro", "solar", "wind"],
        var_name="source",
        value_name="generation_gwh"
    )

    # Compute emissions
    df_long["co2_tonnes"] = df_long.apply(
        lambda row: row["generation_gwh"] * EMISSION_FACTORS.get(row["source"].title(), 0),
        axis=1
    )

    # Normalize text formatting
    df_long["source"] = df_long["source"].str.title()
    df_long["plant"] = df_long["plant"].astype(str).str.title()

    return df_long, None


def human_equivalents(co2_tonnes):
    """
    Convert CO₂ tonnes into human-readable equivalents.
    Approximate multipliers based on EPA data.
    """
    trees = int(co2_tonnes / 0.068)   # 1 tree absorbs ~0.068 tonnes/year
    cars = int(co2_tonnes / 4.6)      # 1 car ~4.6 tonnes/year
    homes = int(co2_tonnes / 7.5)     # 1 home ~7.5 tonnes/year
    return {"trees": trees, "cars": cars, "homes": homes}

