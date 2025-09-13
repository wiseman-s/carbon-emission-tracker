import pandas as pd

def load_energy_data(uploaded_file):
    try:
        # Read CSV or Excel
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Normalize column names
        df.columns = df.columns.str.strip().str.title()

        # Ensure required columns exist, add if missing
        required_cols = ["Month", "Plant", "Geothermal", "Hydro", "Solar", "Wind"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0

        # Convert to long format
        df = df.melt(
            id_vars=["Month", "Plant"],
            value_vars=["Geothermal", "Hydro", "Solar", "Wind"],
            var_name="source",
            value_name="generation_gwh"
        )

        # Handle missing / invalid dates
        df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
        df = df.dropna(subset=["Month"])
        df["year"] = df["Month"].dt.year

        # Dummy emission factors (tonnes per GWh)
        emission_factors = {"Geothermal": 45, "Hydro": 5, "Solar": 3, "Wind": 2}
        df["co2_tonnes"] = df.apply(
            lambda x: x["generation_gwh"] * emission_factors.get(x["source"], 0), axis=1
        )

        return df, None

    except Exception as e:
        return None, f"Error loading file: {e}"


def human_equivalents(emissions_tonnes):
    """Convert emissions saved into human-understandable equivalents."""
    trees = int(emissions_tonnes * 0.68)    # ~0.68 trees offset per tonne CO2/year
    cars = int(emissions_tonnes / 4.6)      # Avg car ~4.6 tonnes/year
    homes = int(emissions_tonnes / 10)      # Rough estimate: 10 tonnes/home/year
    return {"trees": trees, "cars": cars, "homes": homes}
