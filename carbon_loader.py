# carbon_loader.py
import pandas as pd

# Default emission factors in gCO2/kWh
EMISSION_DEFAULTS = {
    "thermal": 800,   # generic fossil
    "diesel": 730,
    "coal": 1000,
    "oil": 850,
    "gas": 490,
    "hydro": 24,
    "geothermal": 45,
    "solar": 50,
    "wind": 12,
    "nuclear": 15,
}

def load_energy_data(file):
    """
    Reads a CSV/Excel and returns a cleaned dataframe with:
    - year (int)
    - source (str)
    - generation_gwh (float)
    - emission_factor_gco2_per_kwh (float)
    - co2_tonnes (float)
    """

    try:
        if isinstance(file, str):
            if file.endswith(".csv"):
                df = pd.read_csv(file)
            elif file.endswith(".xlsx"):
                df = pd.read_excel(file)
            else:
                return None, "Unsupported file type."
        else:
            try:
                df = pd.read_csv(file)
            except Exception:
                file.seek(0)
                df = pd.read_excel(file)

        df.columns = [c.strip().lower() for c in df.columns]

        required = {"year", "source", "generation_gwh"}
        if not required.issubset(df.columns):
            return None, f"Missing required columns: {required - set(df.columns)}"

        # ensure correct dtypes
        df["year"] = df["year"].astype(int)
        df["source"] = df["source"].astype(str).str.strip()
        df["generation_gwh"] = pd.to_numeric(df["generation_gwh"], errors="coerce").fillna(0)

        # add emission factor if missing
        if "emission_factor_gco2_per_kwh" not in df.columns:
            df["emission_factor_gco2_per_kwh"] = df["source"].str.lower().map(EMISSION_DEFAULTS).fillna(0)

        # compute emissions
        df["co2_tonnes"] = df["generation_gwh"] * 1e6 * df["emission_factor_gco2_per_kwh"] / 1e9
        # Explanation: 1 GWh = 1e6 kWh, multiply by gCO2/kWh, then /1e6 for tonnes (g→tonnes = 1e-6)

        return df, None
    except Exception as e:
        return None, f"Error loading file: {e}"

def human_equivalents(emissions_tonnes: float):
    """
    Convert emissions (tonnes of CO2) into equivalent trees, cars, homes.
    Sources: EPA & academic averages.
    """
    trees = int(emissions_tonnes / 0.021)   # 1 tree absorbs ~21 kg CO2/year
    cars = int(emissions_tonnes / 4.6)      # 1 car ~4.6 t CO2/year
    homes = int(emissions_tonnes / 7.5)     # 1 home ~7.5 t CO2/year

    return {"trees": trees, "cars": cars, "homes": homes}
