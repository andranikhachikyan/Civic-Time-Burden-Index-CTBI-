import os
import json
import pandas as pd
import plotly.express as px
import requests

CTBI_PATH = "outputs/ctbi_master_with_capped.csv"
OUT_HTML = "outputs/maps/ctbi_toggle_map.html"
GEOJSON_CACHE = "data/us_counties_fips.geojson"

COUNTIES_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

def load_counties_geojson():
    os.makedirs("data", exist_ok=True)

    if os.path.exists(GEOJSON_CACHE):
        with open(GEOJSON_CACHE, "r") as f:
            return json.load(f)

    r = requests.get(COUNTIES_URL, timeout=60)
    r.raise_for_status()
    geojson = r.json()

    with open(GEOJSON_CACHE, "w") as f:
        json.dump(geojson, f)

    return geojson

def main():
    os.makedirs("outputs/maps", exist_ok=True)

    df = pd.read_csv(CTBI_PATH, dtype={"GEOID": str})
    df["GEOID"] = df["GEOID"].str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(5)

    metric_map = {
        "CTBI (capped)": "CTBI_capped",
        "Commute burden (z)": "commute_z2",
        "Grocery low-access (z)": "grocery_z2",
        "Hospital access (z)": "hospital_z2",
    }

    missing = [c for c in metric_map.values() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {CTBI_PATH}: {missing}\nAvailable: {list(df.columns)}")

    long = df.melt(
        id_vars=["GEOID"],
        value_vars=list(metric_map.values()),
        var_name="metric_col",
        value_name="value",
    )

    inv = {v: k for k, v in metric_map.items()}
    long["metric"] = long["metric_col"].map(inv)

    counties = load_counties_geojson()

    fig = px.choropleth(
        long,
        geojson=counties,
        locations="GEOID",
        color="value",
        animation_frame="metric",  # slider = toggle
        scope="usa",
        hover_data={"GEOID": True, "metric": True, "value": ":.3f"},
    )

    fig.update_layout(
        title="Civic Time Burden Index (CTBI) — Toggle View",
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        sliders=[dict(currentvalue=dict(prefix="Metric: "))],
    )

    fig.write_html(OUT_HTML, include_plotlyjs="cdn")
    print("✅ Saved:", OUT_HTML)

if __name__ == "__main__":
    main()