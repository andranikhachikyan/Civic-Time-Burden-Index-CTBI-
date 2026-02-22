import os
import pandas as pd

IN_PATH = "Food Access Research Atlas-Table 1.csv"
OUT_PATH = "outputs/county_grocery_access.csv"

# Choose ONE definition to keep your project clean:
# LAPOP1_10 = people low-access (>1 mile urban OR >10 miles rural)
LOW_ACCESS_POP_COL = "LAPOP1_10"
TOTAL_POP_COL = "Pop2010"

def main():
    os.makedirs("outputs", exist_ok=True)

    # Load
    df = pd.read_csv(IN_PATH, dtype={"CensusTract": str})

    # Basic cleaning: ensure tract id is 11 digits (some files may load as int/float otherwise)
    df["CensusTract"] = df["CensusTract"].str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(11)

    # County FIPS is first 5 digits of tract GEOID
    df["county_fips"] = df["CensusTract"].str[:5]

    # Convert key columns to numeric (NULL -> NaN)
    df[TOTAL_POP_COL] = pd.to_numeric(df[TOTAL_POP_COL], errors="coerce")
    df[LOW_ACCESS_POP_COL] = pd.to_numeric(df[LOW_ACCESS_POP_COL], errors="coerce")

    # Drop rows missing essentials
    df = df.dropna(subset=["county_fips", TOTAL_POP_COL, LOW_ACCESS_POP_COL])

    # Aggregate to county
    county = (
        df.groupby("county_fips", as_index=False)
          .agg(
              county_pop2010=(TOTAL_POP_COL, "sum"),
              county_lowaccess_pop=(LOW_ACCESS_POP_COL, "sum"),
          )
    )

    # Compute share (% of people low-access to grocery)
    county["grocery_lowaccess_share"] = county["county_lowaccess_pop"] / county["county_pop2010"]

    # Optional: nice percentage column for display
    county["grocery_lowaccess_pct"] = county["grocery_lowaccess_share"] * 100

    # Save
    county.to_csv(OUT_PATH, index=False)

    # Quick sanity prints
    print("✅ Saved:", OUT_PATH)
    print("Rows (counties):", len(county))
    print(county.sort_values("grocery_lowaccess_share", ascending=False).head(10).to_string(index=False))
    print("\nSummary:")
    print(county["grocery_lowaccess_share"].describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]))

if __name__ == "__main__":
    main()