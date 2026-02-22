import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# =====================================================
# PART 1 — GROCERY BURDEN (TRACT → COUNTY)
# =====================================================

df = pd.read_csv("Food Access Research Atlas-Table 1.csv")

# Preserve leading zeros
df["CensusTract"] = df["CensusTract"].astype(str).str.zfill(11)
df["GEOID"] = df["CensusTract"].str[:5]

county_grocery = (
    df.groupby("GEOID")
      .agg(
          total_pop=("Pop2010", "sum"),
          low_access_pop=("LAPOP1_10", "sum")
      )
      .reset_index()
)

county_grocery["grocery_burden_pct"] = (
    county_grocery["low_access_pop"] / county_grocery["total_pop"]
) * 100

print("Grocery rows:", len(county_grocery))


# =====================================================
# PART 2 — COMMUTE MEAN (ACS B08303)
# =====================================================

commute_raw = pd.read_csv("TTW.csv")

# Drop metadata row
commute = commute_raw.iloc[1:].copy()

# Extract 5-digit FIPS
commute["GEOID"] = commute["GEO_ID"].str[-5:]

# Convert numeric columns
for col in commute.columns:
    if col.endswith("E"):
        commute[col] = pd.to_numeric(commute[col], errors="coerce")

# Midpoints for commute bins
midpoints = {
    "B08303_002E": 2.5,
    "B08303_003E": 7,
    "B08303_004E": 12,
    "B08303_005E": 17,
    "B08303_006E": 22,
    "B08303_007E": 27,
    "B08303_008E": 32,
    "B08303_009E": 37,
    "B08303_010E": 42,
    "B08303_011E": 52,
    "B08303_012E": 75,
    "B08303_013E": 95
}

weighted_sum = sum(commute[col] * midpoint for col, midpoint in midpoints.items())
total_workers = commute["B08303_001E"]

commute["commute_min"] = weighted_sum / total_workers

commute_clean = commute[["GEOID", "commute_min"]].copy()

print("Commute rows:", len(commute_clean))


# =====================================================
# PART 3 — MERGE GROCERY + COMMUTE
# =====================================================

merged = county_grocery.merge(commute_clean, on="GEOID", how="inner")

print("After commute merge:", len(merged))


# =====================================================
# PART 4 — HOSPITAL ACCESS
# =====================================================

hospital_df = pd.read_csv("county_nearest_hospital.csv")

hospital_df["GEOID"] = hospital_df["county_fips"].astype(str).str.zfill(5)

hospital_clean = hospital_df[["GEOID", "nearest_hospital_minutes_est"]].copy()
hospital_clean.rename(
    columns={"nearest_hospital_minutes_est": "hospital_min"},
    inplace=True
)

print("Hospital rows:", len(hospital_clean))


# =====================================================
# PART 5 — FINAL MASTER MERGE
# =====================================================

master = merged.merge(hospital_clean, on="GEOID", how="inner")

print("Final merged rows:", len(master))
print(master.head())


# =====================================================
# PART 6 — STANDARDIZE VARIABLES
# =====================================================

scaler = StandardScaler()

master[["commute_z", "grocery_z", "hospital_z"]] = scaler.fit_transform(
    master[["commute_min", "grocery_burden_pct", "hospital_min"]]
)

# =====================================================
# PART 7 — BUILD INDEX
# =====================================================

master["CTBI"] = (
    master["commute_z"] +
    master["grocery_z"] +
    master["hospital_z"]
)

print("\nCTBI Summary:")
print(master["CTBI"].describe())


# =====================================================
# OPTIONAL — SAVE FINAL DATASET
# =====================================================

master.to_csv("ctbi_master_dataset.csv", index=False)