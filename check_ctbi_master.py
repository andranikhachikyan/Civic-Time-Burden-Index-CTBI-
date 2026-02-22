import pandas as pd
import numpy as np

PATH = "ctbi_master_dataset.csv"

# Expected columns (adjust if your file differs)
REQUIRED_COLS = [
    "GEOID",
    "total_pop",
    "low_access_pop",
    "grocery_burden_pct",
    "commute_min",
    "hospital_min",
    "commute_z",
    "grocery_z",
    "hospital_z",
    "CTBI",
]

def to_num(df, col):
    df[col] = pd.to_numeric(df[col], errors="coerce")

def main():
    df = pd.read_csv(PATH, dtype={"GEOID": str})

    print("\n=== BASIC INFO ===")
    print("Rows:", len(df))
    print("Unique GEOID:", df["GEOID"].nunique() if "GEOID" in df.columns else "NO GEOID COLUMN")
    print("Columns:", len(df.columns))

    print("\n=== COLUMN CHECK ===")
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in REQUIRED_COLS]
    if missing_cols:
        print("❌ Missing required columns:", missing_cols)
    else:
        print("✅ All required columns present.")
    if extra_cols:
        print("ℹ️ Extra columns present (fine):", extra_cols)

    if "GEOID" not in df.columns:
        print("\n❌ Cannot proceed: GEOID column missing.")
        return

    # GEOID formatting sanity
    df["GEOID"] = df["GEOID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(5)

    print("\n=== DUPLICATES CHECK ===")
    dup_count = df["GEOID"].duplicated().sum()
    print("Duplicate GEOID rows:", dup_count)
    if dup_count > 0:
        print(df[df["GEOID"].duplicated(keep=False)].sort_values("GEOID").head(20).to_string(index=False))

    # Convert numerics
    numeric_cols = [c for c in REQUIRED_COLS if c != "GEOID" and c in df.columns]
    for c in numeric_cols:
        to_num(df, c)

    print("\n=== MISSING VALUES ===")
    na_counts = df[numeric_cols].isna().sum().sort_values(ascending=False)
    print(na_counts.to_string())
    if na_counts.sum() == 0:
        print("✅ No missing numeric values.")
    else:
        print("⚠️ Some numeric values are missing. Those counties may have been dropped upstream or partially merged.")

    print("\n=== BASIC RANGE CHECKS ===")
    # population
    if "total_pop" in df.columns:
        bad_pop = (df["total_pop"] < 0).sum()
        print("total_pop < 0:", bad_pop)

    # grocery percent: should be [0,100]
    if "grocery_burden_pct" in df.columns:
        out_groc = ((df["grocery_burden_pct"] < 0) | (df["grocery_burden_pct"] > 100)).sum()
        print("grocery_burden_pct outside [0,100]:", out_groc)

    # commute/hospital minutes: should be >= 0, and not absurdly huge
    for col, upper in [("commute_min", 300), ("hospital_min", 5000)]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            huge = (df[col] > upper).sum()
            print(f"{col} < 0:", neg)
            print(f"{col} > {upper}:", huge)

    print("\n=== SUMMARY STATS ===")
    show_cols = [c for c in ["grocery_burden_pct","commute_min","hospital_min","CTBI"] if c in df.columns]
    if show_cols:
        print(df[show_cols].describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]).to_string())

    print("\n=== TOP/BOTTOM COUNTIES BY CTBI ===")
    cols_show = [c for c in ["GEOID","commute_min","grocery_burden_pct","hospital_min","CTBI"] if c in df.columns]
    if "CTBI" in df.columns:
        print("\nTop 15 highest CTBI:")
        print(df.sort_values("CTBI", ascending=False)[cols_show].head(15).to_string(index=False))
        print("\nTop 15 lowest CTBI:")
        print(df.sort_values("CTBI", ascending=True)[cols_show].head(15).to_string(index=False))

    print("\n=== CHECK CTBI CONSTRUCTION (approx) ===")
    # This checks whether CTBI is consistent with a simple sum/mean of z-scores.
    # If your CTBI uses different weights, this will show a mismatch (that's OK, but you should know).
    if all(c in df.columns for c in ["commute_z","grocery_z","hospital_z","CTBI"]):
        approx_sum = df["commute_z"] + df["grocery_z"] + df["hospital_z"]
        approx_mean = approx_sum / 3.0

        # Correlations
        corr_sum = df[["CTBI"]].join(approx_sum.rename("approx_sum")).corr().iloc[0,1]
        corr_mean = df[["CTBI"]].join(approx_mean.rename("approx_mean")).corr().iloc[0,1]
        print(f"Corr(CTBI, commute_z+grocery_z+hospital_z) = {corr_sum:.4f}")
        print(f"Corr(CTBI, mean(z)) = {corr_mean:.4f}")

        # If CTBI is literally mean(z), the max abs diff should be tiny
        max_abs_diff_mean = (df["CTBI"] - approx_mean).abs().max()
        max_abs_diff_sum = (df["CTBI"] - approx_sum).abs().max()
        print("Max |CTBI - mean(z)|:", float(max_abs_diff_mean))
        print("Max |CTBI - sum(z)| :", float(max_abs_diff_sum))

        # Show a few rows with biggest discrepancy vs mean(z)
        tmp = df[["GEOID","CTBI","commute_z","grocery_z","hospital_z"]].copy()
        tmp["approx_mean"] = approx_mean
        tmp["abs_diff_vs_mean"] = (tmp["CTBI"] - tmp["approx_mean"]).abs()
        print("\nLargest 10 discrepancies vs mean(z) (if weights differ, this is expected):")
        print(tmp.sort_values("abs_diff_vs_mean", ascending=False).head(10).to_string(index=False))
    else:
        print("ℹ️ Cannot check CTBI construction: missing one of commute_z/grocery_z/hospital_z/CTBI.")

    print("\n=== COVERAGE CHECK (rough expectation) ===")
    # US counties + equivalents are usually ~3143; some datasets may include/exclude territories.
    # This check is informational, not pass/fail.
    n = df["GEOID"].nunique()
    if n < 2800:
        print("⚠️ Very low county count (<2800): likely merge dropped many counties.")
    elif n < 3050:
        print("⚠️ Some counties missing (<3050). Might be fine depending on scope, but verify.")
    else:
        print("✅ County count looks in the expected range for a national file.")

if __name__ == "__main__":
    main()