import pandas as pd

IN_PATH = "ctbi_master_dataset.csv"
OUT_PATH = "outputs/ctbi_master_with_capped.csv"

def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)

def main():
    df = pd.read_csv(IN_PATH, dtype={"GEOID": str})
    df["GEOID"] = df["GEOID"].str.zfill(5)

    # Keep raw CTBI
    df["CTBI_raw"] = df["CTBI"]

    # Cap hospital_min at 99th percentile
    cap = df["hospital_min"].quantile(0.99)
    df["hospital_min_capped"] = df["hospital_min"].clip(upper=cap)

    # Recompute z-scores using capped hospital minutes
    df["commute_z2"] = zscore(df["commute_min"])
    df["grocery_z2"] = zscore(df["grocery_burden_pct"])
    df["hospital_z2"] = zscore(df["hospital_min_capped"])

    # New CTBI
    df["CTBI_capped"] = df["commute_z2"] + df["grocery_z2"] + df["hospital_z2"]

    # Helpful ranks for visualization
    df["CTBI_capped_pct_rank"] = df["CTBI_capped"].rank(pct=True)

    df.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Hospital cap (99th pct):", cap)
    print("\nTop 10 CTBI_raw:")
    print(df.sort_values("CTBI_raw", ascending=False)[["GEOID","CTBI_raw","hospital_min"]].head(10).to_string(index=False))
    print("\nTop 10 CTBI_capped:")
    print(df.sort_values("CTBI_capped", ascending=False)[["GEOID","CTBI_capped","hospital_min","hospital_min_capped"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()