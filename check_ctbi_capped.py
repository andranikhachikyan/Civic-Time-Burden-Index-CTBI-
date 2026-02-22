import pandas as pd

PATH = "outputs/ctbi_master_with_capped.csv"

def main():
    df = pd.read_csv(PATH, dtype={"GEOID": str})

    print("\n=== BASIC INFO ===")
    print("Rows:", len(df))

    print("\n=== Hospital cap check ===")
    print("Max hospital_min:", df["hospital_min"].max())
    print("Max hospital_min_capped:", df["hospital_min_capped"].max())

    print("\n=== CTBI distributions ===")
    print(df[["CTBI_raw","CTBI_capped"]].describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]))

    print("\n=== Top 10 RAW CTBI ===")
    print(df.sort_values("CTBI_raw", ascending=False)[
        ["GEOID","CTBI_raw","hospital_min"]
    ].head(10).to_string(index=False))

    print("\n=== Top 10 CAPPED CTBI ===")
    print(df.sort_values("CTBI_capped", ascending=False)[
        ["GEOID","CTBI_capped","hospital_min","hospital_min_capped"]
    ].head(10).to_string(index=False))

    print("\n=== Bottom 10 CAPPED CTBI ===")
    print(df.sort_values("CTBI_capped", ascending=True)[
        ["GEOID","CTBI_capped"]
    ].head(10).to_string(index=False))

    print("\n=== Percentile sanity ===")
    print(df["CTBI_capped_pct_rank"].describe())

if __name__ == "__main__":
    main()