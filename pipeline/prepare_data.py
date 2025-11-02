import pandas as pd 
import numpy as np 
import pathlib 


RAW_PATH = pathlib.Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUT_DIR = pathlib.Path("data")
CLEAN_PATH = OUT_DIR/"ibm_telco_clean.csv"

##standardise Column names 
def load_raw(path):
    df = pd.read_csv(path)
    df.columns = (df.columns
                  .str.strip().str.lower()
                  .str.replace(" ", "_")
                  .str.replace("-","_")
                  )
    return df 


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Fix data types
    # totalcharges sometimes comes as string with blanks → numeric (blanks → NaN)
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
    df["monthlycharges"] = pd.to_numeric(df["monthlycharges"], errors="coerce")

    # tenure to integer; also a helper in years
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
    df["tenure_years"] = (df["tenure"] / 12).round(2)

    # 2) Create numeric label 0/1 from "Churn" (yes/no)
    df["churn_flag"] = (df["churn"].str.strip().str.lower() == "yes").astype(int)

    # 3) Normalize categorical noise
    # Replace "No internet service" and "No phone service" with plain "No"
    for c in ["onlinesecurity", "onlinebackup", "deviceprotection",
              "techsupport", "streamingtv", "streamingmovies"]:
        if c in df.columns:
            df[c] = df[c].replace({"No internet service": "No"})
    if "multiplelines" in df.columns:
        df["multiplelines"] = df["multiplelines"].replace({"No phone service": "No"})

    # 4) Map common Yes/No fields to 1/0 (keeps modeling simple)
    bin_map = {"Yes": 1, "No": 0}
    for col in ["partner","dependents","phoneservice","paperlessbilling",
                "onlinesecurity","onlinebackup","deviceprotection","techsupport",
                "streamingtv","streamingmovies"]:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].map(bin_map)

    # 5) Drop rows that would break modeling (very few)
    df = df.dropna(subset=["monthlycharges", "totalcharges"])

    # 6) Keep a curated, business-relevant subset (and stable order)
    keep_cols = [
        "customerid","gender","seniorcitizen","partner","dependents","tenure","tenure_years",
        "phoneservice","multiplelines","internetservice","onlinebackup","onlinesecurity",
        "deviceprotection","techsupport","streamingtv","streamingmovies","contract",
        "paperlessbilling","paymentmethod","monthlycharges","totalcharges","churn_flag"
    ]
    existing = [c for c in keep_cols if c in df.columns]
    df = df[existing].reset_index(drop=True)
    return df


def quick_summary(df: pd.DataFrame) -> None:
    print("\n=== Target distribution (churn_flag) ===")
    print(df["churn_flag"].value_counts(normalize=True).round(3))
    print("\n=== Tenure summary ===")
    print(df["tenure"].describe())
    print("\n=== Monthly charges summary ===")
    print(df["monthlycharges"].describe())
    if "contract" in df.columns:
        print("\n=== Contract types ===")
        print(df["contract"].value_counts())


if __name__ == "__main__":
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw CSV not found at: {RAW_PATH.resolve()}")
    raw = load_raw(RAW_PATH)
    print("=== RAW LOADED ===")
    print(f"Rows: {raw.shape[0]}, Cols: {raw.shape[1]}")
    clean_df = clean(raw)
    print("\n=== CLEANED SUMMARY ===")
    quick_summary(clean_df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved cleaned dataset -> {CLEAN_PATH.resolve()}")
