import numpy as np
import pandas as pd

FILE_IN = "subjects_AND_sampling_metadata_anonymized_full.csv"
FILE_OUT = "subjects_cleaned_basic.csv"

dtype_map = {
    "entities.url.urls": "string",
    "attachments.media_keys.subject_pool": "string",
    "attachments.media_source_tweet_id.tweets_historical": "string",
}

NROWS = 100000  # None

def load_raw():
    return pd.read_csv(
        FILE_IN,
        nrows=NROWS,
        dtype=dtype_map,
        low_memory=False,
    )


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Most basic data cleaning: names, typo, dates, types, filters."""
    df = df.copy()

    # 1. Rename columns (clean...* -> clean_*)
    renames = {
        "clean...pool": "clean_pool",
        "clean...pool.date": "clean_pool_date",
        "clean...state_simple": "clean_state_simple",
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

    # 2. Fix typo
    if "clean_pool" in df.columns:
        df["clean_pool"] = df["clean_pool"].replace("frees speech", "free speech")

    # 3. Parse datetime columns
    for col in ["created_at", "created_at.tweets", "created_at.users", "sampling_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # 4. Ensure ID columns are string (avoid scientific notation)
    id_cols = [c for c in df.columns if c.endswith("_code")]
    for col in id_cols:
        df[col] = df[col].astype("string")

    # 5. Basic filters: keep only non-excluded, non-test
    if "EXCLUDED" in df.columns:
        df = df[df["EXCLUDED"].astype(str).str.lower() == "no"]
    if "TEST" in df.columns:
        df = df[df["TEST"].astype(str).str.lower() == "no"]

    # 6. Drop rows missing key analysis fields
    for col in ["clean_pool", "sampling_date"]:
        if col in df.columns:
            df = df[df[col].notna()]

    return df


if __name__ == "__main__":
    df = load_raw()
    n_before = len(df)
    df = basic_clean(df)
    n_after = len(df)

    df.to_csv(FILE_OUT, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    print(f"Rows: {n_before} -> {n_after}")
    print(f"Saved: {FILE_OUT}")
    print(df.head(5))