import ast

import pandas as pd

file_path = DATA_CSV
chunk_size = 100_000

cols_needed = [
    "clean...state_simple",
    "created_at.users",
    "tweets_historical",
    "sampling_tweet",
    "entities.hashtags.subject_pool",
    "public_metrics.like_count",
    "public_metrics.retweet_count.tweets_historical",
]

df_list = []
swing_states = [
    "Arizona", "Georgia", "Michigan", "Nevada",
    "North Carolina", "Pennsylvania", "Wisconsin",
]

for chunk in pd.read_csv(
    file_path,
    chunksize=chunk_size,
    usecols=lambda x: x in cols_needed,
    dtype=str,
    low_memory=False,
):
    chunk = chunk.dropna(
        subset=["created_at.users", "clean...state_simple", "sampling_tweet"]
    )
    chunk["date"] = pd.to_datetime(chunk["created_at.users"], errors="coerce", utc=True)
    chunk["like_count"] = pd.to_numeric(
        chunk["public_metrics.like_count"], errors="coerce"
    )
    chunk["retweet_count"] = pd.to_numeric(
        chunk["public_metrics.retweet_count.tweets_historical"], errors="coerce"
    )
    chunk = chunk[chunk["clean...state_simple"].isin(swing_states)]

    historical_rows = []
    for _, row in chunk.iterrows():
        hist_raw = row.get("tweets_historical")
        if pd.isna(hist_raw) or isinstance(hist_raw, float) or str(hist_raw).strip() == "":
            continue
        try:
            hist_tweets = ast.literal_eval(hist_raw)
            if isinstance(hist_tweets, dict):
                hist_tweets = [hist_tweets]
            if not isinstance(hist_tweets, list):
                continue
        except (ValueError, SyntaxError):
            continue

        for ht in hist_tweets:
            try:
                if isinstance(ht, dict):
                    text = ht.get("text", "")
                    date = pd.to_datetime(ht.get("created_at"), errors="coerce", utc=True)
                    like_count = pd.to_numeric(ht.get("like_count"), errors="coerce")
                    retweet_count = pd.to_numeric(ht.get("retweet_count"), errors="coerce")
                else:
                    text = str(ht)
                    date = row["date"]
                    like_count = row["like_count"]
                    retweet_count = row["retweet_count"]
                historical_rows.append({
                    "clean...state_simple": row["clean...state_simple"],
                    "sampling_tweet": text,
                    "date": date,
                    "like_count": like_count,
                    "retweet_count": retweet_count,
                    "entities.hashtags.subject_pool": row.get(
                        "entities.hashtags.subject_pool", None
                    ),
                })
            except Exception:
                continue

    chunk_hist = pd.DataFrame(historical_rows) if historical_rows else pd.DataFrame()
    combined = pd.concat([chunk, chunk_hist], ignore_index=True)
    df_list.append(combined)

df = pd.concat(df_list, ignore_index=True)
start_date = pd.Timestamp("2024-01-01", tz="UTC")
end_date = pd.Timestamp("2024-11-05", tz="UTC")
df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
df = df.drop_duplicates(
    subset=["sampling_tweet", "date", "clean...state_simple"]
).reset_index(drop=True)

print(f"Total tweets in swing states from 2024-01-01 to Election Day: {len(df)}")
