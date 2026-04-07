import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import statsmodels.formula.api as smf

# ================================
# Read & Clean Data
# ================================

file_path = r"E:\Csci_3\subjects_AND_sampling_metadata_anonymized_full.csv"
chunk_size = 100000

cols_needed = [
    "clean...state_simple",
    "created_at.users",
    "tweets_historical",
    "sampling_tweet",
    "entities.hashtags.subject_pool",
    "public_metrics.like_count",
    "public_metrics.retweet_count.tweets_historical"
]

df_list = []

for chunk in pd.read_csv(file_path,
                         chunksize=chunk_size,
                         usecols=lambda x: x in cols_needed,
                         dtype=str,
                         low_memory=False):
    
    chunk = chunk.dropna(subset=["created_at.users", "clean...state_simple", "sampling_tweet"])
    
    chunk["date"] = pd.to_datetime(chunk["created_at.users"], errors="coerce")
    chunk["like_count"] = pd.to_numeric(chunk["public_metrics.like_count"], errors="coerce")
    chunk["retweet_count"] = pd.to_numeric(chunk["public_metrics.retweet_count.tweets_historical"], errors="coerce")
    
    df_list.append(chunk)

df = pd.concat(df_list, ignore_index=True)

# ✅ Rename column (CRITICAL FIX)
df = df.rename(columns={"clean...state_simple": "state"})

# Filter swing states
swing_states = ["Arizona", "Georgia", "Michigan", "Nevada", "North Carolina", "Pennsylvania", "Wisconsin"]
df = df[df["state"].isin(swing_states)]

# ================================
# Aggregate Daily Data
# ================================

state_daily = df.groupby(["state", "date"]).agg({
    "like_count": "mean",
    "retweet_count": "mean",
    "tweets_historical": "count",
    "entities.hashtags.subject_pool": lambda x: x.notna().sum()
}).reset_index()

state_daily = state_daily[state_daily["date"] >= "2022-01-01"].copy()
state_daily.reset_index(drop=True, inplace=True)

state_daily.rename(columns={
    "tweets_historical": "tweet_volume",
    "entities.hashtags.subject_pool": "hashtag_count"
}, inplace=True)

# ================================
# Engagement Features
# ================================

state_daily["engagement_score"] = state_daily["like_count"].fillna(0) + state_daily["retweet_count"].fillna(0)

state_daily["engagement_per_tweet"] = (
    state_daily["engagement_score"] / state_daily["tweet_volume"].replace(0, np.nan)
)

state_daily["engagement_baseline"] = state_daily.groupby("state")["engagement_per_tweet"]\
    .transform(lambda x: x.rolling(7, min_periods=3).mean())

state_daily["engagement_spike"] = state_daily["engagement_per_tweet"] - state_daily["engagement_baseline"]

threshold = state_daily["engagement_spike"].quantile(0.75)
state_daily["high_spike"] = (state_daily["engagement_spike"] > threshold).astype(int)

# ================================
# Event Features
# ================================

events = {
    "Event1": "2024-01-01",
    "Event2": "2024-06-01",
    "Event3": "2024-07-13",
    "Event4": "2024-07-21",
    "Event5": "2024-10-20"
}

state_daily["date"] = pd.to_datetime(state_daily["date"], utc=True)

for key, date in events.items():
    event_date = pd.to_datetime(date, utc=True)
    
    state_daily[f"{key}_days_since"] = (state_daily["date"] - event_date).dt.days
    
    state_daily[f"{key}_post"] = (
        (state_daily[f"{key}_days_since"] > 0) &
        (state_daily[f"{key}_days_since"] <= 15)
    ).astype(int)

    temp = state_daily[
        (state_daily[f"{key}_days_since"] >= 0) &
        (state_daily[f"{key}_days_since"] <= 3)
    ]

    spike_by_state = temp.groupby("state")["engagement_spike"].mean()
    treated_states = spike_by_state[spike_by_state > spike_by_state.quantile(0.5)].index

    state_daily[f"{key}_treated"] = state_daily["state"].isin(treated_states).astype(int)

    state_daily[f"{key}_interaction"] = (
        state_daily[f"{key}_treated"] * state_daily[f"{key}_post"]
    )

# ================================
# NLP Features
# ================================

df["date"] = pd.to_datetime(df["date"]).dt.date
state_daily["date"] = pd.to_datetime(state_daily["date"]).dt.date

state_text = df.groupby(["state", "date"])["sampling_tweet"]\
               .apply(lambda x: " ".join(x))\
               .reset_index()

vectorizer = CountVectorizer(max_features=1000, stop_words="english")
doc_term_matrix = vectorizer.fit_transform(state_text["sampling_tweet"])

lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda_topics = lda.fit_transform(doc_term_matrix)

for i in range(10):
    state_text[f"topic_{i}"] = lda_topics[:, i]

state_daily = state_daily.merge(
    state_text.drop(columns=["sampling_tweet"]),
    on=["state", "date"],
    how="left"
)

# ================================
# DiD MODEL
# ================================

state_daily["y"] = state_daily["engagement_spike"].shift(-1)
state_daily_model = state_daily.dropna(subset=["y"]).copy()

scaler = StandardScaler()
state_daily_model[["engagement_per_tweet", "tweet_volume"]] = scaler.fit_transform(
    state_daily_model[["engagement_per_tweet", "tweet_volume"]]
)

interaction_terms = " + ".join([f"{e}_interaction" for e in events.keys()])
topic_terms = " + ".join([f"topic_{i}" for i in range(10)])

formula = f"""
y ~ engagement_per_tweet + tweet_volume
    + {interaction_terms}
    + {topic_terms}
    + C(state)
    + C(date)
"""

model = smf.ols(formula=formula, data=state_daily_model).fit(cov_type='HC3')

print(model.summary())

# ================================
# DiD Effects
# ================================

did_effects = pd.DataFrame({
    "Event": list(events.keys()),
    "DiD_coef": [model.params.get(f"{e}_interaction", np.nan) for e in events.keys()]
})

print("\nEstimated DiD effects per event:")
print(did_effects)

# ================================
# Visualization
# ================================

state_daily_model["predicted"] = model.predict(state_daily_model)

sns.lineplot(
    data=state_daily_model,
    x="date",
    y="predicted",
    hue="state"
)

plt.title("Estimated engagement-based polling trajectory (DiD + NLP)")
plt.ylabel("Predicted outcome (proxy)")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()