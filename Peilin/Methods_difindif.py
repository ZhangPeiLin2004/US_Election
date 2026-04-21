import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# ================================
# 1. LOAD DATA
# ================================
file_path = r"E:\Csci_3\subjects_AND_sampling_metadata_anonymized_full.csv"

df = pd.read_csv(
    file_path,
    usecols=[
        "created_at",
        "tweets_historical",
        "clean...state_simple",
        "in_reply_to_user_id_code"
    ],
    dtype={
        "tweets_historical": "float64",      # or int64 if safe
        "clean...state_simple": "string",
        "in_reply_to_user_id_code": "string"  # <- important fix
    },
    low_memory=False
)

# ================================
# 2. CLEAN
# ================================
df = df.dropna(subset=["created_at", "clean...state_simple"])

df["date"] = pd.to_datetime(df["created_at"], errors="coerce")
df = df.dropna(subset=["date"])

df.rename(columns={"clean...state_simple": "state"}, inplace=True)

# ================================
# 3. KEEP ALL STATES (CRITICAL FIX)
# ================================
# DO NOT filter to swing states only

swing_states = [
    "Arizona", "Georgia", "Michigan",
    "Nevada", "North Carolina",
    "Pennsylvania", "Wisconsin"
]

df["treated"] = df["state"].isin(swing_states).astype(int)

# ================================
# 4. DAILY AGGREGATION
# ================================
daily = df.groupby(["state", "date"]).agg(
    tweet_volume=("tweets_historical", "count"),
    reply_volume=("in_reply_to_user_id_code", lambda x: x.notna().sum())
).reset_index()

daily["date"] = pd.to_datetime(daily["date"])

# IMPORTANT: keep scale interpretable (NO z-scoring)
daily["log_volume"] = np.log1p(daily["tweet_volume"])

# ================================
# 5. EVENT DEFINITIONS
# ================================
events = {
    "campaign_start": "2024-01-15",
    "debate": "2024-06-27",
    "trump_incident": "2024-07-13",
    "biden_dropout": "2024-07-21",
    "election_day": "2024-11-05"
}

# ================================
# 6. EVENT VARIABLES (FIXED: event time + bins instead of overlap dummies)
# ================================
for e, d in events.items():
    ed = pd.to_datetime(d)

    daily[f"{e}_time"] = (daily["date"] - ed).dt.days

    # CLEAN EVENT WINDOWS (NO OVERLAP COLLISION)
    daily[f"{e}_lead"] = ((daily[f"{e}_time"] >= -10) & (daily[f"{e}_time"] <= -1)).astype(int)
    daily[f"{e}_post"] = ((daily[f"{e}_time"] >= 0) & (daily[f"{e}_time"] <= 10)).astype(int)

# ================================
# 7. FIXED EFFECTS STRUCTURE
# ================================
daily["state_id"] = daily["state"]

# ================================
# 8. DIFFERENCE-IN-DIFFERENCES (CORRECT SPEC)
# ================================
for e in events.keys():
    daily[f"{e}_did"] = daily["treated"] * daily[f"{e}_post"]

# ================================
# 9. MODEL (IDENTIFIED)
# ================================
formula = """
log_volume ~ treated
+ campaign_start_did
+ debate_did
+ trump_incident_did
+ biden_dropout_did
+ election_day_did
+ C(state_id)
"""

model = smf.ols(formula, data=daily).fit(cov_type="HC3")

print(model.summary())

# ================================
# 10. PREDICTIONS (NOW VARIES PROPERLY)
# ================================
daily["predicted"] = model.predict(daily)

# ================================
# 11. VISUALIZATION
# ================================
plt.figure(figsize=(12,6))

sns.lineplot(
    data=daily,
    x="date",
    y="predicted",
    hue="state",
    alpha=0.7
)

plt.title("Corrected Event-driven Attention Trajectories")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ================================
# 12. EVENT EFFECTS TABLE
# ================================
effects = pd.DataFrame({
    "event": list(events.keys()),
    "effect": [model.params.get(f"{e}_did", np.nan) for e in events.keys()]
})

print(effects)