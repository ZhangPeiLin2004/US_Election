import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# ================================
# 1. LOAD DATA
# ================================
file_path = r"C:\Users\Lucky\Desktop\WG1\subjects_AND_sampling_metadata_anonymized_full.csv"

df = pd.read_csv(
    file_path,
    usecols=[
        "created_at",
        "tweets_historical",
        "clean...state_simple",
        "in_reply_to_user_id_code",
    ],
    dtype={
        "clean...state_simple": "string",
        "in_reply_to_user_id_code": "string",
    },
    low_memory=False,
)

# ================================
# 2. CLEAN
# ================================
df = df.dropna(subset=["created_at", "clean...state_simple"])

# Parse UTC timestamps and collapse to calendar days (not raw timestamps)
df["date"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True).dt.normalize()
df = df.dropna(subset=["date"])

df.rename(columns={"clean...state_simple": "state"}, inplace=True)

# Restrict to the election cycle window so pre-2024 noise does not dominate
analysis_start = pd.Timestamp("2023-01-01", tz="UTC")
df = df[df["date"] >= analysis_start].copy()

# ================================
# 3. TREATMENT: SWING STATES VS ALL OTHERS
# ================================
swing_states = [
    "Arizona", "Georgia", "Michigan",
    "Nevada", "North Carolina",
    "Pennsylvania", "Wisconsin",
]

df["treated"] = df["state"].isin(swing_states).astype(int)

# ================================
# 4. DAILY AGGREGATION (STATE x CALENDAR DAY)
# ================================
# tweets_historical holds tweet text; count = sampled subjects per state-day
daily = df.groupby(["state", "date"], as_index=False).agg(
    tweet_volume=("tweets_historical", "count"),
    reply_volume=("in_reply_to_user_id_code", lambda x: x.notna().sum()),
    treated=("treated", "max"),
)

daily["log_volume"] = np.log1p(daily["tweet_volume"])
# Use plain str labels for formula API compatibility (not pandas StringDtype)
daily["state_id"] = daily["state"].astype(str)
daily["date_id"] = daily["date"].dt.strftime("%Y-%m-%d")

# ================================
# 5. EVENT DEFINITIONS
# ================================
events = {
    "campaign_start": "2024-01-15",
    "debate": "2024-06-27",
    "trump_incident": "2024-07-13",
    "biden_dropout": "2024-07-21",
    # election_day omitted: data ends 2024-10-31, no post-window coverage
}

data_end = daily["date"].max()

# ================================
# 6. EVENT WINDOWS (LEAD / POST RELATIVE TO EACH EVENT)
# ================================
for event_name, event_date in events.items():
    event_dt = pd.to_datetime(event_date, utc=True)
    daily[f"{event_name}_time"] = (daily["date"] - event_dt).dt.days

    daily[f"{event_name}_lead"] = (
        (daily[f"{event_name}_time"] >= -10) & (daily[f"{event_name}_time"] <= -1)
    ).astype(int)
    daily[f"{event_name}_post"] = (
        (daily[f"{event_name}_time"] >= 0) & (daily[f"{event_name}_time"] <= 10)
    ).astype(int)
    daily[f"{event_name}_did"] = daily["treated"] * daily[f"{event_name}_post"]

# ================================
# 7. DIFFERENCE-IN-DIFFERENCES (ONE EVENT PER MODEL)
# ================================
# State FE absorbs time-invariant treated; date FE absorbs national shocks.
# Do not include a separate 'treated' level term (collinear with C(state_id)).
# Estimate events separately to avoid overlapping post-window collinearity.

def run_event_did(data, event_name):
    """OLS DiD for a single event with state and calendar-day fixed effects."""
    did_col = f"{event_name}_did"
    post_col = f"{event_name}_post"

    if data[post_col].sum() == 0:
        return None

    formula = f"log_volume ~ {did_col} + C(state_id) + C(date_id)"
    return smf.ols(formula, data=data).fit(cov_type="HC3")


results = []
for event_name in events:
    model = run_event_did(daily, event_name)
    if model is None:
        results.append({
            "event": event_name,
            "coef": np.nan,
            "std_err": np.nan,
            "p_value": np.nan,
            "n_obs": len(daily),
            "note": "no post-window observations",
        })
        continue

    did_col = f"{event_name}_did"
    results.append({
        "event": event_name,
        "coef": model.params[did_col],
        "std_err": model.bse[did_col],
        "p_value": model.pvalues[did_col],
        "n_obs": int(model.nobs),
        "r_squared": model.rsquared,
        "note": "",
    })
    print(f"\n{'=' * 60}\nEvent: {event_name}\n{'=' * 60}")
    print(model.summary())

effects = pd.DataFrame(results)
print("\n=== DiD treatment effects (log volume, one model per event) ===")
print(effects.to_string(index=False))

# ================================
# 8. PARALLEL TRENDS CHECK (LEAD INTERACTION, SINGLE EVENT EXAMPLE)
# ================================
# If pre-trends are flat, treated x lead should be insignificant.
example_event = "debate"
lead_col = f"{example_event}_lead"
daily[f"{example_event}_lead_did"] = daily["treated"] * daily[lead_col]

pretrend_formula = (
    f"log_volume ~ {example_event}_lead_did + {example_event}_did "
    f"+ C(state_id) + C(date_id)"
)
pretrend_model = smf.ols(pretrend_formula, data=daily).fit(cov_type="HC3")
print(f"\n=== Parallel trends check ({example_event}: lead x treated) ===")
print(
    f"lead_did coef = {pretrend_model.params[f'{example_event}_lead_did']:.4f}, "
    f"p = {pretrend_model.pvalues[f'{example_event}_lead_did']:.4f}"
)

# ================================
# 9. VISUALIZATION: SWING VS NON-SWING MEAN TRAJECTORY
# ================================
daily["group"] = daily["treated"].map({1: "Swing states", 0: "Other states"})
group_daily = (
    daily.groupby(["date", "group"], as_index=False)["log_volume"]
    .mean()
)

plt.figure(figsize=(12, 6))
sns.lineplot(data=group_daily, x="date", y="log_volume", hue="group", linewidth=2)

for event_name, event_date in events.items():
    event_dt = pd.to_datetime(event_date, utc=True)
    if event_dt <= data_end:
        plt.axvline(event_dt, color="gray", linestyle="--", alpha=0.6, linewidth=1)

plt.title("Daily log volume: swing states vs other states (2023+)")
plt.xlabel("Date")
plt.ylabel("log(1 + tweet_volume)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"\nData range: {daily['date'].min().date()} to {data_end.date()}")
print(f"Panel: {len(daily):,} state-days, {daily['state'].nunique()} states")
