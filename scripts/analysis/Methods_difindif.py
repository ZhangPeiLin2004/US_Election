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
df["state"] = df["state"].str.strip()

# Restrict to the election cycle window so pre-2024 noise does not dominate
analysis_start = pd.Timestamp("2023-01-01", tz="UTC")
df = df[df["date"] >= analysis_start].copy()

# ------------------------------------------------------------------
# AMBIGUOUS-LABEL HANDLING (mitigation strategy 1)
# "USA", "Unknown", etc. are aggregate/non-state labels. They are not
# real state-level signal and would pollute the non-swing control group,
# so we flag them, keep them only for the concentration audit, and drop
# them from the causal (DiD) panel.
# ------------------------------------------------------------------
AMBIGUOUS_LABELS = {
    "USA", "United States", "Unknown", "unknown", "Other",
    "N/A", "NA", "None", "", "nan",
}
df["is_ambiguous_state"] = df["state"].isin(AMBIGUOUS_LABELS)
n_ambiguous = int(df["is_ambiguous_state"].sum())

# Full frame (incl. ambiguous labels) is retained for the concentration audit;
# df is the cleaned state-level panel used for causal estimation.
df_full = df.copy()
df = df[~df["is_ambiguous_state"]].copy()

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

# Within-state normalization (mitigation strategy 1): standardize each
# state against its own baseline so high-volume states do not dominate the
# pooled signal. This is the bias-corrected outcome used in robustness checks.
def _zscore(x):
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if sd > 0 else x * 0.0

daily["log_volume_state_z"] = (
    daily.groupby("state")["log_volume"].transform(_zscore)
)

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



# ================================
# 10. LOAD FINAL ELECTION RESULTS
# ================================

results = pd.read_csv(
    r"E:\Csci_3\US_Election\data\trump_vs_harris_margins.csv"
)

results.rename(columns={
    "trump_margin": "final_margin"
}, inplace=True)

results["state"] = results["state"].astype(str)

# ================================
# 11. BUILD STATE-LEVEL FEATURES
# ================================

did_cols = [f"{event}_did" for event in events.keys()]

state_features = daily.groupby("state").agg({
    "tweet_volume": "mean",
    "reply_volume": "mean",
    "log_volume": "mean",
    **{col: "sum" for col in did_cols}
}).reset_index()

# Engagement intensity
state_features["engagement_rate"] = (
    state_features["reply_volume"] /
    state_features["tweet_volume"]
)

# ================================
# 12. MERGE FINAL ELECTION OUTCOMES
# ================================

state_features = state_features.merge(
    results,
    on="state",
    how="left"
)

print("\n=== STATE FEATURES ===")
print(state_features)

# ================================
# 13. FINAL ELECTION PREDICTION MODEL
# ================================

formula = """
final_margin
~ engagement_rate
+ log_volume
+ campaign_start_did
+ debate_did
+ trump_incident_did
+ biden_dropout_did
"""

final_model = smf.ols(
    formula=formula,
    data=state_features
).fit()

print("\n=== FINAL ELECTION MODEL ===")
print(final_model.summary())

# ================================
# 14. GENERATE PREDICTIONS
# ================================

state_features["predicted_margin"] = (
    final_model.predict(state_features)
)

# Prediction error
state_features["prediction_error"] = (
    state_features["final_margin"] -
    state_features["predicted_margin"]
)

print("\n=== PREDICTIONS ===")
print(
    state_features[
        [
            "state",
            "final_margin",
            "predicted_margin",
            "prediction_error"
        ]
    ]
)

# ================================
# 15. VISUALIZATION:
# ACTUAL VS PREDICTED
# ================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=state_features,
    x="final_margin",
    y="predicted_margin",
    s=120
)

for _, row in state_features.iterrows():
    plt.text(
        row["final_margin"] + 0.05,
        row["predicted_margin"] + 0.05,
        row["state"],
        fontsize=9
    )

# Perfect prediction reference line
min_val = min(
    state_features["final_margin"].min(),
    state_features["predicted_margin"].min()
)

max_val = max(
    state_features["final_margin"].max(),
    state_features["predicted_margin"].max()
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--"
)

plt.xlabel("Actual Trump Margin")
plt.ylabel("Predicted Trump Margin")

plt.title(
    "Predicted vs Actual Trump-Harris Margins"
)

plt.tight_layout()
plt.show()

# ================================
# 10. LOAD FINAL ELECTION RESULTS
# ================================

results = pd.read_csv(
    r"E:\Csci_3\US_Election\Peilin\trump_vs_harris_margins.csv"
)

results.rename(columns={
    "trump_margin": "final_margin"
}, inplace=True)

results["state"] = results["state"].astype(str)

# ================================
# 11. BUILD STATE-LEVEL FEATURES
# ================================

did_cols = [f"{event}_did" for event in events.keys()]

state_features = daily.groupby("state").agg({
    "tweet_volume": "mean",
    "reply_volume": "mean",
    "log_volume": "mean",
    **{col: "sum" for col in did_cols}
}).reset_index()

# Engagement intensity
state_features["engagement_rate"] = (
    state_features["reply_volume"] /
    state_features["tweet_volume"]
)

# ================================
# 12. MERGE FINAL ELECTION OUTCOMES
# ================================

state_features = state_features.merge(
    results,
    on="state",
    how="left"
)

# IMPORTANT FIX:
# merge final election results into DAILY too
daily = daily.merge(
    results,
    on="state",
    how="left"
)

print("\n=== STATE FEATURES ===")
print(state_features)

# ================================
# 13. FINAL ELECTION PREDICTION MODEL
# ================================

formula = """
final_margin
~ engagement_rate
+ log_volume
+ campaign_start_did
+ debate_did
+ trump_incident_did
+ biden_dropout_did
"""

final_model = smf.ols(
    formula=formula,
    data=state_features
).fit()

print("\n=== FINAL ELECTION MODEL ===")
print(final_model.summary())

# ================================
# 14. GENERATE PREDICTIONS
# ================================

state_features["predicted_margin"] = (
    final_model.predict(state_features)
)

# Prediction error
state_features["prediction_error"] = (
    state_features["final_margin"] -
    state_features["predicted_margin"]
)

print("\n=== PREDICTIONS ===")
print(
    state_features[
        [
            "state",
            "final_margin",
            "predicted_margin",
            "prediction_error"
        ]
    ]
)

# ================================
# 15. DYNAMIC TRAJECTORY MODEL
# ================================

# Daily momentum proxy
daily["momentum"] = (
    daily["log_volume"]
    .diff()
)

# Remove NA from diff
daily = daily.dropna(subset=["momentum"]).copy()

# Trajectory model
trajectory_formula = """
momentum
~ campaign_start_did
+ debate_did
+ trump_incident_did
+ biden_dropout_did
+ C(state_id)
"""

trajectory_model = smf.ols(
    formula=trajectory_formula,
    data=daily
).fit()

print("\n=== DYNAMIC TRAJECTORY MODEL ===")
print(trajectory_model.summary())

# Predicted momentum
daily["predicted_momentum"] = (
    trajectory_model.predict(daily)
)

# ================================
# CONVERT MOMENTUM INTO TRAJECTORY
# ================================

daily = daily.sort_values(["state", "date"])

daily["trajectory"] = (
    daily.groupby("state")["predicted_momentum"]
    .cumsum()
)

# Smooth trajectories
daily["trajectory_smooth"] = (
    daily.groupby("state")["trajectory"]
    .transform(lambda x: x.rolling(7, min_periods=1).mean())
)

# ================================
# NORMALIZE TO FINAL MARGINS
# ================================

# Anchor trajectory endpoints
final_margin_map = dict(
    zip(results["state"], results["final_margin"])
)

for state in daily["state"].unique():

    mask = daily["state"] == state

    traj = daily.loc[mask, "trajectory_smooth"]

    if traj.std() == 0:
        continue

    # Standardize
    normalized = (
        (traj - traj.mean()) / traj.std()
    )

    # Scale around final election margin
    final_margin = final_margin_map.get(state, 0)

    daily.loc[mask, "trajectory_scaled"] = (
        normalized * 0.8 + final_margin
    )

# ================================
# VISUALIZATION
# ================================

plt.figure(figsize=(15, 7))

sns.lineplot(
    data=daily[daily["treated"] == 1],
    x="date",
    y="trajectory_scaled",
    hue="state",
    linewidth=2
)

# Event markers
for event_name, event_date in events.items():

    plt.axvline(
        pd.to_datetime(event_date, utc=True),
        linestyle="--",
        alpha=0.5
    )

plt.title(
    "Predicted Electoral Momentum Trajectories Across Swing States"
)

plt.xlabel("Date")
plt.ylabel("Predicted Trump Margin Trend")

plt.xticks(rotation=45)

plt.legend(title="Swing State")

plt.tight_layout()
plt.show()


# ##################################################################
# SYSTEMATIC BIAS AUDIT
# Operationalizes the four testing strategies from the Week 8 audit:
# (A) visibility, (B) participation/representation, (C) amplification,
# (D) salience -- plus the normalization / missingness mitigations.
# ##################################################################

# ================================
# 16. ENGAGEMENT CONCENTRATION ANALYSIS  [A: Visibility bias]
# ================================
# Question: is the signal disproportionately driven by a few high-volume
# states? Computed on the FULL frame (incl. ambiguous labels) so that
# aggregate labels like "USA" are visible in the concentration picture.
state_totals = (
    df_full.groupby("state").size().sort_values(ascending=False)
    .rename("volume").to_frame()
)
total_volume = state_totals["volume"].sum()
state_totals["share"] = state_totals["volume"] / total_volume
state_totals["cum_share"] = state_totals["share"].cumsum()


def herfindahl_index(shares):
    """HHI in [0, 1]; higher = more concentrated."""
    return float(np.sum(np.square(shares)))


def gini(values):
    """Gini coefficient of a non-negative distribution."""
    v = np.sort(np.asarray(values, dtype=float))
    n = v.size
    if n == 0 or v.sum() == 0:
        return np.nan
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


hhi = herfindahl_index(state_totals["share"].values)
gini_coef = gini(state_totals["volume"].values)
top4_share = state_totals["share"].head(4).sum()

print(f"\n{'=' * 60}\n16. ENGAGEMENT CONCENTRATION (visibility bias)\n{'=' * 60}")
print(f"States observed:        {len(state_totals)}")
print(f"HHI (0-1, higher=worse): {hhi:.4f}")
print(f"Gini (0-1, higher=worse): {gini_coef:.4f}")
print(f"Top-4 state share:       {top4_share:.3f}")
print("\nTop 10 states by share of engagement volume:")
print(state_totals.head(10).round(4).to_string())

top_n = state_totals.head(15).iloc[::-1]
plt.figure(figsize=(10, 7))
sns.barplot(x=top_n["share"], y=top_n.index, color="#4878CF")
plt.title("Engagement concentration: share of total volume by state (top 15)")
plt.xlabel("Share of total tweet volume")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# ================================
# 17. MISSINGNESS / DATA-COVERAGE REPORT  [mitigation 3]
# ================================
# Sparse states yield unreliable estimates; report coverage transparently.
coverage = (
    daily.groupby("state")
    .agg(
        state_days=("date", "nunique"),
        total_volume=("tweet_volume", "sum"),
        median_daily_volume=("tweet_volume", "median"),
        first_day=("date", "min"),
        last_day=("date", "max"),
    )
    .sort_values("state_days")
)
panel_span_days = (daily["date"].max() - daily["date"].min()).days + 1
coverage["coverage_ratio"] = coverage["state_days"] / panel_span_days
SPARSE_THRESHOLD = 0.25  # <25% of calendar days observed = sparse
coverage["sparse_flag"] = coverage["coverage_ratio"] < SPARSE_THRESHOLD

print(f"\n{'=' * 60}\n17. MISSINGNESS / COVERAGE REPORT\n{'=' * 60}")
print(f"Rows dropped as ambiguous labels (USA/Unknown/...): {n_ambiguous:,}")
print(f"Calendar span: {panel_span_days} days")
print(f"Sparse states (<{SPARSE_THRESHOLD:.0%} day coverage): "
      f"{int(coverage['sparse_flag'].sum())}")
print("\nLeast-covered states:")
print(coverage.head(10).round(3).to_string())

# ================================
# 18. AMPLIFICATION AUDIT  [C: viral-spike sensitivity, mitigation 2]
# ================================
# Flag within-state viral spike days and re-estimate each event DiD with
# spikes removed. If coefficients move sharply, the effect is driven by
# amplified viral activity rather than a stable shift.
SPIKE_QUANTILE = 0.95
daily["state_spike_cut"] = daily.groupby("state")["tweet_volume"].transform(
    lambda x: x.quantile(SPIKE_QUANTILE)
)
daily["is_spike_day"] = (daily["tweet_volume"] > daily["state_spike_cut"]).astype(int)
n_spike = int(daily["is_spike_day"].sum())

daily_no_spike = daily[daily["is_spike_day"] == 0].copy()

amp_rows = []
for event_name in events:
    full_model = run_event_did(daily, event_name)
    ns_model = run_event_did(daily_no_spike, event_name)
    did_col = f"{event_name}_did"
    amp_rows.append({
        "event": event_name,
        "coef_full": (np.nan if full_model is None
                      else full_model.params.get(did_col, np.nan)),
        "coef_no_spike": (np.nan if ns_model is None
                          else ns_model.params.get(did_col, np.nan)),
    })

amp = pd.DataFrame(amp_rows)
amp["abs_change"] = (amp["coef_full"] - amp["coef_no_spike"]).abs()
print(f"\n{'=' * 60}\n18. AMPLIFICATION AUDIT (spike sensitivity)\n{'=' * 60}")
print(f"Spike days flagged (top {1 - SPIKE_QUANTILE:.0%} within state): {n_spike:,}")
print("DiD coefficient with vs without viral spike days:")
print(amp.round(4).to_string(index=False))

# ================================
# 19. NORMALIZED-OUTCOME ROBUSTNESS  [mitigation 1]
# ================================
# Re-estimate each event on the within-state standardized outcome so the
# result cannot be driven by large states' raw volume scale.
norm_rows = []
for event_name in events:
    did_col = f"{event_name}_did"
    if daily[f"{event_name}_post"].sum() == 0:
        continue
    formula_norm = f"log_volume_state_z ~ {did_col} + C(state_id) + C(date_id)"
    m = smf.ols(formula_norm, data=daily).fit(cov_type="HC3")
    norm_rows.append({
        "event": event_name,
        "coef_norm": m.params[did_col],
        "p_value_norm": m.pvalues[did_col],
    })
norm = pd.DataFrame(norm_rows)
print(f"\n{'=' * 60}\n19. NORMALIZED-OUTCOME DiD (within-state z-score)\n{'=' * 60}")
print(norm.round(4).to_string(index=False))

# ================================
# 20. REPRESENTATION AUDIT  [B: engagement vs actual election margins]
# ================================
# Compare online engagement intensity against external ground truth: the
# certified 2024 presidential margin (abs. pp) in each swing state. Tests
# whether higher visibility tracks real electoral competitiveness.
# Source: certified 2024 results (Trump - Harris margin, percentage points).
actual_margin_2024 = {
    "Arizona": 5.5,
    "Georgia": 2.2,
    "Michigan": 1.4,
    "Nevada": 3.1,
    "North Carolina": 3.2,
    "Pennsylvania": 1.7,
    "Wisconsin": 0.9,
}

swing_intensity = (
    daily[daily["state"].isin(swing_states)]
    .groupby("state")
    .agg(mean_daily_volume=("tweet_volume", "mean"),
         mean_log_volume=("log_volume", "mean"))
)
swing_intensity["abs_margin"] = swing_intensity.index.map(actual_margin_2024)
swing_intensity = swing_intensity.dropna(subset=["abs_margin"])

print(f"\n{'=' * 60}\n20. REPRESENTATION AUDIT (engagement vs 2024 margin)\n{'=' * 60}")
if len(swing_intensity) >= 3:
    corr = swing_intensity["mean_log_volume"].corr(swing_intensity["abs_margin"])
    print(f"Correlation(mean log volume, |2024 margin|) = {corr:.3f}")
    print("Positive => louder states were the LESS competitive ones "
          "(visibility != representativeness).")
    print(swing_intensity.round(3).to_string())

    plt.figure(figsize=(8, 6))
    sns.regplot(data=swing_intensity, x="mean_log_volume", y="abs_margin",
                ci=None, scatter_kws={"s": 60})
    for state, row in swing_intensity.iterrows():
        plt.annotate(state, (row["mean_log_volume"], row["abs_margin"]),
                     fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.title("Representation audit: engagement intensity vs actual 2024 margin")
    plt.xlabel("Mean daily log volume (engagement intensity)")
    plt.ylabel("|2024 presidential margin| (pp)")
    plt.tight_layout()
    plt.show()
else:
    print("Insufficient swing-state coverage for representation audit.")

print(f"\n{'=' * 60}\nBIAS AUDIT COMPLETE\n{'=' * 60}")