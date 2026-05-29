# ------------------------------------------
# 0. PREREQUISITE: predicted_momentum (from Poll prediction & trajectory)
# ------------------------------------------
if "predicted_momentum" not in daily.columns:
    raise NameError(
        "Column 'predicted_momentum' is missing. Run the cell "
        "'## Poll prediction & trajectory' first (section 15 builds it)."
    )

# ------------------------------------------
# 1. AMPLIFICATION METRICS
# ------------------------------------------

# Reply intensity
daily["amplification_rate"] = (
    daily["reply_volume"] /
    daily["tweet_volume"]
)

# Replace divide-by-zero or inf values
daily["amplification_rate"] = (
    daily["amplification_rate"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

# ------------------------------------------
# 2. VIRAL SPIKE DETECTION
# ------------------------------------------

# State-level z-score of tweet activity
def _volume_zscore(x):
    sd = x.std()
    if sd == 0 or np.isnan(sd):
        return x * 0.0
    return (x - x.mean()) / sd


daily["volume_zscore"] = (
    daily.groupby("state")["tweet_volume"].transform(_volume_zscore)
)

# Viral periods = unusually high activity
daily["viral_spike"] = (
    daily["volume_zscore"] > 2
).astype(int)

# ------------------------------------------
# 3. AMPLIFICATION MODEL
# ------------------------------------------

# Test whether amplification predicts momentum
amplification_formula = """
predicted_momentum
~ amplification_rate
+ viral_spike
+ C(state_id)
+ C(date_id)
"""

# HC1 is much faster than HC3 on large state×date FE panels (same point estimates).
amplification_model = smf.ols(
    formula=amplification_formula,
    data=daily,
).fit(cov_type="HC1")

print("\n=== AMPLIFICATION AUDIT ===")
print(amplification_model.summary())

# ------------------------------------------
# 4. CONCENTRATION ANALYSIS
# ------------------------------------------

state_engagement = (
    daily.groupby("state")
    .agg(total_volume=("tweet_volume", "sum"))
    .reset_index()
)

# Share of national engagement
total_all = state_engagement["total_volume"].sum()

state_engagement["engagement_share"] = (
    state_engagement["total_volume"] / total_all
)

# Herfindahl-Hirschman Index (HHI)
hhi = (
    state_engagement["engagement_share"] ** 2
).sum()

print("\n=== ENGAGEMENT CONCENTRATION ===")
print(state_engagement.sort_values(
    "engagement_share",
    ascending=False
).head(10))

print(f"\nHHI Concentration Index: {hhi:.4f}")

# ------------------------------------------
# 5. VISUALIZATION: VIRAL SPIKES
# ------------------------------------------

plt.figure(figsize=(12, 6))

sns.scatterplot(
    data=daily,
    x="tweet_volume",
    y="predicted_momentum",
    hue="viral_spike",
    alpha=0.7
)

plt.title(
    "Amplification Bias: Viral Activity vs Predicted Momentum"
)

plt.xlabel("Tweet Volume")
plt.ylabel("Predicted Momentum")

plt.tight_layout()
plt.show()

# ------------------------------------------
# 6. EVENT-LEVEL AMPLIFICATION
# ------------------------------------------

event_summary = daily.groupby("viral_spike").agg({
    "tweet_volume": "mean",
    "reply_volume": "mean",
    "predicted_momentum": "mean",
    "amplification_rate": "mean"
})

print("\n=== VIRAL VS NON-VIRAL PERIODS ===")
print(event_summary)

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


def _demean_twfe(data, col):
    """Two-way fixed-effects within transform (state + calendar day)."""
    s = data[col].astype(float)
    return (
        s
        - s.groupby(data["state_id"]).transform("mean")
        - s.groupby(data["date_id"]).transform("mean")
        + s.mean()
    )


def run_event_did_coef_fast(data, event_name):
    """
    DiD coefficient with state and date FE via within transformation.
    Same beta as OLS with C(state_id)+C(date_id); avoids 8× slow HC3 refits here.
    """
    did_col = f"{event_name}_did"
    post_col = f"{event_name}_post"
    if post_col not in data.columns or data[post_col].sum() == 0:
        return np.nan
    y = _demean_twfe(data, "log_volume")
    x = _demean_twfe(data, did_col)
    ok = y.notna() & x.notna()
    yv, xv = y[ok].to_numpy(), x[ok].to_numpy()
    denom = float(np.dot(xv, xv))
    if denom == 0:
        return np.nan
    return float(np.dot(xv, yv) / denom)


# Reuse main DiD table from the panel cell when available (full OLS+HC3).
_effects_coef = {}
if "effects" in globals() and hasattr(effects, "iterrows"):
    _effects_coef = dict(zip(effects["event"], effects["coef"]))

print(
    "\nSpike-sensitivity DiD (fast FE transform; "
    "coef_full from 'effects' when present)..."
)
amp_rows = []
for event_name in events:
    print(f"  {event_name}...", flush=True)
    amp_rows.append({
        "event": event_name,
        "coef_full": _effects_coef.get(
            event_name, run_event_did_coef_fast(daily, event_name)
        ),
        "coef_no_spike": run_event_did_coef_fast(daily_no_spike, event_name),
    })

amp = pd.DataFrame(amp_rows)
amp["abs_change"] = (amp["coef_full"] - amp["coef_no_spike"]).abs()
print(f"\n{'=' * 60}\n18. AMPLIFICATION AUDIT (spike sensitivity)\n{'=' * 60}")
print(f"Spike days flagged (top {1 - SPIKE_QUANTILE:.0%} within state): {n_spike:,}")
print("DiD coefficient with vs without viral spike days:")
print(amp.round(4).to_string(index=False))