
# ================================
# 15. DYNAMIC TRAJECTORY MODEL
# ================================

daily = daily.sort_values(["state", "date"]).copy()

# Daily momentum proxy (within panel order; first day per state is NA)
daily["momentum"] = daily.groupby("state")["log_volume"].diff()

traj_data = daily.dropna(subset=["momentum"]).copy()

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
    data=traj_data,
).fit()

print("\n=== DYNAMIC TRAJECTORY MODEL ===")
print(trajectory_model.summary())

daily["predicted_momentum"] = np.nan
daily.loc[traj_data.index, "predicted_momentum"] = trajectory_model.predict(traj_data)

# ================================
# CONVERT MOMENTUM INTO TRAJECTORY
# ================================

daily["trajectory"] = (
    daily.groupby("state")["predicted_momentum"]
    .cumsum()
)

daily["trajectory_smooth"] = (
    daily.groupby("state")["trajectory"]
    .transform(lambda x: x.rolling(7, min_periods=1).mean())
)

# ================================
# NORMALIZE TO FINAL MARGINS
# ================================

final_margin_map = dict(zip(results["state"], results["final_margin"]))

for state in daily["state"].unique():
    mask = daily["state"] == state
    traj = daily.loc[mask, "trajectory_smooth"]
    if traj.std() == 0 or traj.isna().all():
        continue
    normalized = (traj - traj.mean()) / traj.std()
    final_margin = final_margin_map.get(state, 0)
    daily.loc[mask, "trajectory_scaled"] = normalized * 0.8 + final_margin

# ================================
# VISUALIZATION
# ================================

plt.figure(figsize=(15, 7))

sns.lineplot(
    data=daily[daily["treated"] == 1],
    x="date",
    y="trajectory_scaled",
    hue="state",
    linewidth=2,
)

for event_name, event_date in events.items():
    plt.axvline(
        pd.to_datetime(event_date, utc=True),
        linestyle="--",
        alpha=0.5,
    )

plt.title("Predicted Electoral Momentum Trajectories Across Swing States")
plt.xlabel("Date")
plt.ylabel("Predicted Trump Margin Trend")
plt.xticks(rotation=45)
plt.legend(title="Swing State")
plt.tight_layout()
plt.show()
