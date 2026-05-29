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

state_features["absolute_error"] = (
    state_features["prediction_error"].abs()
)

print("\n=== REPRESENTATION BIAS TEST ===")

print(
    state_features[
        [
            "state",
            "engagement_rate",
            "predicted_margin",
            "final_margin",
            "absolute_error"
        ]
    ]
)

# Scatterplot

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=state_features,
    x="engagement_rate",
    y="absolute_error",
    s=120
)

for _, row in state_features.iterrows():
    plt.text(
        row["engagement_rate"],
        row["absolute_error"],
        row["state"]
    )

plt.title(
    "Prediction Error vs Engagement Intensity"
)

plt.xlabel("Engagement Rate")
plt.ylabel("Absolute Prediction Error")

plt.tight_layout()
plt.show()