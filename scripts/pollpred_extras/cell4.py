# ================================
# VIRAL SPIKE INEQUALITY TEST
# ================================

daily_total = (
    daily.groupby("date")["tweet_volume"]
    .sum()
    .reset_index()
)

threshold = daily_total["tweet_volume"].quantile(0.95)

viral_days = daily_total[
    daily_total["tweet_volume"] >= threshold
]

print("\n=== VIRAL DAYS ===")
print(viral_days)

# Visualization

plt.figure(figsize=(14,6))

sns.lineplot(
    data=daily_total,
    x="date",
    y="tweet_volume"
)

plt.axhline(
    threshold,
    linestyle="--"
)

plt.title("Daily Political Tweet Volume")
plt.ylabel("Tweet Volume")

plt.tight_layout()
plt.show()