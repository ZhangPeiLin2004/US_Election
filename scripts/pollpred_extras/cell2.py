# ================================
# ENGAGEMENT CONCENTRATION TEST
# ================================

state_engagement = (
    daily.groupby("state")
    .agg(
        total_volume=("tweet_volume", "sum"),
        avg_volume=("tweet_volume", "mean")
    )
    .reset_index()
)

# Share of all engagement
total_all = state_engagement["total_volume"].sum()

state_engagement["engagement_share"] = (
    state_engagement["total_volume"] / total_all
)

print("\n=== ENGAGEMENT CONCENTRATION ===")
print(state_engagement.sort_values(
    "engagement_share",
    ascending=False
))

# ----------------------------
# State abbreviation mapping
# ----------------------------
state_abbrev = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "Unknown": "NA"
}

# Apply abbreviation mapping
state_engagement["state_abbrev"] = state_engagement["state"].map(state_abbrev)

# Optional fallback (in case any state is missing)
state_engagement["state_abbrev"] = state_engagement["state_abbrev"].fillna(state_engagement["state"])

# ----------------------------
# Visualization
# ----------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=state_engagement.sort_values(
        "engagement_share",
        ascending=False
    ),
    x="state_abbrev",
    y="engagement_share"
)

plt.title("Share of Total Political Engagement by State")
plt.ylabel("Share of Total Engagement")
plt.xlabel("State")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()