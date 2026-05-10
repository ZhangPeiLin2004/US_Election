import pandas as pd

# Load dataset
df = pd.read_csv("comments_with_topics.csv")

# Make a copy
swing_df = df.copy()

# Define swing states
swing_states = [
    "Arizona",
    "Georgia",
    "Michigan",
    "Nevada",
    "North Carolina",
    "Pennsylvania",
    "Wisconsin"
]

# Keep ONLY rows where state_simple is a swing state
swing_df = swing_df[
    swing_df["clean...state_simple"].isin(swing_states)
]

# Count comments for each state and topic
counts = (
    swing_df
    .groupby(["clean...state_simple", "topic"])
    .size()
    .reset_index(name="count")
)

# Sort output
counts = counts.sort_values(
    by=["clean...state_simple", "topic"]
)

# Print results
print(counts)

# Save counts file
counts.to_csv("swing_state_topic_counts.csv", index=False)

# Optional: show states included
print("\nSwing states found:")
print(swing_df["clean...state_simple"].unique())