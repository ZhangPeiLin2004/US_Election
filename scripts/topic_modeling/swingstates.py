import pandas as pd

df = pd.read_csv("state_topic_counts.csv")

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

# Filter
df_swing = df[df['clean...state_simple'].isin(swing_states)]

# Save to new file
df_swing.to_csv("swing_state_topic_counts_only.csv", index=False)

# Print preview
print(df_swing)
print("\nSaved to swing_state_topic_counts_only.csv")