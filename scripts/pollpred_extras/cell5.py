# ================================
# MISSINGNESS AUDIT
# ================================

missing_by_state = (
    daily.groupby("state")
    .apply(lambda x: x.isna().mean())
)

print("\n=== MISSINGNESS BY STATE ===")
print(missing_by_state)

# Missing dates

date_counts = (
    daily.groupby("state")["date"]
    .nunique()
    .reset_index(name="observed_days")
)

print("\n=== OBSERVED DAYS ===")
print(date_counts)