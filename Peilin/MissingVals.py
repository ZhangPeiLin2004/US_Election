import pandas as pd

file_path = "subjects_AND_sampling_metadata_anonymized_full.csv"
chunksize = 100_000

missing_counts = None
total_rows = 0
column_names = None

print("Starting chunk-based missingness analysis...\n")

for chunk in pd.read_csv(
        file_path,
        chunksize=chunksize,
        dtype=str,          # Avoid mixed-type warnings
        low_memory=False
    ):

    # Store column names once
    if column_names is None:
        column_names = chunk.columns

    # Count missing values in this chunk
    chunk_missing = chunk.isnull().sum()

    if missing_counts is None:
        missing_counts = chunk_missing
    else:
        missing_counts += chunk_missing

    total_rows += len(chunk)

print(f"\nTotal rows processed: {total_rows:,}")

# Compute missing percentage
missing_percent = (missing_counts / total_rows) * 100
missing_percent = missing_percent.sort_values(ascending=False)

print("\n==============================")
print("FULL Missing Percentage by Column")
print("==============================\n")
print(missing_percent)

# Identify 100% missing columns (structural missingness)
fully_missing = missing_percent[missing_percent == 100]

print("\n==============================")
print("Columns with 100% Missing (Structural)")
print("==============================\n")
print(fully_missing)

# Identify partially missing columns
partial_missing = missing_percent[(missing_percent > 0) & (missing_percent < 100)]

print("\n==============================")
print("Columns with Partial Missingness")
print("==============================\n")
print(partial_missing.sort_values(ascending=False))

# Identify complete columns
no_missing = missing_percent[missing_percent == 0]

print("\n==============================")
print("Columns with 0% Missing")
print("==============================\n")
print(no_missing)

print("\nAnalysis complete.")
