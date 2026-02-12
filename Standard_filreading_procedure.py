import pandas as pd

# Read a small chunk
df = pd.read_csv("subjects_AND_sampling_metadata_anonymized_full.csv", chunksize=100)
chunk = next(df)

print(chunk.columns.tolist())

pd.set_option('display.max_columns', None)
print(chunk.head())

print(chunk.dtypes)
