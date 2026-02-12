import pandas as pd

df = pd.read_csv("subjects_AND_sampling_metadata_anonymized_full.csv", chunksize=100)
for chunk in df:
    print(chunk.head())
    break