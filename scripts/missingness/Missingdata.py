# %%
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"E:\Csci_3\subjects_AND_sampling_metadata_anonymized_full.csv"
chunksize = 100_000

# STEP 1 — Overall Missingness for all variables
missing_counts = None
total_rows = 0

for chunk in pd.read_csv(file_path, chunksize=chunksize, dtype=str, low_memory=False):
    if missing_counts is None:
        missing_counts = chunk.isna().sum()
    else:
        missing_counts += chunk.isna().sum()
    total_rows += len(chunk)

missing_percent = (missing_counts / total_rows) * 100
missing_percent = missing_percent.sort_values(ascending=False)

print("Total rows:", total_rows)
print("\nMissing % by column:")
print(missing_percent)

# %%
# STEP 2 — Identify Structural Missingness
fully_missing = missing_percent[missing_percent == 100]
print("\n100% Missing Columns (Structural):")
print(fully_missing)

# %%
# STEP 3 — Define categories and variables
# categories dictionary is reused here
categories = {
    'User info': ['created_at.users', 'protected', 'verified_type', 'public_metrics.followers_count',
                  'public_metrics.following_count', 'public_metrics.tweet_count', 'public_metrics.listed_count',
                  'public_metrics.like_count.users', 'public_metrics.media_count', 'user_id_code'],
    'Tweet content / metadata': ['sampling_tweet', 'tweets_historical', 'lang.subject_pool', 'lang.tweets_historical',
                                 'entities.url.urls', 'entities.description.mentions', 'entities.description.hashtags',
                                 'entities.description.urls', 'entities.mentions.subject_pool', 'entities.urls.subject_pool',
                                 'entities.annotations.subject_pool', 'entities.hashtags.subject_pool',
                                 'entities.cashtags.subject_pool', 'entities.description.cashtags',
                                 'attachments.media_keys.subject_pool', 'attachments.media_source_tweet_id.subject_pool',
                                 'attachments.poll_ids.subject_pool', 'edit_history_tweet_ids.subject_pool',
                                 'possibly_sensitive'],
    'Engagement metrics': [],  # to fill after reading first chunk
    'Sampling / dataset info': ['pool_type', 'sampling_date', 'clean...pool', 'clean...pool.date',
                                'clean...state_simple', 'EXCLUDED', 'TEST', 'EXTRACTED_USER_TIMELINE'],
    'Conversation / reply tracking': ['conversation_id_code', 'conversation_id_historical_code',
                                      'in_reply_to_user_id_code', 'in_reply_to_user_id_historical_code',
                                      'most_recent_tweet_id_code', 'pinned_tweet_id_code', 'tweet_id_code',
                                      'tweet_id_historical_code'],
    'Articles / external references': ['article.title']
}

# First chunk to detect all engagement metrics dynamically
first_chunk = pd.read_csv(file_path, nrows=chunksize, dtype=str, low_memory=False)
categories['Engagement metrics'] = [col for col in first_chunk.columns if 'public_metrics' in col]

# %%
# STEP 4 — Create Missingness Indicators for all variables
all_vars = [col for sublist in categories.values() for col in sublist]
sample_cols = all_vars.copy()
sample_df_list = []

for chunk in pd.read_csv(file_path, chunksize=chunksize, usecols=lambda x: x in sample_cols, dtype=str, low_memory=False):
    # Create missingness indicators
    for col in chunk.columns:
        chunk[col + "_Missing"] = chunk[col].isna().astype(int)
    # Sample up to 5000 rows per chunk
    sample_df_list.append(chunk.sample(min(5000, len(chunk))))
    if len(sample_df_list) > 20:
        break

df_sample = pd.concat(sample_df_list, ignore_index=True)

# %%
# STEP 5 — Summary of Missingness by variable
missing_indicator_cols = [col for col in df_sample.columns if col.endswith("_Missing")]
print("\nMissingness distribution for sampled variables:")
for col in missing_indicator_cols:
    print(f"{col}:")
    print(df_sample[col].value_counts(normalize=True))

# %%
# STEP 6 — Convert numeric engagement metrics for MCAR tests
numeric_cols = ['public_metrics.like_count', 'public_metrics.retweet_count.tweets_historical']
for col in numeric_cols:
    if col in df_sample.columns:
        df_sample[col] = pd.to_numeric(df_sample[col], errors="coerce")

# Drop rows missing numeric predictors for MCAR tests
df_sample = df_sample.dropna(subset=[col for col in numeric_cols if col in df_sample.columns])

# %%
# STEP 7 — MCAR Tests for all partially missing variables
for var in missing_indicator_cols:
    y = df_sample[var]
    if y.nunique() > 1:  # Only test variables with variation
        X = df_sample[[col for col in numeric_cols if col in df_sample.columns]]
        X = sm.add_constant(X)
        model = sm.Logit(y, X)
        result = model.fit(disp=False)
        print(f"\nMCAR Test Summary for {var}:")
        print(result.summary())
    else:
        print(f"\n{var} has no missing values; MCAR test skipped.")

classification = {}

for var in missing_indicator_cols:
    y = df_sample[var]

    # skip constants
    if y.nunique() <= 1:
        classification[var] = "Structural (Always Missing or Never Missing)"
        continue

    X = df_sample[[col for col in numeric_cols if col in df_sample.columns]]
    X = sm.add_constant(X)

    try:
        result = sm.Logit(y, X).fit(disp=False)

        # check if ANY predictor significantly predicts missingness
        pvals = result.pvalues.drop("const", errors="ignore")

        if (pvals < 0.05).any():
            classification[var] = "MAR or MNAR"
        else:
            classification[var] = "MCAR"

    except:
        classification[var] = "Test Failed"

# convert to table
class_df = pd.DataFrame.from_dict(classification, orient="index", columns=["Missingness_Type"])
print(class_df.sort_values("Missingness_Type"))

content_keywords = ["entities", "hashtags", "urls", "mentions", "media", "poll"]

def refine(var, label):
    if label != "MAR or MNAR":
        return label
    if any(k in var for k in content_keywords):
        return "Likely MNAR (content-dependent)"
    return "Likely MAR (observed-data dependent)"

class_df["Refined"] = [refine(v, classification[v]) for v in class_df.index]

# %%
# STEP 8 — Correlation heatmap of missingness indicators
top_n = 7  # show top 15 missingness indicators
top_missing_cols = df_sample[missing_indicator_cols].mean().sort_values(ascending=False).head(top_n).index

sns.heatmap(df_sample[top_missing_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Top 7 Missingness Indicators Correlation")
plt.show()


# %%
# STEP 9 — Barplot of missingness %
# 1. Only keep variables with some missingness
missing_threshold = 0  # only show vars with >0% missing
vars_with_missing = [col for col in all_vars if df_sample[col].isna().mean() > missing_threshold]

# 2. Compute missing %
missing_percent_sample = df_sample[vars_with_missing].isna().mean() * 100

# 3. Shorten labels for readability
short_labels = {col: col.replace(".subject_pool", "")
                         .replace(".users", "")
                         .replace("public_metrics.", "")
                         .replace("clean...", "")
                         .replace("_code", "") 
                         for col in missing_percent_sample.index}

missing_percent_sample.rename(index=short_labels, inplace=True)

# 4. Plot
plt.figure(figsize=(10, len(vars_with_missing)*0.25))  # adjust height for many variables
missing_percent_sample.sort_values().plot(kind='barh', color='skyblue')
plt.xlabel("Missing %")
plt.title("Missing % by Variables (Filtered & Readable)")
plt.tight_layout()
plt.show()

counts = class_df["Refined"].value_counts()

report = (
    f"The dataset exhibits a mixed missingness structure. "
    f"{counts.get('Structural (Always Missing or Never Missing)',0)} variables are structurally missing, "
    f"{counts.get('MCAR',0)} appear Missing Completely At Random, "
    f"{counts.get('Likely MAR (observed-data dependent)',0)} are likely Missing At Random, "
    f"and {counts.get('Likely MNAR (content-dependent)',0)} are likely Missing Not At Random. "
    f"Overall, missingness is primarily systematic rather than random, indicating that absent values largely reflect platform or behavioral conditions rather than data corruption."
)

print(report)


# %%
