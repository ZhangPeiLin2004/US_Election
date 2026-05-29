import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# =========================
# 1. LOAD DATA (CHUNKED)
# =========================
file_path = r"E:\Csci_3\subjects_AND_sampling_metadata_anonymized_full.csv"
df_iter = pd.read_csv(file_path, chunksize=10000)

start_date = "2024-01-01"
end_date = "2024-11-05"

all_comments = []

# =========================
# 2. EXTRACT RELEVANT COMMENTS
# =========================
for i, chunk in enumerate(df_iter):
    print(f"Processing chunk {i}...")

    data = chunk.copy()

    # Convert datetime
    data['created_at'] = pd.to_datetime(data['created_at'], errors='coerce')

    # Filter date range
    df_2024 = data[
        (data['created_at'] >= start_date) &
        (data['created_at'] <= end_date)
    ]

    # Keep replies only (comments)
    df_comments = df_2024[
        df_2024['in_reply_to_user_id_code'].notna()
    ]

    # Keep rows with state + text
    df_comments = df_comments[
        df_comments['clean...state_simple'].notna()
    ]

    df_comments = df_comments[
        ['tweets_historical', 'clean...state_simple']
    ].dropna()

    all_comments.append(df_comments)

# Combine all chunks
comments_df = pd.concat(all_comments, ignore_index=True)

print("\nTotal comments collected:", len(comments_df))


# =========================
# 3. FILTER ELECTION-RELATED COMMENTS
# =========================
keywords = [
    "election", "vote", "voting", "ballot", "president",
    "trump", "biden", "democrat", "republican",
    "campaign", "poll", "electoral", "debate"
]

pattern = "|".join(keywords)

filtered_df = comments_df[
    comments_df['tweets_historical'].str.contains(pattern, case=False, na=False)
]

print("Election-related comments:", len(filtered_df))


# =========================
# 4. TEXT VECTORIZATION (LDA INPUT)
# =========================
print("\nVectorizing text...")

vectorizer = CountVectorizer(
    stop_words='english',
    max_df=0.95,
    min_df=10
)

X = vectorizer.fit_transform(filtered_df['tweets_historical'])

print("Document-term matrix shape:", X.shape)


# =========================
# 5. RUN LDA
# =========================
print("\nRunning LDA...")

n_topics = 10  # you can tune this
lda_model = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42
)

lda_model.fit(X)


# =========================
# 6. ASSIGN TOPICS TO COMMENTS
# =========================
topic_distribution = lda_model.transform(X)

# Assign dominant topic
filtered_df['topic'] = topic_distribution.argmax(axis=1)


# =========================
# 7. EXTRACT TOP WORDS PER TOPIC
# =========================
words = vectorizer.get_feature_names_out()

topic_summary = []

for topic_idx, topic in enumerate(lda_model.components_):
    top_words = [words[i] for i in topic.argsort()[-10:]]
    topic_summary.append({
        "topic": topic_idx,
        "top_words": ", ".join(top_words)
    })

topic_summary_df = pd.DataFrame(topic_summary)

print("\nTopic Overview:")
print(topic_summary_df.head(10))


# =========================
# 8. SAVE RESULTS
# =========================
filtered_df.to_csv("comments_with_topics.csv", index=False)
topic_summary_df.to_csv("topic_summary.csv", index=False)

print("\nSaved:")
print("- comments_with_topics.csv")
print("- topic_summary.csv")


# =========================
# 9. STATE-TOPIC ANALYSIS
# =========================
state_topic_counts = (
    filtered_df
    .groupby(['clean...state_simple', 'topic'])
    .size()
    .reset_index(name='count')
)

state_topic_counts.to_csv("state_topic_counts.csv", index=False)

print("\nSaved:")
print("- state_topic_counts.csv")


# =========================
# 10. OPTIONAL: PRINT SAMPLE TOPICS
# =========================
print("\nSample Topics:\n")

for _, row in topic_summary_df.head(5).iterrows():
    print(f"Topic {row['topic']}:")
    print(row['top_words'])
    print("-" * 40)