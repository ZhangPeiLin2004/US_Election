import pandas as pd

df = pd.read_csv("E:\Csci_3\subjects_AND_sampling_metadata_anonymized_full.csv", chunksize=100)
chunk = next(df)

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
    'Engagement metrics': [col for col in chunk.columns if 'public_metrics' in col],
    'Sampling / dataset info': ['pool_type', 'sampling_date', 'clean...pool', 'clean...pool.date',
                                'clean...state_simple', 'EXCLUDED', 'TEST', 'EXTRACTED_USER_TIMELINE'],
    'Conversation / reply tracking': ['conversation_id_code', 'conversation_id_historical_code',
                                      'in_reply_to_user_id_code', 'in_reply_to_user_id_historical_code',
                                      'most_recent_tweet_id_code', 'pinned_tweet_id_code', 'tweet_id_code',
                                      'tweet_id_historical_code'],
    'Articles / external references': ['article.title']
}

# Create a summary table
summary_data = []
for col in chunk.columns:
    # Find category
    category = None
    for cat, cols in categories.items():
        if col in cols:
            category = cat
            break
    if not category:
        category = 'Other'

    # Add row: column, category, dtype, example value
    summary_data.append({
        'Column': col,
        'Category': category,
        'Data Type': chunk[col].dtype,
        'Example Value': chunk[col].iloc[0]
    })

summary_df = pd.DataFrame(summary_data)

# Display all columns without truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(summary_df)