# US_Election
US election files portfolio

This file is dedicated to US Election Tweets of the year 2024

Total rows (in the original datatset received): ~2.1 million
Unique users: ~928
Average tweets per user: ~2,314

Structure:
Each row = one historical tweet from a user. Multiple rows per user (one row per historical tweet). The tweets_historical field contains the actual tweet text and varies between rows for the same user. The sampling_tweet and id.tweets fields identify the tweet used to sample the user and stay constant per user

Rules of splitting is given by POSSUM
