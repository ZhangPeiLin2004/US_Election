'''
                    ┌──────────────────────────────┐
                    │   Swing-State Twitter Data   │
                    │  (Posts, Users, Engagement)  │
                    └──────────────┬───────────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     │                           │
                     ▼                           ▼
          ┌──────────────────┐       ┌────────────────────┐
          │   Data Cleaning   │       │   Event Selection  │
          │ • Missing values  │       │ • Debate           │
          │ • State filtering │       │ • Assassination    │
          │ • Date parsing    │       │ • Biden withdrawal │
          └─────────┬────────┘       └─────────┬──────────┘
                    │                          │
                    └─────────────┬────────────┘
                                  ▼
                   ┌───────────────────────────┐
                   │     Feature Engineering    │
                   │ • Engagement per tweet     │
                   │ • Engagement spikes        │
                   │ • Hashtag activity         │
                   │ • Topic extraction (LDA)   │
                   └─────────────┬─────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
              ▼                                     ▼
 ┌─────────────────────────┐          ┌─────────────────────────┐
 │ Difference-in-Difference│          │  NLP Topic Modeling     │
 │ • Treated vs control    │          │ • Ideology signals      │
 │ • Pre/post event windows│          │ • Issue salience        │
 │ • Lingering event effect│          │ • Emotional discourse   │
 └──────────────┬──────────┘          └──────────────┬──────────┘
                │                                    │
                └────────────────┬───────────────────┘
                                 ▼
                   ┌───────────────────────────┐
                   │   Predictive Poll Model    │
                   │ • Estimate poll movement   │
                   │ • Compare swing states     │
                   │ • Detect geographic bias   │
                   └─────────────┬─────────────┘
                                 │
               ┌─────────────────┴─────────────────┐
               │                                   │
               ▼                                   ▼
 ┌──────────────────────────┐        ┌──────────────────────────┐
 │ Technocratic Rationalism │        │ Frankfurt School Lens    │
 │ • Data predicts society  │        │ • Platforms shape voices │
 │ • Algorithms are useful  │        │ • Visibility is unequal  │
 │                          │        │ • Amplification = power  │
 └──────────────┬───────────┘        └──────────────┬───────────┘
                │                                   │
                └────────────────┬──────────────────┘
                                 ▼
                 ┌──────────────────────────────┐
                 │     Bias Audit & Critique     │
                 │ • Who dominates engagement?   │
                 │ • Twitter salience vs reality │
                 │ • Invisible populations       │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │  Ethnographic Interpretation  │
                 │ • Examine sampled posts       │
                 │ • Observe polarization        │
                 │ • Analyze emotional language  │
                 │ • Interpret online behavior   │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │      Integrated Conclusion    │
                 │ • Predictive usefulness       │
                 │ • Structural inequalities     │
                 │ • Socially embedded bias      │
                 └──────────────────────────────┘
'''