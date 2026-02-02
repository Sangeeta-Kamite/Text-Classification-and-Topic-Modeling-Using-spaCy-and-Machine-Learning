# Text Classification and Topic Modeling Using spaCy and Machine Learning.
This project focuses on text classification and topic modeling using economic news data.It leverages **spaCy** for **NLP** preprocessing, **TF-IDF** for **feature extraction**, and machine learning models for classification. Additionally, it applies **LDA**, **NMF**, and **LSA** for topic modeling.

1. Data Loading and Preprocessing

   1. **Dataset:** economic_news.xlsx contains headline, text, and relevance labels.
   2. **Preprocessing Steps:**
        1. Combine headline and text into full_text.
        2. Convert text to lowercase.
        3. Remove special characters using regular expressions (RegEx).
        4. Use spaCy to:
            - Tokenize text.
            - Lemmatize words (convert them to base form).
            - Remove stopwords (common words like "the," "is," etc.).
        5. Target Variable (relevance):
            - Convert labels to binary (yes → 1, no → 0).

2. Text Vectorization Using TF-IDF

     1. TF-IDF (Term Frequency - Inverse Document Frequency) converts raw text into numerical features.
     2. Important Parameters:
        1. max_features=1000: Limits the vocabulary size to 1000 words.
        2. stop_words='english': Removes common English stopwords.
      Transforms text into numerical format for machine learning models.

**3. Text Classification**
**Models Used:**
1. Logistic Regression
   - A linear model that predicts relevance probability.
   - Uses class_weight='balanced' to handle imbalanced data.
2. Random Forest
   - An ensemble learning method based on decision trees.
3. Naïve Bayes
   - A probabilistic classifier useful for text classification.
Model Evaluation:
   - Confusion Matrix: Shows misclassification counts.
   - Classification Report: Includes precision, recall, and F1-score.

**4. Model Interpretability Using LIME**  
LIME (Local Interpretable Model-Agnostic Explanations)
   - Provides explanations for individual predictions.
   - Helps interpret black-box models (e.g., logistic regression).
How It Works:
   - Generates perturbed versions of input text.
   - Examines how prediction probability changes.

**5. Topic Modeling**
Extracts hidden themes in the dataset using:

**1.Latent Dirichlet Allocation (LDA)**
    - Probabilistic model that assigns words to topics.
**2. Non-negative Matrix Factorization (NMF)**
     - Factorizes document-term matrix into topic-based features.
**3. Latent Semantic Analysis (LSA)**
    -Uses Singular Value Decomposition (SVD) to reduce dimensionality.
**Visualization**
  - Extracts top words from each topic.
  - Helps understand dominant themes in economic news.

**Outputs**
1. Classifying Economic News as Relevant/Not Relevant
   Helps automate news filtering for finance applications.
2. Interpreting Model Decisions with LIME
   Makes the classification model explainable.
3. Identifying Key Topics in Economic News
   Useful for trends analysis, market research, and sentiment detection.

**Applications**
1. Financial News Filtering: Identifies relevant news for investors.
2. Market Trend Analysis: Discovers economic trends using topic modeling.
3. Fake News Detection: Flags misleading financial articles.
4. AI-Powered News Recommendation: Suggests articles based on relevance.
