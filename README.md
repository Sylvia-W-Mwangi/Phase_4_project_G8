# **Twitter Sentiment Analysis Project**

**This project focuses on analyzing public sentiment towards brands and products on Twitter using Natural Language Processing (NLP) techniques.** The goal is to classify tweets into positive, negative, or neutral sentiment categories, provide comparative insights between major brands like Apple and Google, and offer actionable recommendations for customer satisfaction. The Cross-Industry Standard Process for Data Mining (CRISP-DM) guides to project approach.

## Table of Contents
* [1. Business Understanding](#1-business-understanding)
* [2. Data Understanding](#2-data-understanding)
* [3. Data Preparation](#3-data-preparation)
* [4. Modeling](#4-modeling)
* [5. Evaluation](#5-evaluation)
* [6. Deployment & Recommendations](#6-deployment--recommendations)
* [7. Conclusion](#7-conclusion)
* [8. Libraries Used](#8-libraries-used)

## 1. Business Understanding

**In today's competitive landscape, understanding customer perceptions is paramount.** Social media, particularly platforms like Twitter, offers a rich source of real-time public sentiment. This project aims to leverage Twitter data to gain insights into customer emotions regarding various products and brands.

### Objectives
* **To classify tweet sentiments as positive, negative, or neutral.**
* **To compare sentiments and common word usage for Apple and Google products.**
* **To build and evaluate binary text classifiers for positive vs. negative emotions.**
* **To build and evaluate multiclass classifiers for positive, negative, and neutral emotions.**
* **To improve model performance, especially for minority classes.**

## 2. Data Understanding

### Dataset Overview
The dataset for this project is sourced from **data.world**, containing tweet texts and their associated sentiment labels towards specific brands or products.

**Initial Data Overview**: The dataset contains three columns: `tweet_text`, `emotion_in_tweet_is_directed_at`, and `is_there_an_emotion_directed_at_a_brand_or_product`.
* **`tweet_text`**: Contains the raw tweet content.
* **`emotion_in_tweet_is_directed_at`**: Indicates the specific product/brand the emotion is directed at (e.g., iPhone, Google, iPad). **This column had a significant number of missing values (5552 out of 8721).**
* **`is_there_an_emotion_directed_at_a_brand_or_product`**: The target variable, initially having four categories: 'No emotion toward brand or product' (majority class), 'Positive emotion', 'Negative emotion', and 'I can't tell'.

The dataset comprises **8721 entries**.
* **`tweet_text`**: **8720 non-null values (1 missing)**. **8693 unique tweets**.
* **`emotion_in_tweet_is_directed_at`**: **3169 non-null values (5552 missing)**. **9 unique brands/products**, with 'iPad' being the **top** (910 frequency).
* **`is_there_an_emotion_directed_at_a_brand_or_product`**: **8721 non-null values (no missing)**. **4 unique emotion categories**, with 'No emotion toward brand or product' being the **top** (5156 frequency).

### Exploratory Data Analysis (EDA)
* **Sentiment Distribution**: Visualizations confirmed significant class imbalance, especially with the 'No emotion toward brand or product' (later 'Neutral emotion') being the dominant category.
* **Brand-wise Sentiment**: Sentiment distribution for Apple and Google products was analyzed, revealing that Apple had more positive mentions, while Google had a higher proportion of neutral mentions.
* **Common Words**: Frequency distributions of words were analyzed for positive and negative emotions, as well as for Apple and Google specific tweets, to identify key terms associated with different sentiments and brands.

## 3. Data Preparation

### Data Cleaning Highlights:
* Duplicated rows were identified and removed (22 duplicates).
* Missing values in `tweet_text` (1 instance) were dropped.
* For sentiment classification, **'No emotion toward brand or product'** and **'I can't tell'** were combined into a 'Neutral emotion' category for multiclass, while only 'Positive emotion' and 'Negative emotion' were used for binary classification.

### Feature Engineering
**A robust text preprocessing pipeline was implemented to prepare the tweet data for modeling**:
* **Lowercase Conversion**: All text was converted to lowercase.
* **Remove Repeated Punctuation**: Multiple occurrences of punctuation (e.g., `...`, `!!!`) were reduced to a single instance.
* **Remove Bracketed Text**: Text enclosed in square brackets (e.g., `[link]`) was removed.
* **Remove URLs**: URLs (`http://`, `https://`, `www.`, `bit.ly/`) were removed.
* **Remove Tags & Hashtags**: HTML tags (`<.*?>+`) and hashtags (`#\w+`) were removed.
* **Remove Alphanumeric Words**: Words containing both letters and numbers (e.g., `3G`, `iPad2`) were removed.
* **Tokenization**: `TweetTokenizer` was used to split text into words, while also stripping Twitter handles (`@mention`).
* **Remove Empty Tokens & Filter by Length**: Empty tokens were removed, and tokens less than 3 characters long were filtered out.
* **Stop Word Removal**: Common English stop words (e.g., "the", "is", "a") were removed using NLTK's stopwords list.
* **Punctuation Removal**: Punctuation tokens were removed.
* **Stemming**: `PorterStemmer` was applied to reduce words to their root form (e.g., "running" to "run").
* **Join Tokens & Normalize Whitespace**: Cleaned tokens were joined back into strings.

**For feature extraction, `TfidfVectorizer` and `CountVectorizer` were employed to convert the preprocessed text into numerical features for machine learning models.**

## 4. Modeling

### Model Building
* **Binary Classification (Positive vs. Negative Emotions)**: The analysis focused on discriminating between positive and negative sentiments.
    * **Logistic Regression (Baseline & Tuned)**: A `Pipeline` with `TfidfVectorizer` and `LogisticRegression` was used. **`GridSearchCV` was applied for hyperparameter tuning, and `class_weight='balanced'` was used to address class imbalance.** The **`scoring`** metric for `GridSearchCV` was **`f1_weighted`**.
    * **Multinomial Naive Bayes (Tuned & with SMOTE)**: A `Pipeline` with `CountVectorizer` / `TfidfVectorizer` and `MultinomialNB` was used. **`GridSearchCV` was performed, and `SMOTE` was integrated into an `ImbPipeline` to handle imbalance.** The **`scoring`** metric for `GridSearchCV` was **`f1_weighted`**.
* **Multiclass Classification (Positive, Negative, Neutral Emotions)**:
    * **Support Vector Classifier (SVC)**: A `Pipeline` with `TfidfVectorizer` and `SVC` was used. **`class_weight='balanced'` and `RandomOverSampler` were explored to mitigate imbalance.** The `SVC` model included `probability=True` for ROC curve generation.
    * **K-Nearest Neighbors (KNN)**: A `Pipeline` with `TfidfVectorizer` and `KNeighborsClassifier` was used, with `RandomOverSampler` for imbalance.
    * **Multi-Layer Perceptron (MLP) Classifier**: A basic neural network was implemented.
        * **Data Preprocessing for MLP**: Features were scaled using `StandardScaler(with_mean=False)`, and target labels were encoded using `LabelEncoder`.
        * **MLP Model Parameters**: The baseline `MLPClassifier` used:
            * `hidden_layer_sizes`: (100,)
            * `activation`: 'relu'
            * `solver`: 'adam'
            * `max_iter`: 500
            * `random_state`: 42
            * `early_stopping`: True
            * `validation_fraction`: 0.1
        * **Hyperparameter Tuning for MLP**: `GridSearchCV` was performed with parameters including:
            * `hidden_layer_sizes`: [(50,), (100,), (50, 50), (40, 30)]
            * `alpha`: [0.0001, 0.001, 0.01]
            * **`scoring`** metric: **`f1_weighted`**

### OVERSAMPLING, UNDERSAMPLING AND COMBINED-SAMPLING USING MLP CLASSIFIER

To further address class imbalance in multiclass classification, various sampling strategies were applied to the MLP classifier:

* **Oversampling (RandomOverSampler)**:
    * **Strategy**: Increased the number of samples in minority classes ('Negative emotion' and 'Positive emotion') to approach the count of the majority 'Neutral emotion' class. Specifically, 'Negative emotion' was oversampled to 80% of 'Neutral emotion' count, and 'Positive emotion' to 60% of 'Neutral emotion' count.
    * **Tool**: `RandomOverSampler` from `imblearn`.
    * **Impact**: Accuracy slightly dropped to **50%** on validation. The precision and recall for 'Negative emotion' and 'Positive emotion' showed minor fluctuations, but overall imbalance issues persisted.

* **Undersampling (RandomUnderSampler)**:
    * **Strategy**: Reduced the number of samples in the majority 'Neutral emotion' class to balance the dataset. 'Neutral emotion' was undersampled to twice the 'Negative emotion' count, and 'Positive emotion' to 1.5 times the 'Negative emotion' count.
    * **Tool**: `RandomUnderSampler` from `imblearn`.
    * **Impact**: Accuracy significantly decreased to **41%** on validation. While 'Negative emotion' recall increased to 15%, 'Neutral emotion' recall dropped considerably, indicating significant information loss.

* **Combined Sampling (SMOTEENN)**:
    * **Strategy**: Applied a hybrid approach combining oversampling (SMOTE) and undersampling (ENN) to create synthetic samples for minority classes and remove noisy majority class samples.
    * **Tool**: `SMOTEENN` from `imblearn`.
    * **Strategy**: Targeted specific total sample counts after resampling: 'Negative emotion' to 4000, 'Neutral emotion' to 3000, and 'Positive emotion' to 3500.
    * **Impact**: Accuracy further decreased to **28%** on the test set. Precision and recall for 'Negative emotion' remained low (**Precision: 0.06, Recall: 0.32**), and performance for 'Neutral emotion' also suffered.

## 5. Evaluation

### Results & Evaluation
* **Binary Classifiers**
    * Logistic Regression:
        * **Baseline**: Achieved **~85% accuracy**. **However, recall for 'Negative emotion' was very low (0.08), indicating bias towards the majority 'Positive emotion' class.**
        * **Tuned & Balanced**: Accuracy slightly reduced to **~83-84%**. **Crucially, the recall for 'Negative emotion' significantly improved to ~0.58 with precision ~0.50, showing a better balance in handling the minority class.**
    * Multinomial Naive Bayes:
        * **Tuned**: Achieved **~86-87% accuracy**. **Precision (~0.60) and recall (~0.50) for 'Negative emotion' were better than the baseline Logistic Regression.**
        * **With SMOTE**: Accuracy slightly dropped to **~83%**. **Precision (~0.50) and recall (~0.51) for 'Negative emotion' achieved a good balance.**
    * **Key takeaway**: The binary classifiers, especially tuned Logistic Regression and Multinomial Naive Bayes with SMOTE, showed reasonable performance for distinguishing positive and negative sentiments, with efforts to balance minority class prediction.
* **Multiclass Classifiers**
    * **SVC, KNN, MLP (without advanced sampling)**: All multiclass models struggled significantly, with overall accuracies ranging from **52% to 60%**. **The precision and recall for 'Negative emotion' were particularly weak across these models (e.g., SVC: 0.18 precision, 0.46 recall).** **The class imbalance, with 'Neutral emotion' being a large majority, heavily biased these models.** **They found it difficult to define clear boundaries between the three sentiment categories.**
    * **MLP (with advanced sampling)**: Despite extensive efforts with oversampling, undersampling, and combined sampling techniques, the **MLP models continued to show low overall accuracy (down to 28% after combined sampling and tuning)** and weak performance for minority classes, indicating that these strategies were insufficient to overcome the dataset's complexities and inherent bias for multiclass sentiment classification.

## 6. Deployment & Recommendations

* **Continuous Sentiment Monitoring**: Implement social media strategies for ongoing tracking of public sentiment to support informed business decisions.
* **Product/Service Enhancement**: Use analyzed sentiment data to directly improve product features and service quality.
* **Competitor Analysis**: Extend the analysis to include industry competitors to gain insights into customer perceptions and identify opportunities for unique positioning.
* **Allocating resources for data annotation** (e.g., for minority classes or ambiguous 'Neutral' instances) to improve data quality and reduce subjectivity.

## 7. Conclusion

The project successfully aimed to develop a text classifier to accurately distinguish between positive, neutral, and negative sentiments, including identifying the reasons for such classifications. It also sought to compare sentiment towards Apple and Google products for competitive analysis and provide insights for increasing customer satisfaction.
The models developed in this project equip companies such as Apple and Google with the means to effectively track sentiment related to their events and products across social media. This allows businesses to remain aware of public sentiment regarding their competitors, potentially offering a competitive advantage. However, a key limitation of the analysis stems from the crowd-sourced dataset, particularly the inherent subjectivity of how the tweets were classified and a significant amount of missing data from one of the features.

## 8. Libraries Used
* Python
* **Pandas**: Data manipulation and analysis.
* **NumPy**: Numerical operations.
* **Matplotlib, Seaborn**: Data visualization.
* **NLTK**: Natural Language Toolkit for text preprocessing (tokenization, stopwords, stemming, VADER).
* **Hugging Face `transformers`**: (Implied, if future BERT-based models are used, though not directly in the provided notebook's execution).
* **TensorFlow/Keras**: (Implied, if neural networks beyond basic MLP are developed, though not directly in the provided notebook's execution).
* **Scikit-learn**: Machine learning models (Logistic Regression, Multinomial Naive Bayes, SVC, KNN, MLPClassifier), feature extraction (TfidfVectorizer, CountVectorizer), model selection (GridSearchCV), and evaluation metrics (classification_report, confusion_matrix, roc_curve, accuracy_score).
* **Imblearn**: For handling imbalanced datasets (**SMOTE, RandomOverSampler, RandomUnderSampler, SMOTEENN, ImbPipeline**).
* 
# **Authors**
[1. Sylvia Mwangi](#https://github.com/Sylvia-W-Mwangi)
[2. Soudie Okwaro](#https://github.com/EdgarSoudie)
[3. Ted Ronoh](#https://github.com/tedronoh-14)
[4. Veronica Aoko](#https://github.com/veronica1948)

