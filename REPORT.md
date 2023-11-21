# Natural Language Processing - Text classification on disaster tweets

## Introduction


### Objective


### Background


### Scope of the Report


## Machine Learning Methods


### Overview of Machine Learning


### Model 1: BERTweet


#### Theory


#### Reason for Selection


### Method 2: TF-IDF with HGBC


#### Theory


#### Reason for Selection


### Method 3: CNN


#### Theory


#### Reason for Selection


## System Description


### Dataset
The dataset used for the evaluation of the machine-learning methods comes from a Kaggle competition titled "Natural Language Processing with Disaster Tweets". It was designed to provide a gateway for data scientists entering the field of Natural Language Processing (NLP). 

The dataset comprises 10,000 tweets that have been manually classified into two categories: tweets that are about real disasters and tweets that are not. The competition is tailored for models that can distinguish between tweets that describe actual emergency situations from those that do not, even if disaster-related terminology is present. 

Each tweet sample in both the training and the test set is accompanied by metadata containing the following information:

1. The text of the tweet: This is the primary data to be used for the text classification task.
2. A keyword from that tweet: The keyword is a term from the tweet which may be related to disaster terms (some samples may have this blank).
3. The location the tweet was sent from: Like the keyword, this can be blank for some samples but when available provides context in relation to the geographic area of the tweet.

The files included in the dataset are as follows:

- `train.csv`: This file constitutes the training set containing examples to train the models.
- `test.csv`: This set is to test the models' performance in predicting whether new, unseen tweets are about real disasters or not.
- `sample_submission.csv`: This file provides a template for submitting predictions in the correct format to the Kaggle platform.

Regarding the structure of `train.csv`, it includes several columns:

- `id`: It serves as a unique identifier for each tweet.
- `text`: Contains the actual tweet text.
- `location`: States the geographic location where the tweet was sent from.
- `keyword`: This is a notable keyword extracted from the tweet's text.
- `target`: A binary indicator, only present in `train.csv`, which shows whether a tweet pertains to a real disaster (1) or not (0).

This dataset is particularly well-suited for this task due to its focus on NLP and its moderate size, allowing for experimentation with different models without requiring extensive computational resources. The inclusion of keywords and location data also provides the opportunity for feature engineering and the exploration of how additional metadata can support or enhance the performance of text classification models.

### Preprocessing
The preprocessing stage of the dataset is critical to ensure that the machine-learning models function effectively, without interference from irrelevant or redundant data. In this phase, we have decided not to utilize the 'keyword' and 'location' columns for classification purposes. While these columns might offer additional context that could enhance predictions in a hybrid model, the focus remains solely on the tweet text for this analysis.

Firstly, to prevent any leakage of training data into the test set, it was essential to scour the dataset for any duplicates. Upon inspection, 110 duplicate instances were discovered. To maintain the integrity of the data, only those duplicates that were consistent in their 'target' values were retained, effectively removing any potential contamination that could bias the models.

Subsequently, a distinct training and a test set were established to segregate the data effectively. This division aims to avoid any inadvertent gain of information about the test set during the training phase.

Further examination confirmed that no 'NA' values were present in the dataset, and due to the earlier removal of duplicates, there were no duplicate values left either. However, an imbalance was noted in the class distribution—there were unequal numbers of disaster and non-disaster tweets. This skewness necessitates careful consideration when choosing an evaluation metric to ensure it accurately reflects the models' performance.

Analysis of the length of the tweets revealed that most contained between 10 to 20 words. Few tweets were longer than 30 words or shorter than 5 words, which may present challenges for the models as shorter tweets could lack sufficient information for accurate classification.

In preparation for TF-IDF analysis, the tweets underwent further cleaning. Special HTML characters were converted to their corresponding correct characters, eliminating the risk of misinterpreting artifacts like "&amp" as valid words. All non-alpha characters such as hashtags, mentions, punctuation, conjunctions, etc., were stripped away, focusing analysis solely on the relevance of words themselves.

Our TF-IDF model training involved refining the stopwords list. The built-in list was too narrow, failing to filter out many common stopwords, so a more extensive list from the internet was utilized. Additionally, accents were removed, and all words were converted to lowercase to prevent TF-IDF from misidentifying identical words due to case differences or diacritics.

Notably, words that carry more negative connotations were found to be more relevant in tweets categorized as disasters, a pattern not observed in non-disaster tweets. This distinction might provide valuable insights into the characteristics that differentiate disaster tweets from their non-disaster counterparts.


### Feature Engineering


### Model Training


### Hyperparameter Tuning


## Evaluation


### Metric Selection


### Results Discussion


### Case Studies


## Conclusion


