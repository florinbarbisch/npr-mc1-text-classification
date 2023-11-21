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


### Feature Engineering


### Model Training


### Hyperparameter Tuning


## Evaluation


### Metric Selection


### Results Discussion


### Case Studies


## Conclusion


