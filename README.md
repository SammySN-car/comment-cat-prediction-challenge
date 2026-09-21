# Comment Category Prediction Challenge

A machine learning pipeline for classifying comments into categories using text features, numerical features, and categorical features.

## Problem

Given a comment with associated metadata (race, gender, religion, votes, emoticons), predict the comment category.

- **Evaluation Metric:** F1 Macro
- **Classes:** 4 categories
- **CV:** 3-Fold Stratified

## Architecture

5 models with stacking ensemble:

| Model | Type | Description |
|-------|------|-------------|
| **LogisticRegression** | Linear | Balanced class weights |
| **SGDClassifier** | Linear | Hinge loss (SVM-like) |
| **LinearSVC** | Linear | Support Vector Machine |
| **XGBoost** | Tree | GPU-accelerated gradient boosting |
| **Stacking** | Ensemble | LR + XGB + SVM with XGB meta-learner |

### Feature Engineering

- **Numeric:** upvote, downvote, emoticons, if_1, if_2
- **Categorical:** race, gender, religion (One-Hot Encoded)
- **Text:** TF-IDF word (1,2) 20K + char (3,5) 15K
- **Engineered:** word_len, char_len, upvote_ratio, emotion_total, if2_emo_int, if2_upvote_rat_int
- **Scaling:** PowerTransformer (yeo-johnson)

## Project Structure

    comment-cat-prediction-challenge/
    |-- config.yaml
    |-- requirements.txt
    |-- main.py
    |-- 24f2006661-notebook-t12026 (6).ipynb
    |-- src/
        |-- utils.py
        |-- data.py
        |-- features.py
        |-- evaluation.py
        |-- ensemble.py
        |-- models/
            |-- lr.py
            |-- sgd.py
            |-- svm.py
            |-- xgb.py
            |-- lgb.py
            |-- stacking.py

## Installation

```bash
git clone https://github.com/SammySN-car/comment-cat-prediction-challenge.git
cd comment-cat-prediction-challenge
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

### Required Data Files

- train.csv - Training data
- test.csv - Test data
- Sample.csv - Sample submission format

## License

MIT License