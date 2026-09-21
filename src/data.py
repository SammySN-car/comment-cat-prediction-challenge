import re
import pandas as pd
import numpy as np


def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    return train, test


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9!?. ,\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def handling_features(df):
    df = df.copy()

    df['race'] = df['race'].fillna('missing')
    df['gender'] = df['gender'].fillna('missing')
    df['religion'] = df['religion'].fillna('missing')
    df['comment'] = df['comment'].fillna('')

    df['comment'] = df['comment'].apply(clean_text)

    df['disability'] = df['disability'].astype(int)

    df['hour'] = pd.to_datetime(
        df['created_date'],
        format='%Y-%m-%d %H:%M:%S.%f%z',
        utc=True
    ).dt.hour
    df = df.drop(columns=['created_date'])
    df = df.drop(columns=['post_id'])

    df['word_len'] = df['comment'].fillna('').apply(
        lambda x: len(str(x).split())
    )
    df['char_len'] = df['comment'].fillna('').apply(len)
    df['upvote_ratio'] = df['upvote'] / (df['upvote'] + df['downvote'] + 1e-6)
    df['emotion_total'] = df['emoticon_1'] + df['emoticon_2'] + df['emoticon_3']
    df['if2_emo_int'] = df['if_2'] * df['emotion_total']
    df['if2_upvote_rat_int'] = df['if_2'] * df['upvote_ratio']

    return df


def prepare_data(train, test):
    Y = train['label']
    X = train.drop(columns=['label'])

    X = handling_features(X)
    test = handling_features(test)

    cat_cols = ['race', 'gender', 'religion']
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    return X, Y, test, cat_cols, num_cols
