import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix


def build_preprocessor(num_cols, cat_cols):
    num_pipeline = Pipeline([
        ('scaler', PowerTransformer(method='yeo-johnson')),
    ])
    cat_pipeline = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    return preprocessor


def build_tfidf(word_config, char_config):
    tfidf_word = TfidfVectorizer(**word_config)
    tfidf_char = TfidfVectorizer(**char_config)
    return tfidf_word, tfidf_char


def create_features(X, preprocessor, tfidf_word, tfidf_char, fit=True):
    if fit:
        X_processed = preprocessor.fit_transform(X)
        tw = tfidf_word.fit_transform(X['comment'])
        tc = tfidf_char.fit_transform(X['comment'])
    else:
        X_processed = preprocessor.transform(X)
        tw = tfidf_word.transform(X['comment'])
        tc = tfidf_char.transform(X['comment'])
    X_final = hstack([csr_matrix(X_processed), tw, tc])
    return X_final
