from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_lr(config):
    return Pipeline([
        ('clf', LogisticRegression(**config))
    ])
