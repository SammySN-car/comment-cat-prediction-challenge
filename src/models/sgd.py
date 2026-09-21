from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


def build_sgd(config):
    return Pipeline([
        ('clf', SGDClassifier(**config))
    ])
