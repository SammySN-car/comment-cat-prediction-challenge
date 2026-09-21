from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


def build_svm(config):
    return Pipeline([
        ('clf', LinearSVC(**config))
    ])
