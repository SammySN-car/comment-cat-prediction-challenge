import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score


def f1_macro_lgb(y_true, y_pred):
    y_pred = np.argmax(y_pred.reshape(-1, 4), axis=1)
    return 'f1_macro', f1_score(y_true, y_pred, average='macro'), True


def build_lgb(config):
    return lgb.LGBMClassifier(**config)
