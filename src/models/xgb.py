from xgboost import XGBClassifier


def build_xgb(config):
    return XGBClassifier(**config)
