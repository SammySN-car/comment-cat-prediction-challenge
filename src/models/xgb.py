import torch
from xgboost import XGBClassifier


def build_xgb(config):
    config = config.copy()
    if config.get('device') == 'cuda':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return XGBClassifier(**config)
