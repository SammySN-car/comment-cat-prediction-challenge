import torch
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


def build_stacking(config, xgb_config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    xgb_params = xgb_config.copy()
    xgb_params['device'] = device
    
    return StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, n_jobs=-1)),
            ('xgb', XGBClassifier(**xgb_params)),
            ('svm', LinearSVC(max_iter=2000, random_state=42))
        ],
        final_estimator=XGBClassifier(device=device, tree_method='hist'),
        stack_method='predict',
        cv=config.get('cv', 3),
        n_jobs=config.get('n_jobs', 1)
    )
