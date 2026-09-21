from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


def build_stacking(config, xgb_config):
    return StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, n_jobs=-1)),
            ('xgb', XGBClassifier(**xgb_config)),
            ('svm', LinearSVC(max_iter=2000, random_state=42))
        ],
        final_estimator=XGBClassifier(device='cuda', tree_method='hist'),
        cv=config.get('cv', 3),
        n_jobs=config.get('n_jobs', 1)
    )
