import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import hstack, csr_matrix

from src.utils import load_config, set_seed
from src.data import load_data, prepare_data
from src.features import build_preprocessor, build_tfidf, create_features
from src.models.lr import build_lr
from src.models.sgd import build_sgd
from src.models.svm import build_svm
from src.models.xgb import build_xgb
from src.models.stacking import build_stacking
from src.evaluation import evaluate, print_report
from src.ensemble import create_submission


def main():
    config = load_config()
    set_seed(config['seed'])

    data_dir = config['data'].get('data_dir', './data')
    output_dir = config['data'].get('output_dir', './output')
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(data_dir, config['data']['train_file'])
    test_path = os.path.join(data_dir, config['data']['test_file'])

    print("Loading data...")
    raw_train, raw_test = load_data(train_path, test_path)

    X, Y, test, cat_cols, num_cols = prepare_data(raw_train, raw_test)

    n_splits = config['cv']['n_splits']
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config['seed'])

    preprocessor = build_preprocessor(num_cols, cat_cols)
    tfidf_word, tfidf_char = build_tfidf(config['tfidf']['word'], config['tfidf']['char'])

    lr_scores = []
    sgd_scores = []
    svm_scores = []
    xgb_scores = []

    last_fold_preds = {}

    print(f"\n{'='*50}")
    print(f"Training with {n_splits}-Fold Stratified CV")
    print(f"{'='*50}")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, Y)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = Y.iloc[train_idx], Y.iloc[val_idx]

        X_train_feat = create_features(X_train, preprocessor, tfidf_word, tfidf_char, fit=True)
        X_val_feat = create_features(X_val, preprocessor, tfidf_word, tfidf_char, fit=False)

        lr_model = build_lr(config['models']['lr'])
        lr_model.fit(X_train_feat, y_train)
        lr_pred = lr_model.predict(X_val_feat)
        lr_scores.append(evaluate(y_val, lr_pred))

        sgd_model = build_sgd(config['models']['sgd'])
        sgd_model.fit(X_train_feat, y_train)
        sgd_pred = sgd_model.predict(X_val_feat)
        sgd_scores.append(evaluate(y_val, sgd_pred))

        svm_model = build_svm(config['models']['svm'])
        svm_model.fit(X_train_feat, y_train)
        svm_pred = svm_model.predict(X_val_feat)
        svm_scores.append(evaluate(y_val, svm_pred))

        xgb_model = build_xgb(config['models']['xgb'])
        xgb_model.fit(X_train_feat, y_train)
        xgb_pred = xgb_model.predict(X_val_feat)
        xgb_scores.append(evaluate(y_val, xgb_pred))

        last_fold_preds = {
            'y_true': y_val,
            'lr': lr_pred,
            'sgd': sgd_pred,
            'svm': svm_pred,
            'xgb': xgb_pred,
        }

        print(f"  LR:  {lr_scores[-1]:.4f}")
        print(f"  SGD: {sgd_scores[-1]:.4f}")
        print(f"  SVM: {svm_scores[-1]:.4f}")
        print(f"  XGB: {xgb_scores[-1]:.4f}")

        del X_train_feat, X_val_feat, y_train, y_val
        del lr_model, sgd_model, svm_model, xgb_model
        gc.collect()

    print(f"\n{'='*50}")
    print("Model Comparison (Mean F1 Macro)")
    print(f"{'='*50}")
    print(f"LR:  {np.mean(lr_scores):.4f}")
    print(f"SGD: {np.mean(sgd_scores):.4f}")
    print(f"SVM: {np.mean(svm_scores):.4f}")
    print(f"XGB: {np.mean(xgb_scores):.4f}")

    print(f"\n--- Classification Reports (Last Fold) ---")
    for name in ['lr', 'sgd', 'svm', 'xgb']:
        print(f"\n{name.upper()}:")
        print_report(last_fold_preds['y_true'], last_fold_preds[name])

    print(f"\n{'='*50}")
    print("Training Final Stacking Model")
    print(f"{'='*50}")

    X_final = create_features(X, preprocessor, tfidf_word, tfidf_char, fit=True)
    test_final = create_features(test, preprocessor, tfidf_word, tfidf_char, fit=False)

    stack = build_stacking(config['stacking'], config['models']['xgb'])
    stack.fit(X_final, Y)
    preds = stack.predict(test_final)

    sample_path = os.path.join(data_dir, config['data'].get('sample_file', 'Sample.csv'))
    if os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        test_ids = sample.iloc[:, 0]
    else:
        test_ids = np.arange(len(test))

    create_submission(test_ids, preds, os.path.join(output_dir, 'submission.csv'))

    print("\nDone!")


if __name__ == '__main__':
    main()
