import pandas as pd
import numpy as np


def create_submission(test_ids, preds, output_path='submission.csv'):
    submission = pd.DataFrame({
        'post_id': test_ids,
        'label': preds
    })
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(submission.head(10))
    return submission


def blend_predictions(preds_list, weights):
    blended = np.zeros_like(preds_list[0], dtype=np.float64)
    for preds, weight in zip(preds_list, weights):
        blended += weight * preds
    return np.argmax(blended, axis=1)
