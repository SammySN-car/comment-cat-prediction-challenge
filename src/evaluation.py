from sklearn.metrics import f1_score, classification_report, confusion_matrix


def evaluate(y_true, y_pred, average='macro'):
    score = f1_score(y_true, y_pred, average=average)
    return score


def print_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
