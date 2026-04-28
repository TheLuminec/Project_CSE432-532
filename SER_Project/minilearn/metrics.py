import numpy as np

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    if len(classes) == 2 and set(classes).issubset({0, 1}):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    precisions = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return np.mean(precisions)

def recall_score(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    if len(classes) == 2 and set(classes).issubset({0, 1}):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    recalls = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return np.mean(recalls)

def f1_score(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    if len(classes) == 2 and set(classes).issubset({0, 1}):
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    f1s = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * (p * r) / (p + r) if (p + r) > 0 else 0.0)
    return np.mean(f1s)

def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        matrix[class_to_idx[t], class_to_idx[p]] += 1
    return matrix