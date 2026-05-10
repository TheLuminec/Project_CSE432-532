import numpy as np

# Workings of a lot of these functions are inspired by scikit-learn's implementations

def _trapezoid_area(y_values, x_values):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y_values, x_values)
    return np.trapz(y_values, x_values)


def _as_arrays(y_true, y_pred):
    return np.asarray(y_true), np.asarray(y_pred)


def _resolve_labels(y_true, y_pred, labels=None):
    if labels is not None:
        return np.asarray(labels)
    return np.unique(np.concatenate((y_true, y_pred)))


def _confusion_counts(y_true, y_pred, labels):
    counts = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        support = np.sum(y_true == label)
        counts.append((tp, fp, fn, support))
    return counts


def _average_scores(scores, supports, average):
    if average in (None, "none"):
        return np.asarray(scores)
    if average == "macro":
        return float(np.mean(scores))
    if average == "weighted":
        total_support = np.sum(supports)
        if total_support == 0:
            return 0.0
        return float(np.average(scores, weights=supports))
    raise ValueError("average must be one of None, 'none', 'macro', 'weighted', 'micro', or 'binary'.")


def accuracy_score(y_true, y_pred):
    y_true, y_pred = _as_arrays(y_true, y_pred)
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, average=None, labels=None, pos_label=1, zero_division=0.0):
    y_true, y_pred = _as_arrays(y_true, y_pred)
    resolved_labels = _resolve_labels(y_true, y_pred, labels)

    if average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        denom = tp + fp
        return float(tp / denom) if denom > 0 else float(zero_division)

    if average is None:
        average = "binary" if len(resolved_labels) == 2 and set(resolved_labels).issubset({0, 1}) else "macro"
        if average == "binary":
            return precision_score(y_true, y_pred, average="binary", labels=labels, pos_label=pos_label, zero_division=zero_division)
    if average == "micro":
        tp = np.sum(y_true == y_pred)
        total_predicted = len(y_pred)
        return float(tp / total_predicted) if total_predicted > 0 else float(zero_division)

    counts = _confusion_counts(y_true, y_pred, resolved_labels)
    scores = []
    supports = []
    for tp, fp, _, support in counts:
        denom = tp + fp
        scores.append(tp / denom if denom > 0 else zero_division)
        supports.append(support)
    return _average_scores(scores, np.asarray(supports), average)


def recall_score(y_true, y_pred, average=None, labels=None, pos_label=1, zero_division=0.0):
    y_true, y_pred = _as_arrays(y_true, y_pred)
    resolved_labels = _resolve_labels(y_true, y_pred, labels)

    if average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        denom = tp + fn
        return float(tp / denom) if denom > 0 else float(zero_division)

    if average is None:
        average = "binary" if len(resolved_labels) == 2 and set(resolved_labels).issubset({0, 1}) else "macro"
        if average == "binary":
            return recall_score(y_true, y_pred, average="binary", labels=labels, pos_label=pos_label, zero_division=zero_division)
    if average == "micro":
        tp = np.sum(y_true == y_pred)
        total_true = len(y_true)
        return float(tp / total_true) if total_true > 0 else float(zero_division)

    counts = _confusion_counts(y_true, y_pred, resolved_labels)
    scores = []
    supports = []
    for tp, _, fn, support in counts:
        denom = tp + fn
        scores.append(tp / denom if denom > 0 else zero_division)
        supports.append(support)
    return _average_scores(scores, np.asarray(supports), average)


def f1_score(y_true, y_pred, average=None, labels=None, pos_label=1, zero_division=0.0):
    y_true, y_pred = _as_arrays(y_true, y_pred)
    resolved_labels = _resolve_labels(y_true, y_pred, labels)

    if average == "binary":
        precision = precision_score(
            y_true,
            y_pred,
            average="binary",
            labels=labels,
            pos_label=pos_label,
            zero_division=zero_division,
        )
        recall = recall_score(
            y_true,
            y_pred,
            average="binary",
            labels=labels,
            pos_label=pos_label,
            zero_division=zero_division,
        )
        denom = precision + recall
        return float((2 * precision * recall) / denom) if denom > 0 else float(zero_division)

    if average is None:
        average = "binary" if len(resolved_labels) == 2 and set(resolved_labels).issubset({0, 1}) else "macro"
        if average == "binary":
            return f1_score(y_true, y_pred, average="binary", labels=labels, pos_label=pos_label, zero_division=zero_division)
    if average == "micro":
        precision = precision_score(y_true, y_pred, average="micro", labels=labels, zero_division=zero_division)
        recall = recall_score(y_true, y_pred, average="micro", labels=labels, zero_division=zero_division)
        denom = precision + recall
        return float((2 * precision * recall) / denom) if denom > 0 else float(zero_division)

    counts = _confusion_counts(y_true, y_pred, resolved_labels)
    scores = []
    supports = []
    for tp, fp, fn, support in counts:
        precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
        recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
        denom = precision + recall
        scores.append((2 * precision * recall) / denom if denom > 0 else zero_division)
        supports.append(support)
    return _average_scores(scores, np.asarray(supports), average)


def confusion_matrix(y_true, y_pred, labels=None):
    y_true, y_pred = _as_arrays(y_true, y_pred)
    resolved_labels = _resolve_labels(y_true, y_pred, labels)
    n_classes = len(resolved_labels)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    class_to_idx = {label: idx for idx, label in enumerate(resolved_labels)}
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in class_to_idx and pred_label in class_to_idx:
            matrix[class_to_idx[true_label], class_to_idx[pred_label]] += 1
    return matrix


def roc_curve(y_true, y_score, pos_label=1):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    order = np.argsort(y_score)[::-1]
    y_true = (y_true[order] == pos_label).astype(int)
    y_score = y_score[order]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, y_score[threshold_idxs]]

    positives = np.sum(y_true)
    negatives = len(y_true) - positives

    tpr = tps / positives if positives > 0 else np.zeros_like(tps, dtype=float)
    fpr = fps / negatives if negatives > 0 else np.zeros_like(fps, dtype=float)
    return fpr, tpr, thresholds


def roc_auc_score(y_true, y_score, average="macro", labels=None):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    if y_score.ndim == 1:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        return float(_trapezoid_area(tpr, fpr))

    resolved_labels = _resolve_labels(y_true, y_true, labels)
    if y_score.shape[1] != len(resolved_labels):
        raise ValueError("y_score column count must match the number of class labels.")

    aucs = []
    supports = []
    for idx, label in enumerate(resolved_labels):
        binary_true = (y_true == label).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, y_score[:, idx], pos_label=1)
        aucs.append(float(_trapezoid_area(tpr, fpr)))
        supports.append(np.sum(binary_true))

    return _average_scores(aucs, np.asarray(supports), average)
