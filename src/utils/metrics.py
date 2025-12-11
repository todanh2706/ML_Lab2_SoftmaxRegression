import numpy as np


def _prepare_inputs(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples")
    return y_true, y_pred


def accuracy_score(y_true, y_pred):
    """
    Fraction of correct predictions.
    """
    y_true, y_pred = _prepare_inputs(y_true, y_pred)
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute confusion matrix.
    """
    y_true, y_pred = _prepare_inputs(y_true, y_pred)
    if labels is None:
        labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))
    labels = np.array(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    cm = np.zeros((labels.size, labels.size), dtype=int)
    for t, p in zip(y_true, y_pred):
        ti = label_to_idx.get(t)
        pi = label_to_idx.get(p)
        if ti is None or pi is None:
            continue
        cm[ti, pi] += 1
    return cm


def precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=None,
    average=None,
    zero_division=0,
):
    """
    Compute precision, recall, f1-score and support.
    Supports average=None or average="macro".
    """
    y_true, y_pred = _prepare_inputs(y_true, y_pred)
    if labels is None:
        labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))
    labels = np.array(labels)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    pred_sum = cm.sum(axis=0).astype(float)
    fp = pred_sum - tp
    fn = support - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.divide(
            tp,
            tp + fp,
            out=np.full_like(tp, float(zero_division), dtype=float),
            where=(tp + fp) != 0,
        )
        recall = np.divide(
            tp,
            tp + fn,
            out=np.full_like(tp, float(zero_division), dtype=float),
            where=(tp + fn) != 0,
        )

    f1_den = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        f1_den,
        out=np.full_like(precision, float(zero_division), dtype=float),
        where=f1_den != 0,
    )

    if average is None:
        return precision, recall, f1, support.astype(int)

    if average == "macro":
        return (
            float(precision.mean()),
            float(recall.mean()),
            float(f1.mean()),
            int(support.sum()),
        )

    raise ValueError(f"Unsupported average: {average}")


def classification_report(
    y_true,
    y_pred,
    labels=None,
    digits=4,
    zero_division=0,
):
    """
    Text summary of precision, recall, f1-score and support.
    """
    y_true, y_pred = _prepare_inputs(y_true, y_pred)
    if labels is None:
        labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))
    labels = np.array(labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=zero_division
    )

    total_support = support.sum()
    acc = accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=zero_division
    )

    weighted_p = (
        float(np.average(precision, weights=support)) if total_support > 0 else 0.0
    )
    weighted_r = (
        float(np.average(recall, weights=support)) if total_support > 0 else 0.0
    )
    weighted_f1 = (
        float(np.average(f1, weights=support)) if total_support > 0 else 0.0
    )

    head = f"{'':>9} {'precision':>10} {'recall':>9} {'f1-score':>9} {'support':>9}"
    lines = [head]
    for label, p, r, f, s in zip(labels, precision, recall, f1, support):
        lines.append(
            f"{str(label):>9} {p:10.{digits}f} {r:9.{digits}f} "
            f"{f:9.{digits}f} {int(s):9d}"
        )

    lines.append(
        f"\n{'accuracy':>9} {acc:>30.{digits}f} {int(total_support):9d}"
    )
    lines.append(
        f"{'macro avg':>9} {macro_p:10.{digits}f} {macro_r:9.{digits}f} "
        f"{macro_f1:9.{digits}f} {int(total_support):9d}"
    )
    lines.append(
        f"{'weighted avg':>9} {weighted_p:10.{digits}f} {weighted_r:9.{digits}f} "
        f"{weighted_f1:9.{digits}f} {int(total_support):9d}"
    )

    return "\n".join(lines)
