"""annotation metrics; no model training or result persistence."""
import numpy as np
from sklearn import metrics as skm


def annotation(pred, truth, probabilities=None, classes=None):
    p, t = np.asarray(pred), np.asarray(truth)
    if t.ndim != 1 or p.shape != t.shape or not len(t):
        raise ValueError('Annotation labels must be aligned nonempty vectors')
    labels = np.unique(np.concatenate([t, p])) if classes is None else np.asarray(classes)
    if len(np.unique(labels)) != len(labels) or not np.isin(np.r_[t, p], labels).all():
        raise ValueError('Class vocabulary must be unique and contain all labels')
    # Average only over test-supported classes; absent classes are explicitly recorded.
    supported = labels[np.isin(labels, t)]
    precision, recall, f1, support = skm.precision_recall_fscore_support(
        t, p, labels=labels, zero_division=0)
    valid = support > 0
    result = dict(acc=float(skm.accuracy_score(t, p)),
                  f1_m=float(np.mean(f1[valid])), precision_macro=float(np.mean(precision[valid])),
                  recall_macro=float(np.mean(recall[valid])),
                  balanced_accuracy=float(skm.balanced_accuracy_score(t, p)),
                  per_class={str(k): dict(precision=float(a), recall=float(b), f1=float(c), support=int(n))
                             for k, a, b, c, n in zip(labels, precision, recall, f1, support)},
                  n_test_classes=len(supported), n_configured_classes=len(labels),
                  absent_test_classes=[str(k) for k in labels[~valid]])
    result['AvgCA'] = (result['acc'] + result['f1_m']) / 2
    if probabilities is not None:
        prob = np.asarray(probabilities, dtype=float)
        if prob.shape != (len(t), len(labels)) or not np.isfinite(prob).all():
            raise ValueError('Probability columns must match the declared class order')
        if np.any(prob < 0) or np.any(prob > 1) or not np.allclose(prob.sum(1), 1, atol=1e-5):
            raise ValueError('Expected normalized class probabilities')
        roc, ap = [], []
        for j, k in enumerate(labels):
            target = t == k
            if 0 < target.sum() < len(t):
                roc.append(skm.roc_auc_score(target, prob[:, j]))
                ap.append(skm.average_precision_score(target, prob[:, j]))
        result.update(AUROC_macro=float(np.mean(roc)) if roc else None,
                      AUPRC_macro=float(np.mean(ap)) if ap else None,
                      probability_metric_classes=len(roc), auprc_definition='one-vs-rest average_precision')
    return result
