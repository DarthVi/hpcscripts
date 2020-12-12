from collections import Counter, OrderedDict
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score

def get_f1_score(y_true, y_pred, clsResults, nodename, labels=None):
    # list to store f1 values
    F = []
    # calculate global overall F1-score with weighted average
    f1 = f1_score(y_true, y_pred, average='weighted')
    F.append(f1)
    print('Overall score: %f.' % f1)

    # calculate F1-score for each class
    if labels is not None:
        test_all = f1_score(y_true, y_pred, average=None, labels=labels)
    else:
        test_all = f1_score(y_true, y_pred, average=None)

    F.extend(list(test_all))

    for i, f in enumerate(test_all):
        print('Fault: %d,  F1: %f.' % (i, f))

    clsResults[nodename] = F

def get_sensitivity(counts, mcm, clsResults, nodename):
    S = []

    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]

    all_sensitivity = tp / (tp + fn)
    overall_sensitivity = sum(all_sensitivity[x]*counts[x] for x in counts.keys())/sum(counts[x] for x in counts.keys())

    S.append(overall_sensitivity)
    S.extend(list(all_sensitivity))

    clsResults[nodename] = S

def get_specificity(counts, mcm, clsResults, nodename):
    S = []

    tn = mcm[:, 0, 0]
    fp = mcm[:, 0, 1]

    all_specificity = tn / (tn + fp)
    overall_specificity = sum(all_specificity[x] * counts[x] for x in counts.keys()) / sum(
        counts[x] for x in counts.keys())

    S.append(overall_specificity)
    S.extend(list(all_specificity))

    clsResults[nodename] = S

def get_FP_rate(counts, mcm, clsResults, nodename):
    S = []

    tn = mcm[:, 0, 0]
    fp = mcm[:, 0, 1]

    all_FP_rate = fp / (fp + tn)
    overall_FP_rate = sum(all_FP_rate[x] * counts[x] for x in counts.keys()) /sum(
        counts[x] for x in counts.keys())

    S.append(overall_FP_rate)
    S.extend(list(all_FP_rate))

    clsResults[nodename] = S

def get_FN_rate(counts, mcm, clsResults, nodename):
    S = []

    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]

    all_FN_rate = fn / (fn + tp)
    overall_FN_rate = sum(all_FN_rate[x] * counts[x] for x in counts.keys())/sum(
        counts[x] for x in counts.keys())

    S.append(overall_FN_rate)
    S.extend(list(all_FN_rate))

    clsResults[nodename] = S
