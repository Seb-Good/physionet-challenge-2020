import numpy as np


def compute_beta_score(labels, output, beta, num_classes, check_errors=True):
    # Check inputs for errors.
    if check_errors:
        if len(output) != len(labels):
            raise Exception('Numbers of outputs and labels must be the same.')

    # Populate contingency table.
    num_recordings = len(labels)

    fbeta_l = np.zeros(num_classes)
    gbeta_l = np.zeros(num_classes)
    fmeasure_l = np.zeros(num_classes)
    accuracy_l = np.zeros(num_classes)

    f_beta = 0
    g_beta = 0
    f_measure = 0
    accuracy = 0

    # Weight function
    C_l = np.ones(num_classes)

    for j in range(num_classes):
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for i in range(num_recordings):

            num_labels = np.sum(labels[i])

            if labels[i][j] and output[i][j]:
                tp += 1 / num_labels
            elif not labels[i][j] and output[i][j]:
                fp += 1 / num_labels
            elif labels[i][j] and not output[i][j]:
                fn += 1 / num_labels
            elif not labels[i][j] and not output[i][j]:
                tn += 1 / num_labels

        # Summarize contingency table.
        if (1 + beta ** 2) * tp + (fn * beta ** 2) + fp:
            fbeta_l[j] = float((1 + beta ** 2) * tp) / float(((1 + beta ** 2) * tp) + (fn * beta ** 2) + fp)
        else:
            fbeta_l[j] = 1.0

        if tp + fp + beta * fn:
            gbeta_l[j] = float(tp) / float(tp + fp + beta * fn)
        else:
            gbeta_l[j] = 1.0

        if tp + fp + fn + tn:
            accuracy_l[j] = float(tp + tn) / float(tp + fp + fn + tn)
        else:
            accuracy_l[j] = 1.0

        if 2 * tp + fp + fn:
            fmeasure_l[j] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            fmeasure_l[j] = 1.0

    for i in range(num_classes):
        f_beta += fbeta_l[i] * C_l[i]
        g_beta += gbeta_l[i] * C_l[i]
        f_measure += fmeasure_l[i] * C_l[i]
        accuracy += accuracy_l[i] * C_l[i]

    f_beta = float(f_beta) / float(num_classes)
    g_beta = float(g_beta) / float(num_classes)
    f_measure = float(f_measure) / float(num_classes)
    accuracy = float(accuracy) / float(num_classes)

    return f_beta, g_beta
