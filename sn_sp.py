from sklearn.metrics import confusion_matrix, roc_curve
import numpy as np

# Calculate Sensitivity and Specificity, or best threshold
def sn_sp(true_label, pred_label, cutoff=None):

    # Use a confusion matrix to calculate
    # true positives (TP), false positives (FP), 
    # true negatives (TN), and false negatives (FN).
    if not cutoff:
        tn, fp, fn, tp = confusion_matrix(true_label,
                                          pred_label).ravel()
        best_threshold = None
        best_label = None

    elif cutoff == 'best':
        
        fpr, tpr, thresholds = roc_curve(true_label,
                                         pred_label)

        # Calculate the Youden's index for each threshold and 
        # find the threshold corresponding to the maximum value.
        best_threshold = thresholds[np.argmax(tpr - fpr)]

        tn, fp, fn, tp = confusion_matrix(true_label,
                                          (pred_label>best_threshold).astype('int')).ravel()
        best_label = (pred_label>best_threshold).astype('int')

    else:

        # if threshold has been specified
        tn, fp, fn, tp = confusion_matrix(true_label,
                                          (pred_label>cutoff).astype('int')).ravel()
        best_threshold = cutoff
        best_label = (pred_label>best_threshold).astype('int')

    # Sensitivity
    sensitivity = tp / (tp + fn)

    # Specificity
    specificity = tn / (tn + fp)

    return sensitivity, specificity, best_threshold, best_label