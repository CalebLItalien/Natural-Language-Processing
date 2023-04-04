"""Evaluation Metrics

Author: Kristina Striegnitz and Caleb L'Italien

I affirm that I have carried out my academic endeavors with full
academic honesty. [Caleb L'Italien]

Complete this file for part 1 of the project.
"""
B_FSCORE = 1

def get_accuracy(y_pred, y_true):
    """Calculate the accuracy of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    index = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for prediction in y_pred:
        if prediction == 1:
            if prediction == y_true[index]:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if prediction == y_true[index]:
                true_negatives += 1
            else:
                false_negatives += 1
        index += 1
    if true_positives == 0 and false_positives == 0:
        return 0
    return (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)

def get_precision(y_pred, y_true):
    """Calculate the precision of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    index = 0
    true_positives = 0
    false_positives = 0
    for prediction in y_pred:
        if prediction == 1:
            if prediction == y_true[index]:
                true_positives += 1
            else:
                false_positives += 1
        index += 1
    if true_positives == 0 and false_positives == 0:
        return 0
    return true_positives / (true_positives + false_positives)

def get_recall(y_pred, y_true):
    """Calculate the recall of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    index = 0
    true_positives = 0
    false_negatives = 0
    for prediction in y_pred:
        if prediction == 1:
            if prediction == y_true[index]:
                true_positives += 1
        else:
            if prediction != y_true[index]:
                false_negatives += 1
        index += 1
    if true_positives == 0 and false_negatives == 0:
        return 0
    return true_positives / (true_positives + false_negatives)

def get_fscore(y_pred, y_true):
    """Calculate the f-score of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    if precision == 0 and recall == 0:
        return 0
    return ((B_FSCORE**2 + 1) * precision * recall) / (B_FSCORE**2 * precision + recall)

def evaluate(y_pred, y_true):
    """Calculate precision, recall, and f-score of the predicted labels
    and print out the results.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    print("Accuracy: ", str(round(int((get_accuracy(y_pred, y_true)* 100)), 0)) + "%")
    print("Precision: ", str(round(int((get_precision(y_pred, y_true)* 100)), 0)) + "%")
    print("Recall: ", str(round(int((get_recall(y_pred, y_true)* 100)), 0)) + "%")
    print("F-score: ", str(round(int((get_fscore(y_pred, y_true)* 100)), 0)) + "%")