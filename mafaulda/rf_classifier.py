"""
Random Forest Classification and Evaluation (Pure-Python Version)

This module handles the training and evaluation steps of the rotating-machine
fault diagnosis pipeline, using our pure-python RandomForestClassifier.
"""

from typing import List, Dict
from mafaulda.random_forest import RandomForestClassifier
from mafaulda.logging_utils import log


def train_classifier(
    X_train: List[List[float]],
    y_train: List[str]
) -> RandomForestClassifier:
    """
    Initializes and trains our pure-python Random Forest classifier.
    """
    # Using 100 trees in pure Python for extremely fast execution and high accuracy
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(
    clf: RandomForestClassifier,
    X_test: List[List[float]],
    y_test: List[str],
    y_train_labels: List[str] = None
) -> List[str]:
    """
    Evaluates the performance of the trained Random Forest classifier on the test split,
    printing overall accuracy, labeled confusion matrix, and classification report.
    """
    y_pred = clf.predict(X_test)

    num_correct = sum(1 for yt, yp in zip(y_test, y_pred) if yt == yp)
    accuracy = num_correct / len(y_test)

    log("\n================ EVALUATION RESULTS ================", level=1)
    log(f"Overall Classification Accuracy: {accuracy * 100.0:.2f}%", level=1)
    log(f"Expected Accuracy from Paper:   ~98.49%", level=1)
    log("====================================================", level=1)

    # Labeled Confusion Matrix
    unique_labels = sorted(list(set(y_test + (y_train_labels or []))))
    cm = compute_confusion_matrix(y_test, y_pred, unique_labels)
    print_formatted_confusion_matrix(cm, unique_labels)

    # Classification Report
    print_classification_report(y_test, y_pred, unique_labels)

    return y_pred


def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str]
) -> List[List[int]]:
    """Computes a 2D confusion matrix."""
    n = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    cm = [[0] * n for _ in range(n)]

    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_idx and yp in label_to_idx:
            r = label_to_idx[yt]
            c = label_to_idx[yp]
            cm[r][c] += 1

    return cm


def print_formatted_confusion_matrix(
    cm: List[List[int]],
    labels: List[str]
) -> None:
    """Outputs a beautifully aligned ASCII representation of the confusion matrix."""
    num_classes = len(labels)
    max_label_len = max(len(lbl) for lbl in labels)
    col_width = max(max_label_len, 8)

    log("\n--- CONFUSION MATRIX (True \\ Predicted) ---", level=1)
    header = f"{'True Class':<{max_label_len}} |"
    for lbl in labels:
        header += f" {lbl:^{col_width}} |"
    log(header, level=1)
    log("-" * len(header), level=1)

    for i in range(num_classes):
        row_str = f"{labels[i]:<{max_label_len}} |"
        for j in range(num_classes):
            cell_val = cm[i][j]
            row_str += f" {cell_val:^{col_width}} |"
        log(row_str, level=1)
    log("-" * len(header), level=1)


def print_classification_report(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str]
) -> None:
    """Computes and prints a precision, recall, and F1-score report for each class."""
    log("\n--- CLASSIFICATION REPORT ---", level=1)
    max_label_len = max(len(lbl) for lbl in labels)

    header = f"{'Class':<{max_label_len}} | {'Precision':^10} | {'Recall':^10} | {'F1-Score':^10} | {'Support':^10}"
    log(header, level=1)
    log("-" * len(header), level=1)

    total_tp = 0
    total_samples = len(y_true)

    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        support = sum(1 for yt in y_true if yt == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0

        total_tp += tp
        log(f"{label:<{max_label_len}} | {precision:^10.4f} | {recall:^10.4f} | {f1:^10.4f} | {support:^10}", level=1)

    log("-" * len(header), level=1)
    macro_precision = sum(
        sum(1 for yt, yp in zip(y_true, y_pred) if yt == lbl and yp == lbl) /
        max(1, sum(1 for yp in y_pred if yp == lbl))
        for lbl in labels
    ) / len(labels)

    macro_recall = sum(
        sum(1 for yt, yp in zip(y_true, y_pred) if yt == lbl and yp == lbl) /
        max(1, sum(1 for yt in y_true if yt == lbl))
        for lbl in labels
    ) / len(labels)

    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0.0
    accuracy = total_tp / total_samples

    log(f"{'Accuracy':<{max_label_len}} | {'':^10} | {'':^10} | {accuracy:^10.4f} | {total_samples:^10}", level=1)
    log(f"{'Macro Average':<{max_label_len}} | {macro_precision:^10.4f} | {macro_recall:^10.4f} | {macro_f1:^10.4f} | {total_samples:^10}", level=1)
