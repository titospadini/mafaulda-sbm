"""
Random Forest Classification and Evaluation on Extended SBM Features

This module handles the classification step (Step 4) of the rotating-machine
fault diagnosis pipeline, exposing function interfaces for training and
evaluation.
"""

from typing import List
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from mafaulda.logging_utils import log


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> RandomForestClassifier:
    """
    Initializes and trains the Random Forest classifier using standardized
    hyperparameters.

    Pedagogical Context:
        The classification task is handled by a Random Forest Ensemble.
        Standardizing the forest's
        hyperparameters is crucial for model stability and reproducibility:
          - `n_estimators=500`: A high number of trees reduces voting variance
            and guarantees highly stable
            decision boundaries without increasing overfitting risk.
          - `max_features='sqrt'`: Selecting a random subset of $\\sqrt{d}$
            features at each node split
            decorrelates the individual trees, improving the overall ensemble
            robustness.
          - `class_weight='balanced'`: Automatically scales weights inversely
            proportional to class frequencies.
            This is critical because the MaFaulDa dataset is extremely
            unbalanced (e.g. only 49 normal files vs 558
            underhang files), ensuring that normal operating states are not
            ignored or misclassified.

    Parameters:
        X_train (np.ndarray): The extended feature matrix of shape (num_samples,
        num_features).
        y_train (np.ndarray): The 1D target label array.

    Returns:
        RandomForestClassifier: A fully trained Random Forest Classifier model.
    """
    clf = RandomForestClassifier(
        n_estimators=500,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(
    clf: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_train_labels: np.ndarray = None,
    X_train: np.ndarray = None,
    y_train: np.ndarray = None
) -> np.ndarray:
    """
    Evaluates the performance of the trained Random Forest classifier on the test split
    (and training split if provided), printing the partition accuracies, generalization gap,
    labeled confusion matrix, and a comprehensive classification report.

    Parameters:
        clf (RandomForestClassifier): A fully trained Random Forest model.
        X_test (np.ndarray): The extended testing feature matrix of shape (num_test, num_features).
        y_test (np.ndarray): True test label array.
        y_train_labels (np.ndarray, optional): Training labels (used to enforce consistent class sorting).
        X_train (np.ndarray, optional): The extended training feature matrix of shape (num_train, num_features).
        y_train (np.ndarray, optional): True training label array.

    Returns:
        np.ndarray: Predicted label array of shape (num_test,).
    """
    y_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    log("\n=================== EVALUATION RESULTS ===================", level=1)
    if X_train is not None and y_train is not None:
        y_train_pred = clf.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        gen_gap = (train_accuracy - test_accuracy) * 100.0

        log(f"Training Set Accuracy:           {train_accuracy * 100.0:.2f}%", level=1)
        log(f"Test Set Accuracy (Generalization): {test_accuracy * 100.0:.2f}%", level=1)
        log(f"Generalization Gap (Train - Test): {gen_gap:.2f}%", level=1)
    else:
        log(f"Overall Classification Accuracy: {test_accuracy * 100.0:.2f}%", level=1)

    log(f"Expected Accuracy from Paper:    ~98.49%", level=1)
    log("==========================================================", level=1)

    # Labeled Confusion Matrix
    if y_train_labels is None:
        if y_train is not None:
            unique_labels = sorted(list(np.unique(y_train)))
        else:
            unique_labels = sorted(list(clf.classes_))
    else:
        unique_labels = sorted(list(np.unique(y_train_labels)))

    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    print_formatted_confusion_matrix(cm, unique_labels)

    # Classification Report
    log("\n--- TEST SET CLASSIFICATION REPORT ---", level=1)
    log(classification_report(y_test, y_pred, labels=unique_labels), level=1)

    return y_pred


def print_formatted_confusion_matrix(
    cm: np.ndarray,
    labels: List[str]
) -> None:
    """
    Outputs a beautifully aligned, ASCII-based text representation of the
    confusion matrix to the terminal.

    Pedagogical Context:
        Standard SciKit-Learn confusion matrix arrays are raw 2D integers, which
        are hard to interpret
        without labels. This helper function dynamically measures string widths
        and outputs a clean grid
        with vertical separators and aligned columns, matching the visual
        excellence expected in a premium
        CLI application.

    Parameters:
        cm (np.ndarray): Raw 2D confusion matrix of shape (num_classes,
        num_classes).
        labels (List[str]): Alphabetically ordered list of the 6 fault class
        names.
    """
    num_classes = len(labels)
    # Determine column width based on longest label length
    max_label_len = max(len(lbl) for lbl in labels)
    col_width = max(max_label_len, 8)

    # 1. Print Header Row (Predicted Classes)
    log("\n--- CONFUSION MATRIX (True \\ Predicted) ---", level=1)
    header = f"{'True Class':<{max_label_len}} |"
    for lbl in labels:
        header += f" {lbl:^{col_width}} |"
    log(header, level=1)
    log("-" * len(header), level=1)

    # 2. Print Rows
    for i in range(num_classes):
        row_str = f"{labels[i]:<{max_label_len}} |"
        for j in range(num_classes):
            cell_val = cm[i, j]
            row_str += f" {cell_val:^{col_width}} |"
        log(row_str, level=1)
    log("-" * len(header), level=1)
