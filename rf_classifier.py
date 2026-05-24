"""
Random Forest Classification and Evaluation on Extended SBM Features

This module handles the classification step (Step 4) of the rotating-machine
fault diagnosis pipeline, exposing function interfaces for training and evaluation.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_classifier(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Initializes and trains the Random Forest classifier with standardized hyperparameters:
    n_estimators=500, max_features='sqrt', class_weight='balanced', random_state=42.

    Parameters:
        X_train (np.ndarray): Extended training feature matrix.
        y_train (np.ndarray): Training labels.

    Returns:
        RandomForestClassifier: Trained model.
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


def evaluate_classifier(clf: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray, y_train_labels: np.ndarray) -> np.ndarray:
    """
    Predicts classes and evaluates the trained classifier by printing performance metrics.

    Parameters:
        clf (RandomForestClassifier): Trained model.
        X_test (np.ndarray): Extended testing feature matrix.
        y_test (np.ndarray): Testing labels.
        y_train_labels (np.ndarray): Full training labels (used to determine class labels order).

    Returns:
        np.ndarray: Predicted labels.
    """
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n================ EVALUATION RESULTS ================")
    print(f"Overall Classification Accuracy: {accuracy * 100.0:.2f}%")
    print(f"Expected Accuracy from Paper:   ~98.49%")
    print("====================================================")

    # Labeled Confusion Matrix
    unique_labels = sorted(list(np.unique(y_train_labels)))
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    print_formatted_confusion_matrix(cm, unique_labels)

    # Classification Report
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, labels=unique_labels))

    return y_pred


def print_formatted_confusion_matrix(cm: np.ndarray, labels: list) -> None:
    """
    Prints a beautifully formatted, aligned text-based confusion matrix
    with column and row labels for professional terminal presentation.

    Parameters:
        cm (np.ndarray): Confusion matrix array of shape (num_classes, num_classes).
        labels (list): Sorted list of unique class names.
    """
    num_classes = len(labels)
    # Determine column width based on longest label length
    max_label_len = max(len(lbl) for lbl in labels)
    col_width = max(max_label_len, 8)

    # 1. Print Header Row (Predicted Classes)
    print("\n--- CONFUSION MATRIX (True \\ Predicted) ---")
    header = f"{'True Class':<{max_label_len}} |"
    for lbl in labels:
        header += f" {lbl:^{col_width}} |"
    print(header)
    print("-" * len(header))

    # 2. Print Rows
    for i in range(num_classes):
        row_str = f"{labels[i]:<{max_label_len}} |"
        for j in range(num_classes):
            cell_val = cm[i, j]
            row_str += f" {cell_val:^{col_width}} |"
        print(row_str)
    print("-" * len(header))
