"""
Random Forest Classification and Evaluation on Extended SBM Features

This module handles the classification step (Step 4) of the rotating-machine
fault diagnosis pipeline, exposing function interfaces for training and
evaluation.
"""

from typing import List
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_classifier(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
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


def evaluate_classifier(clf: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray, y_train_labels: np.ndarray) -> np.ndarray:
    """
    Evaluates the performance of the trained Random Forest classifier on the
    test split,
    printing the overall accuracy, labeled confusion matrix, and a comprehensive
    classification report.

    Pedagogical Context:
        To verify model fidelity and replication accuracy, we calculate:
          - Overall Classification Accuracy: The fraction of correctly diagnosed
            operating scenarios.
          - Labeled Confusion Matrix: A tabular breakdown of true vs predicted
            classes, critical for
            inspecting exactly which machine fault states are confused (e.g.
            horizontal misalignment vs vertical).
          - Precision, Recall, and F1-Score: Metric bounds computed per class,
            ensuring that the model's
            performance is evaluated with equal rigor across both abundant and
            highly scarce (Normal) classes.

    Parameters:
        clf (RandomForestClassifier): A fully trained Random Forest model.
        X_test (np.ndarray): The extended testing feature matrix of shape
        (num_test, num_features).
        y_test (np.ndarray): True test label array.
        y_train_labels (np.ndarray): Training labels (used to enforce
        consistent, alphabetized class sorting).

    Returns:
        np.ndarray: Predicted label array of shape (num_test,).
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


def print_formatted_confusion_matrix(cm: np.ndarray, labels: List[str]) -> None:
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
