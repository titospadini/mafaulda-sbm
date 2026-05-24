"""
Random Forest Classification and Evaluation on Extended SBM Features

This script implements Step 4 of the Rotating-Machine Fault Diagnosis pipeline:
1. Loads the 92-dimensional extended training and testing feature matrices
   along with their respective labels from the data/ directory.
2. Initializes a Random Forest Classifier with random_state=42 for reproducibility.
3. Trains the classifier on the extended training features.
4. Predicts classes for the extended test features.
5. Computes and displays the overall classification accuracy.
6. Computes and displays a beautifully formatted, labeled confusion matrix
   representing classification outcomes (directly comparable to the paper's Table 6b).
7. Prints a detailed classification report featuring precision, recall, and F1-score.
"""

import os
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def print_formatted_confusion_matrix(cm: np.ndarray, labels: list):
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


if __name__ == '__main__':
    print("=== MaFaulDa Step 4: Random Forest Classification ===")
    start_time = time.time()

    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')

    X_train_path = os.path.join(data_dir, 'X_train_extended.npy')
    X_test_path = os.path.join(data_dir, 'X_test_extended.npy')
    y_train_path = os.path.join(data_dir, 'y_train.npy')
    y_test_path = os.path.join(data_dir, 'y_test.npy')

    # 1. Load extended feature matrices and labels
    print(f"Loading extended data from {data_dir}...")
    X_train_ext = np.load(X_train_path)
    X_test_ext = np.load(X_test_path)
    y_train = np.load(y_train_path, allow_pickle=True)
    y_test = np.load(y_test_path, allow_pickle=True)

    print(f"  Training set shape: {X_train_ext.shape}")
    print(f"  Testing set shape:  {X_test_ext.shape}")

    # 2. Initialize Random Forest Classifier
    # Explicitly configure robust hyper-parameters (n_estimators=500, max_features='sqrt', class_weight='balanced')
    # Use random_state=42 for scientific reproducibility
    print("\nInitializing Random Forest Classifier (n_estimators=500, max_features='sqrt', class_weight='balanced', random_state=42)...")
    clf = RandomForestClassifier(n_estimators=500, max_features='sqrt', class_weight='balanced', random_state=42)

    # 3. Train Classifier
    print("Training classifier on extended features...")
    train_start = time.time()
    clf.fit(X_train_ext, y_train)
    train_elapsed = time.time() - train_start
    print(f"Training completed in {train_elapsed:.2f} seconds.")

    # 4. Predict on Test Set
    print("Predicting classes for testing set...")
    pred_start = time.time()
    y_pred = clf.predict(X_test_ext)
    pred_elapsed = time.time() - pred_start
    print(f"Prediction completed in {pred_elapsed:.4f} seconds.")

    # 5. Evaluate Performance
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_percentage = accuracy * 100.0
    print("\n================ EVALUATION RESULTS ================")
    print(f"Overall Classification Accuracy: {accuracy_percentage:.2f}%")
    print(f"Expected Accuracy from Paper:   ~98.49%")
    print("====================================================")

    # 6. Labeled Confusion Matrix
    # We sort the labels alphabetically to ensure deterministic order matching y_train classes
    unique_labels = sorted(list(np.unique(y_train)))
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    print_formatted_confusion_matrix(cm, unique_labels)

    # 7. Classification Report
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, labels=unique_labels))

    total_elapsed = time.time() - start_time
    print(f"\nEvaluation pipeline completed in {total_elapsed:.2f} seconds!")
    print("====================================================")
