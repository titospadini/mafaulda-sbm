"""
MaFaulDa SBM Hyperparameter Tuning & Cross-Validation Module

This module handles the hyperparameter tuning and model validation step (Step 5)
of the rotating-machine fault diagnosis pipeline, exposing function interfaces to
run multiple validation strategies over SBM parameters (gamma and tau) and Random
Forest ensembles with explicit separation and reporting across Training,
Validation, and Test partitions.

Supported Validation Strategies:
  1. 'stratified' (Default / Paper Reproduction): StratifiedKFold (10-fold by default).
     Preserves relative class proportions across folds as published in Marins et al. (2018).
  2. 'stratified_group': StratifiedGroupKFold.
     Groups samples by physical operating condition (rotational speed regime f_r) to
     eliminate condition leakage between train and validation folds while preserving
     class balance.
  3. 'kfold': Standard KFold without class stratification.
  4. 'repeated_stratified': RepeatedStratifiedKFold.
     Repeats stratified K-fold multiple times to estimate performance variance.
  5. 'nested': Nested Cross-Validation.
     Executes an outer CV loop for unbiased generalization estimation and an inner CV
     loop for data-leakage-free hyperparameter selection.
"""

import os
import time
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    StratifiedGroupKFold,
    RepeatedStratifiedKFold,
)
from sklearn.metrics import accuracy_score

from mafaulda.sbm_model import (
    construct_class_dictionary,
    generate_extended_features,
)

from mafaulda.rf_classifier import (
    train_classifier,
    evaluate_classifier,
)
from mafaulda.logging_utils import log


VALIDATION_METHODS = {
    'stratified': 'Stratified K-Fold (Paper Reproduction Baseline)',
    'stratified_group': 'Stratified Group K-Fold (Operating Condition Isolation)',
    'kfold': 'Standard K-Fold',
    'repeated_stratified': 'Repeated Stratified K-Fold',
    'nested': 'Nested Cross-Validation (Outer Generalization + Inner Tuning)',
}


def get_cv_splitter(
    cv_method: str = 'stratified',
    n_splits: int = 10,
    n_repeats: int = 5,
    random_state: int = 42
) -> Union[StratifiedKFold, StratifiedGroupKFold, KFold, RepeatedStratifiedKFold]:
    """
    Factory function to instantiate the requested scikit-learn cross-validation splitter.

    Parameters:
        cv_method (str): Name of the validation method ('stratified', 'stratified_group',
                         'kfold', 'repeated_stratified').
        n_splits (int): Number of folds / splits (default: 10).
        n_repeats (int): Number of repetitions for RepeatedStratifiedKFold (default: 5).
        random_state (int): Random seed for reproducibility (default: 42).

    Returns:
        BaseCrossValidator: An initialized scikit-learn cross-validation splitter object.

    Raises:
        ValueError: If cv_method is unrecognized.
    """
    method = cv_method.lower().strip()
    if method == 'stratified':
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif method == 'stratified_group':
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif method == 'kfold':
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif method == 'repeated_stratified':
        return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    else:
        raise ValueError(
            f"Unsupported cv_method '{cv_method}'. Supported choices: {list(VALIDATION_METHODS.keys())}"
        )


def run_tuning(
    data_dir: str,
    cv_method: str = 'stratified',
    n_splits: int = 10,
    n_repeats: int = 5,
    groups: Optional[np.ndarray] = None,
    gammas: Optional[List[float]] = None,
    taus: Optional[List[float]] = None,
    use_gpu: bool = False,
    random_state: int = 42
) -> None:
    """
    Executes hyperparameter tuning and cross-validation to evaluate SBM hyperparameters
    (WSF sensitivity gamma and threshold tau) on pre-extracted training features, with
    explicit reporting across Training, Validation, and Test partitions.

    Pedagogical Context:
        - Separate Partition Reporting:
          * Training: Evaluates fitting capacity and over-memorization on training folds.
          * Validation: Guides selection of gamma and tau across CV folds without touching test data.
          * Test: Evaluates the chosen optimal model on the untouched holdout test set to report
            true out-of-sample generalization.
        - Strict Manifold Modeling per Fold: In each fold iteration, SBM dictionaries (D_c)
          are constructed from scratch using only that fold's training portion, guaranteeing
          zero data leakage.

    Parameters:
        data_dir (str): Absolute path to the directory containing pre-extracted feature files.
        cv_method (str): Validation method to use ('stratified', 'stratified_group', 'kfold',
                         'repeated_stratified', 'nested').
        n_splits (int): Number of folds (default: 10).
        n_repeats (int): Number of repetitions for repeated CV (default: 5).
        groups (np.ndarray, optional): Group assignments per sample for group-aware CV.
                                      If None and cv_method is 'stratified_group', groups are
                                      auto-derived from rotation frequency bins (Feature 0).
        gammas (List[float], optional): Custom list of WSF sensitivity gamma parameters.
        taus (List[float], optional): Custom list of SBM dictionary thresholds.
        use_gpu (bool): Whether to enable GPU acceleration for SBM projections.
        random_state (int): Random seed for reproducibility (default: 42).

    Raises:
        FileNotFoundError: If pre-extracted feature files do not exist under `data_dir`.
        ValueError: If cv_method is invalid.
    """
    method = cv_method.lower().strip()
    method_title = VALIDATION_METHODS.get(method, method)

    log("\n" + "="*60, level=1)
    log(f"=== STEP 5: SBM Hyperparameter Tuning & Partition Evaluation ===", level=1)
    log(f"Validation Strategy: {method_title}", level=1)
    log(f"Number of Folds:     {n_splits}" + (f" (Repeats: {n_repeats})" if method == 'repeated_stratified' else ""), level=1)
    log("="*60, level=1)

    X_train_path = os.path.join(data_dir, 'X_train_features.npy')
    y_train_path = os.path.join(data_dir, 'y_train.npy')
    X_test_path = os.path.join(data_dir, 'X_test_features.npy')
    y_test_path = os.path.join(data_dir, 'y_test.npy')

    if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
        raise FileNotFoundError("Training features not found! Please run the pipeline first to generate them.")

    log(f"Loading training features from {data_dir}...", level=2)
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path, allow_pickle=True)

    log(f"  Training features shape: {X_train.shape}", level=3)
    log(f"  Training labels shape:   {y_train.shape}", level=3)

    # Optional test set loading
    has_test_set = os.path.exists(X_test_path) and os.path.exists(y_test_path)
    X_test, y_test = None, None
    if has_test_set:
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path, allow_pickle=True)
        log(f"  Holdout test features shape: {X_test.shape} (isolated from tuning)", level=3)

    # Determine groups if using stratified_group
    groups_train = None
    if method == 'stratified_group':
        if groups is not None:
            groups_train = np.array(groups)
        else:
            groups_path = os.path.join(data_dir, 'groups_train.npy')
            if os.path.exists(groups_path):
                log(f"Loading pre-saved group assignments from {groups_path}...", level=2)
                groups_train = np.load(groups_path)
            else:
                # Group by rounded physical rotation frequency f_r (Feature 0)
                log("Auto-grouping samples by operational rotation frequency (Feature 0: f_r)...", level=2)
                groups_train = np.round(X_train[:, 0]).astype(int)

        n_unique_groups = len(np.unique(groups_train))
        log(f"  Total operational condition groups identified: {n_unique_groups}", level=2)

    # Grid search parameter space
    if gammas is None:
        gammas = [0.0005, 0.001, 0.01, 0.1]
    if taus is None:
        taus = [0.75, 0.80, 0.85, 0.90]

    log(f"\nEvaluating hyperparameter grid search over gammas (γ): {gammas} | taus (τ): {taus}", level=1)

    # Nested Cross-Validation Path
    if method == 'nested':
        run_nested_cv(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            gammas=gammas,
            taus=taus,
            n_outer_splits=n_splits,
            use_gpu=use_gpu,
            random_state=random_state
        )
        return

    # Standard / Group / Repeated Grid Search Path
    splitter = get_cv_splitter(
        cv_method=method,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )

    results = []
    grid_start_time = time.time()

    for gamma in gammas:
        for tau in taus:
            comb_start = time.time()
            log(f"\nEvaluating combination: gamma={gamma}, tau={tau}...", level=2)
            fold_train_accs = []
            fold_val_accs = []

            # Generate splits (with groups if supported)
            if method == 'stratified_group':
                split_iter = splitter.split(X_train, y_train, groups=groups_train)
            elif method == 'kfold':
                split_iter = splitter.split(X_train)
            else:
                split_iter = splitter.split(X_train, y_train)

            for fold, (train_idx, val_idx) in enumerate(split_iter):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                # Dictionary matrix construction per fold (zero leakage)
                D_c_dict = {}
                unique_classes = np.unique(y_tr)
                for cls in unique_classes:
                    X_c = X_tr[y_tr == cls]
                    D_c = construct_class_dictionary(X_c, tau=tau, gamma=gamma)
                    D_c_dict[cls] = D_c

                # SBM estimation and error vector generation
                X_tr_ext = generate_extended_features(X_tr, D_c_dict, gamma=gamma, use_gpu=use_gpu)
                X_val_ext = generate_extended_features(X_val, D_c_dict, gamma=gamma, use_gpu=use_gpu)

                # Train Random Forest Classifier
                clf = train_classifier(X_tr_ext, y_tr)

                # Evaluate both Training and Validation folds
                tr_pred = clf.predict(X_tr_ext)
                val_pred = clf.predict(X_val_ext)

                acc_tr = accuracy_score(y_tr, tr_pred)
                acc_val = accuracy_score(y_val, val_pred)

                fold_train_accs.append(acc_tr)
                fold_val_accs.append(acc_val)

            mean_tr_acc = np.mean(fold_train_accs)
            mean_val_acc = np.mean(fold_val_accs)
            std_val_acc = np.std(fold_val_accs)
            overfit_gap = (mean_tr_acc - mean_val_acc) * 100.0

            results.append((gamma, tau, mean_tr_acc, mean_val_acc, std_val_acc, overfit_gap))
            log(
                f"  -> Train: {mean_tr_acc * 100.0:.2f}% | "
                f"Val ({n_splits}-Fold): {mean_val_acc * 100.0:.2f}% (±{std_val_acc * 100.0:.2f}%) | "
                f"Gap: {overfit_gap:.2f}% [computed in {time.time() - comb_start:.2f}s]",
                level=2
            )

    grid_elapsed = time.time() - grid_start_time
    log(f"\nGrid search completed in {grid_elapsed:.2f} seconds.", level=2)

    # Sort results by mean validation accuracy descending
    results.sort(key=lambda x: x[3], reverse=True)
    best_gamma, best_tau, best_tr_acc, best_val_acc, best_val_std, best_gap = results[0]

    # Print formatted comparison table with Train, Validation, and Overfitting Gap
    print_results_table(results, cv_title=f"{n_splits}-Fold ({method})")

    log("\n" + "="*60, level=1)
    log("================== OPTIMAL CONFIGURATION ==================", level=1)
    log(f"Validation Method:            {method_title}", level=1)
    log(f"Best WSF Gamma (γ):           {best_gamma:.4f}", level=1)
    log(f"Best Threshold Tau (τ):       {best_tau:.2f}", level=1)
    log(f"Cross-Validation Train Score: {best_tr_acc * 100.0:.2f}%", level=1)
    log(f"Cross-Validation Val Score:   {best_val_acc * 100.0:.2f}% (±{best_val_std * 100.0:.2f}%)", level=1)
    log(f"CV Overfitting Gap:           {best_gap:.2f}%", level=1)
    log("="*60, level=1)

    # Final Holdout Test Evaluation using the selected optimal configuration
    if has_test_set and X_test is not None and y_test is not None:
        log("\nEvaluating optimal configuration on the full training set & untouched test set...", level=1)
        
        # Build dictionaries on full training set with best (gamma, tau)
        D_c_full = {}
        for cls in np.unique(y_train):
            X_c = X_train[y_train == cls]
            D_c_full[cls] = construct_class_dictionary(X_c, tau=best_tau, gamma=best_gamma)

        X_train_ext = generate_extended_features(X_train, D_c_full, gamma=best_gamma, use_gpu=use_gpu)
        X_test_ext = generate_extended_features(X_test, D_c_full, gamma=best_gamma, use_gpu=use_gpu)

        final_clf = train_classifier(X_train_ext, y_train)
        
        train_final_acc = accuracy_score(y_train, final_clf.predict(X_train_ext))
        test_final_acc = accuracy_score(y_test, final_clf.predict(X_test_ext))
        final_gap = (train_final_acc - test_final_acc) * 100.0

        log("\n" + "="*68, level=1)
        log("================ UNIFIED PARTITION EVALUATION SUMMARY ================", level=1)
        log(f"Training Partition (Fitted on Full Train):      {train_final_acc * 100.0:.2f}%", level=1)
        log(f"Validation Partition (Mean CV Score):          {best_val_acc * 100.0:.2f}% (±{best_val_std * 100.0:.2f}%)", level=1)
        log(f"Test Partition (Untouched Holdout Test):       {test_final_acc * 100.0:.2f}%", level=1)
        log(f"Real-World Generalization Gap (Train - Test):  {final_gap:.2f}%", level=1)
        log("="*68, level=1)

        # Output detailed classification report on test set
        evaluate_classifier(
            clf=final_clf,
            X_test=X_test_ext,
            y_test=y_test,
            y_train_labels=y_train,
            X_train=X_train_ext,
            y_train=y_train
        )


def run_nested_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    gammas: List[float],
    taus: List[float],
    n_outer_splits: int = 5,
    use_gpu: bool = False,
    random_state: int = 42
) -> None:
    """
    Executes Nested Cross-Validation (Outer loop for unbiased generalization,
    Inner loop for SBM hyperparameter tuning).
    """
    log(f"\nSetting up Nested Cross-Validation ({n_outer_splits} Outer Folds)...", level=2)
    outer_cv = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=random_state)
    inner_n_splits = max(2, min(5, n_outer_splits))

    outer_scores = []
    outer_train_scores = []
    selected_configs = []
    nested_start_time = time.time()

    for outer_fold, (outer_tr_idx, outer_val_idx) in enumerate(outer_cv.split(X_train, y_train)):
        outer_fold_start = time.time()
        log(f"\n--- Outer Fold {outer_fold + 1}/{n_outer_splits} ---", level=2)

        X_out_tr, X_out_val = X_train[outer_tr_idx], X_train[outer_val_idx]
        y_out_tr, y_out_val = y_train[outer_tr_idx], y_train[outer_val_idx]

        # Inner CV on X_out_tr to select best (gamma, tau)
        inner_cv = StratifiedKFold(n_splits=inner_n_splits, shuffle=True, random_state=random_state)
        best_inner_acc = -1.0
        best_inner_cfg = (gammas[0], taus[0])

        for gamma in gammas:
            for tau in taus:
                inner_fold_accs = []
                for in_tr_idx, in_val_idx in inner_cv.split(X_out_tr, y_out_tr):
                    X_in_tr, X_in_val = X_out_tr[in_tr_idx], X_out_tr[in_val_idx]
                    y_in_tr, y_in_val = y_out_tr[in_tr_idx], y_out_tr[in_val_idx]

                    D_c_dict = {}
                    for cls in np.unique(y_in_tr):
                        X_c = X_in_tr[y_in_tr == cls]
                        D_c = construct_class_dictionary(X_c, tau=tau, gamma=gamma)
                        D_c_dict[cls] = D_c

                    X_in_tr_ext = generate_extended_features(X_in_tr, D_c_dict, gamma=gamma, use_gpu=use_gpu)
                    X_in_val_ext = generate_extended_features(X_in_val, D_c_dict, gamma=gamma, use_gpu=use_gpu)

                    clf = train_classifier(X_in_tr_ext, y_in_tr)
                    in_acc = accuracy_score(y_in_val, clf.predict(X_in_val_ext))
                    inner_fold_accs.append(in_acc)

                mean_in_acc = np.mean(inner_fold_accs)
                if mean_in_acc > best_inner_acc:
                    best_inner_acc = mean_in_acc
                    best_inner_cfg = (gamma, tau)

        best_gamma, best_tau = best_inner_cfg
        log(f"  Outer Fold {outer_fold + 1} Selected Best Inner Config: gamma={best_gamma}, tau={best_tau} (Inner Val Acc: {best_inner_acc * 100.0:.2f}%)", level=2)

        # Train on entire outer training fold with optimal hyperparameters
        D_c_dict = {}
        for cls in np.unique(y_out_tr):
            X_c = X_out_tr[y_out_tr == cls]
            D_c = construct_class_dictionary(X_c, tau=best_tau, gamma=best_gamma)
            D_c_dict[cls] = D_c

        X_out_tr_ext = generate_extended_features(X_out_tr, D_c_dict, gamma=best_gamma, use_gpu=use_gpu)
        X_out_val_ext = generate_extended_features(X_out_val, D_c_dict, gamma=best_gamma, use_gpu=use_gpu)

        clf = train_classifier(X_out_tr_ext, y_out_tr)
        out_tr_acc = accuracy_score(y_out_tr, clf.predict(X_out_tr_ext))
        outer_acc = accuracy_score(y_out_val, clf.predict(X_out_val_ext))
        
        outer_train_scores.append(out_tr_acc)
        outer_scores.append(outer_acc)
        selected_configs.append((best_gamma, best_tau))

        log(f"  Outer Fold {outer_fold + 1} Train Acc: {out_tr_acc * 100.0:.2f}% | Val Acc: {outer_acc * 100.0:.2f}% (in {time.time() - outer_fold_start:.2f}s)", level=2)

    total_nested_time = time.time() - nested_start_time
    mean_outer_tr_acc = np.mean(outer_train_scores)
    mean_outer_acc = np.mean(outer_scores)
    std_outer_acc = np.std(outer_scores)
    nested_gap = (mean_outer_tr_acc - mean_outer_acc) * 100.0

    log("\n" + "="*72, level=1)
    log("==================== NESTED CV GENERALIZATION REPORT ===================", level=1)
    log(f"{'Outer Fold':^12} | {'Selected γ':^12} | {'Selected τ':^12} | {'Train Acc':^12} | {'Val Acc':^12}", level=1)
    log("-" * 72, level=1)
    for fold_idx, (tr_score, val_score, (g, t)) in enumerate(zip(outer_train_scores, outer_scores, selected_configs)):
        tr_str = f"{tr_score * 100.0:.2f}%"
        val_str = f"{val_score * 100.0:.2f}%"
        log(f"{fold_idx + 1:^12} | {g:^12.4f} | {t:^12.2f} | {tr_str:^12} | {val_str:^12}", level=1)
    log("="*72, level=1)
    log(f"Mean Nested Training Accuracy:   {mean_outer_tr_acc * 100.0:.2f}%", level=1)
    log(f"Unbiased Nested CV Val Accuracy: {mean_outer_acc * 100.0:.2f}% (±{std_outer_acc * 100.0:.2f}%)", level=1)
    log(f"Nested Generalization Gap:       {nested_gap:.2f}%", level=1)
    log(f"Total Nested CV Time:            {total_nested_time:.2f}s", level=1)
    log("="*72, level=1)


def print_results_table(
    results: List[Tuple[float, float, float, float, float, float]],
    cv_title: str = "10-Fold CV"
) -> None:
    """
    Outputs an aligned ASCII text table displaying all evaluated hyperparameter
    combinations with Train Accuracy, Validation Accuracy (±std), and Overfitting Gap.

    Parameters:
        results: List of tuples (gamma, tau, mean_tr_acc, mean_val_acc, std_val_acc, overfit_gap).
        cv_title: Label describing the cross-validation strategy.
    """
    log("\n============================= GRID SEARCH RESULTS =============================", level=1)
    val_header = f"Val Accuracy ({cv_title})"
    log(f"{'WSF Gamma (γ)':^13} | {'Threshold Tau (τ)':^17} | {'Train Acc':^11} | {val_header:^25} | {'Gap (Tr-Val)':^12}", level=1)
    log("-" * 88, level=1)
    for gamma, tau, mean_tr_acc, mean_val_acc, std_val_acc, overfit_gap in results:
        tr_str = f"{mean_tr_acc * 100.0:.2f}%"
        val_str = f"{mean_val_acc * 100.0:.2f}% (±{std_val_acc * 100.0:.2f}%)"
        gap_str = f"{overfit_gap:.2f}%"
        log(f"{gamma:^13.4f} | {tau:^17.2f} | {tr_str:^11} | {val_str:^25} | {gap_str:^12}", level=1)
    log("================================================================================", level=1)
