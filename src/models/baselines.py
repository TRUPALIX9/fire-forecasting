import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import time
import warnings

def train_logistic_baseline(X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           class_weight: bool = True) -> Dict:
    """
    Train logistic regression baseline model
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        class_weight: Whether to use balanced class weights
    
    Returns:
        Dictionary with model, metrics, and training info
    """
    print("Training logistic regression baseline...")
    
    start_time = time.time()
    
    # Create model
    if class_weight:
        model = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000,
            solver='liblinear'
        )
    else:
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='liblinear'
        )
    
    # Train model
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Make predictions
    start_infer = time.time()
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_val_pred = model.predict_proba(X_val)[:, 1]
    inference_time = time.time() - start_infer
    
    # Calculate metrics
    metrics = calculate_classification_metrics(y_val, y_val_pred, y_train, y_train_pred)
    
    # Add timing information
    metrics['train_time_s'] = training_time
    metrics['infer_time_s'] = inference_time
    
    # Add model info
    metrics['model_type'] = 'logistic_regression'
    metrics['n_features'] = X_train.shape[1]
    metrics['n_train_samples'] = len(X_train)
    metrics['n_val_samples'] = len(X_val)
    
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Inference time: {inference_time:.2f}s")
    print(f"  Validation PR-AUC: {metrics['pr_auc']:.3f}")
    print(f"  Validation ROC-AUC: {metrics['roc_auc']:.3f}")
    
    return {
        'model': model,
        'metrics': metrics,
        'predictions': {
            'train': y_train_pred,
            'val': y_val_pred
        }
    }

def train_random_forest_baseline(X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                class_weight: bool = True) -> Dict:
    """
    Train random forest baseline model
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        class_weight: Whether to use balanced class weights
    
    Returns:
        Dictionary with model, metrics, and training info
    """
    print("Training random forest baseline...")
    
    start_time = time.time()
    
    # Create model
    if class_weight:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    
    # Train model
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Make predictions
    start_infer = time.time()
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_val_pred = model.predict_proba(X_val)[:, 1]
    inference_time = time.time() - start_infer
    
    # Calculate metrics
    metrics = calculate_classification_metrics(y_val, y_val_pred, y_train, y_train_pred)
    
    # Add timing information
    metrics['train_time_s'] = training_time
    metrics['infer_time_s'] = inference_time
    
    # Add model info
    metrics['model_type'] = 'random_forest'
    metrics['n_features'] = X_train.shape[1]
    metrics['n_train_samples'] = len(X_train)
    metrics['n_val_samples'] = len(X_val)
    
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Inference time: {inference_time:.2f}s")
    print(f"  Validation PR-AUC: {metrics['pr_auc']:.3f}")
    print(f"  Validation ROC-AUC: {metrics['roc_auc']:.3f}")
    
    return {
        'model': model,
        'metrics': metrics,
        'predictions': {
            'train': y_train_pred,
            'val': y_val_pred
        }
    }

def calculate_classification_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   y_train_true: Optional[np.ndarray] = None,
                                   y_train_pred_proba: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        y_train_true: Training true labels (optional)
        y_train_pred_proba: Training predicted probabilities (optional)
    
    Returns:
        Dictionary with all metrics
    """
    # Find optimal threshold using validation set
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculate F1 score for each threshold
    f1_scores = []
    for i in range(len(precision) - 1):  # Skip last point (precision=1, recall=0)
        if precision[i] + recall[i] > 0:
            f1 = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1 = 0
        f1_scores.append(f1)
    
    # Find threshold that maximizes F1 score
    best_f1_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_f1_idx]
    
    # Calculate metrics at optimal threshold
    y_pred_binary = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Basic metrics
    precision_at_threshold = precision_score(y_true, y_pred_binary, zero_division=0)
    recall_at_threshold = recall_score(y_true, y_pred_binary, zero_division=0)
    f1_at_threshold = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # AUC metrics
    pr_auc = auc(recall, precision)
    
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Training metrics if provided
    train_metrics = {}
    if y_train_true is not None and y_train_pred_proba is not None:
        train_precision, train_recall, _ = precision_recall_curve(y_train_true, y_train_pred_proba)
        train_pr_auc = auc(train_recall, train_precision)
        
        train_fpr, train_tpr, _ = roc_curve(y_train_true, y_train_pred_proba)
        train_roc_auc = auc(train_fpr, train_tpr)
        
        train_metrics = {
            'train_pr_auc': train_pr_auc,
            'train_roc_auc': train_roc_auc
        }
    
    metrics = {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'precision_at_threshold': precision_at_threshold,
        'recall_at_threshold': recall_at_threshold,
        'f1_at_threshold': f1_at_threshold,
        'optimal_threshold': optimal_threshold,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'precision_recall_curve': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist()
        },
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': roc_thresholds.tolist()
        }
    }
    
    # Add training metrics if available
    metrics.update(train_metrics)
    
    return metrics

def evaluate_baseline_models(X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            class_weight: bool = True) -> Dict:
    """
    Evaluate multiple baseline models
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        class_weight: Whether to use balanced class weights
    
    Returns:
        Dictionary with results for all models
    """
    print("Evaluating baseline models...")
    
    results = {}
    
    # Logistic Regression
    print("\n--- Logistic Regression ---")
    lr_results = train_logistic_baseline(X_train, y_train, X_val, y_val, class_weight)
    
    # Test set evaluation
    y_test_pred = lr_results['model'].predict_proba(X_test)[:, 1]
    test_metrics = calculate_classification_metrics(y_test, y_test_pred)
    
    lr_results['test_metrics'] = test_metrics
    lr_results['test_predictions'] = y_test_pred
    
    results['logistic_regression'] = lr_results
    
    # Random Forest
    print("\n--- Random Forest ---")
    rf_results = train_random_forest_baseline(X_train, y_train, X_val, y_val, class_weight)
    
    # Test set evaluation
    y_test_pred = rf_results['model'].predict_proba(X_test)[:, 1]
    test_metrics = calculate_classification_metrics(y_test, y_test_pred)
    
    rf_results['test_metrics'] = test_metrics
    rf_results['test_predictions'] = y_test_pred
    
    results['random_forest'] = rf_results
    
    # Summary comparison
    print("\n--- Baseline Model Comparison ---")
    for model_name, model_results in results.items():
        metrics = model_results['metrics']
        test_metrics = model_results['test_metrics']
        
        print(f"\n{model_name.replace('_', ' ').title()}:")
        print(f"  Validation PR-AUC: {metrics['pr_auc']:.3f}")
        print(f"  Validation ROC-AUC: {metrics['roc_auc']:.3f}")
        print(f"  Test PR-AUC: {test_metrics['pr_auc']:.3f}")
        print(f"  Test ROC-AUC: {test_metrics['roc_auc']:.3f}")
        print(f"  Training time: {metrics['train_time_s']:.2f}s")
    
    return results

def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from trained model
    
    Args:
        model: Trained model
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        # Random Forest, etc.
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Logistic Regression, etc.
        importance = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df
