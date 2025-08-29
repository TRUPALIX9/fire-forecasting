"""
LSTM model for wildfire prediction.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.metrics import (
    precision_recall_curve, 
    roc_curve, 
    auc, 
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from ..utils import timer, ensure_dir, save_json, save_pickle
from ..config import get_config_section

logger = logging.getLogger(__name__)


def create_lstm_model(config: Dict[str, Any]) -> keras.Model:
    """
    Create LSTM model architecture.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Compiled LSTM model
    """
    train_config = get_config_section(config, 'train')
    
    # Model parameters
    lookback_days = train_config.get('lookback_days', 14)
    n_features = train_config.get('n_features', None)  # Will be set during training
    lstm_units = train_config.get('lstm_units', [128, 64])
    dropout_rate = train_config.get('dropout_rate', 0.3)
    learning_rate = train_config.get('lr', 0.001)
    
    # Build model
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(lookback_days, n_features)),
        
        # LSTM layers
        layers.LSTM(lstm_units[0], return_sequences=True, dropout=dropout_rate),
        layers.BatchNormalization(),
        
        layers.LSTM(lstm_units[1], dropout=dropout_rate),
        layers.BatchNormalization(),
        
        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(dropout_rate),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    logger.info(f"Created LSTM model with {model.count_params():,} parameters")
    logger.info(f"Architecture: Input({lookback_days}, {n_features}) -> "
               f"LSTM({lstm_units[0]}) -> LSTM({lstm_units[1]}) -> Dense(64) -> Dense(32) -> Dense(1)")
    
    return model


def create_lstm_callbacks(config: Dict[str, Any], model_dir: Path) -> list:
    """
    Create callbacks for LSTM training.
    
    Args:
        config: Configuration dictionary
        model_dir: Directory to save model checkpoints
        
    Returns:
        List of callbacks
    """
    train_config = get_config_section(config, 'train')
    
    callbacks_list = [
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=train_config.get('patience_es', 15),
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=train_config.get('patience_rlr', 8),
            min_lr=1e-7,
            verbose=1
        ),
        
        # Model checkpoint
        callbacks.ModelCheckpoint(
            filepath=model_dir / 'best_lstm_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # TensorBoard logging
        callbacks.TensorBoard(
            log_dir=model_dir / 'logs',
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
    ]
    
    return callbacks_list


def prepare_lstm_data(
    X: np.ndarray, 
    y: np.ndarray, 
    lookback_days: int = 14
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for LSTM model with lookback window.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        lookback_days: Number of days to look back
        
    Returns:
        Tuple of (X_lstm, y_lstm) where X_lstm has shape (n_samples - lookback_days, lookback_days, n_features)
    """
    n_samples, n_features = X.shape
    
    if n_samples <= lookback_days:
        raise ValueError(f"Not enough samples ({n_samples}) for lookback window of {lookback_days}")
    
    # Create sequences
    X_lstm = []
    y_lstm = []
    
    for i in range(lookback_days, n_samples):
        X_lstm.append(X[i-lookback_days:i])
        y_lstm.append(y[i])
    
    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)
    
    logger.info(f"Prepared LSTM data: {X_lstm.shape} -> {y_lstm.shape}")
    
    return X_lstm, y_lstm


def train_lstm_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any]
) -> keras.callbacks.History:
    """
    Train LSTM model.
    
    Args:
        model: LSTM model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        config: Configuration dictionary
        
    Returns:
        Training history
    """
    train_config = get_config_section(config, 'train')
    
    # Training parameters
    batch_size = train_config.get('batch', 32)
    epochs = train_config.get('epochs', 200)
    class_weight = train_config.get('class_weight', True)
    
    # Calculate class weights if requested
    class_weights = None
    if class_weight:
        n_pos = np.sum(y_train)
        n_neg = len(y_train) - n_pos
        class_weights = {
            0: n_pos / len(y_train),
            1: n_neg / len(y_train)
        }
        logger.info(f"Using class weights: {class_weights}")
    
    # Create callbacks
    model_dir = Path("artifacts/models")
    ensure_dir(model_dir)
    callbacks_list = create_lstm_callbacks(config, model_dir)
    
    # Train model
    logger.info("Starting LSTM training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=1
    )
    
    logger.info("LSTM training completed")
    
    return history


def evaluate_lstm_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate LSTM model performance.
    
    Args:
        model: Trained LSTM model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating LSTM model...")
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred_proba = y_pred_proba.flatten()
    
    # Calculate optimal threshold using PR curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    # Make predictions at optimal threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics
    pr_auc = auc(recall, precision)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    precision_score_val = precision_score(y_test, y_pred, zero_division=0)
    recall_score_val = recall_score(y_test, y_pred, zero_division=0)
    f1_score_val = f1_score(y_test, y_pred, zero_division=0)
    
    # Compile results
    metrics = {
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'precision': float(precision_score_val),
        'recall': float(recall_score_val),
        'f1_score': float(f1_score_val),
        'threshold': float(optimal_threshold),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'accuracy': float((tp + tn) / (tp + tn + fp + fn))
    }
    
    logger.info(f"LSTM Evaluation Results:")
    logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"  Optimal Threshold: {metrics['threshold']:.3f}")
    
    return metrics


def save_lstm_model(
    model: keras.Model,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    history: keras.callbacks.History
) -> None:
    """
    Save LSTM model and related artifacts.
    
    Args:
        model: Trained LSTM model
        config: Configuration dictionary
        metrics: Evaluation metrics
        history: Training history
    """
    # Create directories
    models_dir = Path("artifacts/models")
    metrics_dir = Path("artifacts/metrics")
    figures_dir = Path("artifacts/figures")
    
    ensure_dir(models_dir)
    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)
    
    # Save model
    model_path = models_dir / "lstm_model.h5"
    model.save(model_path)
    logger.info(f"LSTM model saved to {model_path}")
    
    # Save metrics
    metrics_path = metrics_dir / "lstm_metrics.json"
    save_json(metrics, metrics_path)
    logger.info(f"LSTM metrics saved to {metrics_path}")
    
    # Save training history
    history_path = metrics_dir / "lstm_history.pkl"
    save_pickle(history.history, history_path)
    logger.info(f"LSTM training history saved to {history_path}")
    
    # Save configuration
    config_path = metrics_dir / "lstm_config.json"
    save_json(config, config_path)
    logger.info(f"LSTM configuration saved to {config_path}")


def load_lstm_model(model_path: Path) -> keras.Model:
    """
    Load saved LSTM model.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded LSTM model
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading LSTM model from {model_path}")
    model = keras.models.load_model(model_path)
    
    return model


def plot_lstm_training_history(history: keras.callbacks.History, save_path: Path) -> None:
    """
    Plot LSTM training history.
    
    Args:
        history: Training history
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"LSTM training history plot saved to {save_path}")


def calculate_lstm_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics for LSTM model.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of classification metrics
    """
    # Make predictions at threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Specificity and sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall  # Same as recall
    
    # FPR and FNR
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    metrics = {
        'threshold': threshold,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'accuracy': float(accuracy),
        'specificity': float(specificity),
        'sensitivity': float(sensitivity),
        'fpr': float(fpr),
        'fnr': float(fnr)
    }
    
    return metrics
