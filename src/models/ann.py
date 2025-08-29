import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    precision_score, recall_score, f1_score,
    confusion_matrix
)
import time
import warnings
import matplotlib.pyplot as plt
from pathlib import Path

def create_ann_model(input_dim: int, class_weight: bool = True) -> keras.Model:
    """
    Create ANN model architecture
    
    Args:
        input_dim: Number of input features
        class_weight: Whether to use class weights
    
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = layers.Input(shape=(input_dim,))
    
    # Hidden layers
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    optimizer = optimizers.Adam(learning_rate=0.001)
    loss = 'binary_crossentropy'
    
    if class_weight:
        # Use weighted loss for imbalanced classes
        loss = 'weighted_binary_crossentropy'
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy', 'AUC']
    )
    
    return model

def create_callbacks(patience_es: int = 10, patience_rlr: int = 5) -> List[callbacks.Callback]:
    """
    Create training callbacks
    
    Args:
        patience_es: Patience for early stopping
        patience_rlr: Patience for learning rate reduction
    
    Returns:
        List of callbacks
    """
    callbacks_list = [
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience_es,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate reduction
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_rlr,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Model checkpoint
        callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks_list

def train_ann_model(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    config: Dict, class_weights: Optional[Dict] = None) -> Dict:
    """
    Train ANN model
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        config: Configuration dictionary
        class_weights: Class weights dictionary
    
    Returns:
        Dictionary with model, metrics, and training info
    """
    print("Training ANN model...")
    
    # Get training configuration
    batch_size = config.get('train', {}).get('batch', 64)
    epochs = config.get('train', {}).get('epochs', 200)
    lr = config.get('train', {}).get('lr', 0.001)
    patience_es = config.get('train', {}).get('patience_es', 10)
    patience_rlr = config.get('train', {}).get('patience_rlr', 5)
    use_class_weight = config.get('train', {}).get('class_weight', True)
    
    start_time = time.time()
    
    # Create model
    model = create_ann_model(X_train.shape[1], class_weight=use_class_weight)
    
    # Print model summary
    model.summary()
    
    # Create callbacks
    callbacks_list = create_callbacks(patience_es, patience_rlr)
    
    # Prepare class weights if needed
    sample_weights = None
    if use_class_weight and class_weights is not None:
        # Map class weights to sample weights
        sample_weights = np.array([class_weights[int(y)] for y in y_train])
        print(f"Using class weights: {class_weights}")
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks_list,
        sample_weight=sample_weights,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Make predictions
    start_infer = time.time()
    y_train_pred = model.predict(X_train, verbose=0).flatten()
    y_val_pred = model.predict(X_val, verbose=0).flatten()
    inference_time = time.time() - start_infer
    
    # Calculate metrics
    metrics = calculate_classification_metrics(y_val, y_val_pred, y_train, y_train_pred)
    
    # Add timing information
    metrics['train_time_s'] = training_time
    metrics['infer_time_s'] = inference_time
    
    # Add model info
    metrics['model_type'] = 'ann'
    metrics['n_features'] = X_train.shape[1]
    metrics['n_train_samples'] = len(X_train)
    metrics['n_val_samples'] = len(X_val)
    metrics['training_epochs'] = len(history.history['loss'])
    
    # Add training history
    metrics['training_history'] = {
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'auc': history.history['auc'],
        'val_auc': history.history['val_auc']
    }
    
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Inference time: {inference_time:.2f}s")
    print(f"  Training epochs: {len(history.history['loss'])}")
    print(f"  Validation PR-AUC: {metrics['pr_auc']:.3f}")
    print(f"  Validation ROC-AUC: {metrics['roc_auc']:.3f}")
    
    return {
        'model': model,
        'metrics': metrics,
        'predictions': {
            'train': y_train_pred,
            'val': y_val_pred
        },
        'history': history,
        'config': config
    }

def evaluate_ann_model(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Evaluate ANN model on test set
    
    Args:
        model: Trained ANN model
        X_test: Test features
        y_test: Test targets
    
    Returns:
        Dictionary with test metrics and predictions
    """
    print("Evaluating ANN model on test set...")
    
    start_time = time.time()
    
    # Make predictions
    y_test_pred = model.predict(X_test, verbose=0).flatten()
    
    inference_time = time.time() - start_time
    
    # Calculate metrics
    test_metrics = calculate_classification_metrics(y_test, y_test_pred)
    
    # Add timing information
    test_metrics['test_infer_time_s'] = inference_time
    
    print(f"  Test inference time: {inference_time:.2f}s")
    print(f"  Test PR-AUC: {test_metrics['pr_auc']:.3f}")
    print(f"  Test ROC-AUC: {test_metrics['roc_auc']:.3f}")
    
    return {
        'metrics': test_metrics,
        'predictions': y_test_pred
    }

def save_ann_model(model: keras.Model, model_path: str) -> None:
    """
    Save ANN model
    
    Args:
        model: Trained model
        model_path: Path to save model
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.save(model_path)
    print(f"Model saved to {model_path}")

def load_ann_model(model_path: str) -> keras.Model:
    """
    Load ANN model
    
    Args:
        model_path: Path to saved model
    
    Returns:
        Loaded model
    """
    model = keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model

def plot_training_history(history: keras.callbacks.History, save_path: Optional[str] = None) -> None:
    """
    Plot training history
    
    Args:
        history: Training history
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # AUC
    axes[1, 0].plot(history.history['auc'], label='Training AUC')
    axes[1, 0].plot(history.history['val_auc'], label='Validation AUC')
    axes[1, 0].set_title('Model AUC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate (if available)
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

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
