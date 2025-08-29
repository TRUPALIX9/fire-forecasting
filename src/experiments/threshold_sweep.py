"""
Threshold sweep experiment: precision/recall trade-off analysis.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from config import load_config, set_random_seed
from utils import timer, ensure_dir
from data_sources.raws import fetch_all_sites_raws_data
from data_sources.firms import create_fire_labels_for_sites
from features.engineer import engineer_all_features
from splits.chrono import create_chronological_pipeline
from models.ann import create_ann_model, train_ann_model, evaluate_ann_model


def run_threshold_sweep_experiment(config_path: str):
    """Run threshold sweep experiment to find optimal operating point."""
    
    # Load configuration
    config = load_config(config_path)
    set_random_seed(config['project']['seed'])
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🎯 Starting Threshold Sweep Experiment")
    logger.info(f"Project: {config['project']['area']}")
    
    # Ensure output directories
    output_dir = Path("artifacts/figures")
    ensure_dir(output_dir)
    
    # Fetch and prepare data
    logger.info("📊 Fetching RAWS data...")
    raws_data = fetch_all_sites_raws_data(config)
    
    logger.info("🔥 Fetching FIRMS data...")
    fire_labels = create_fire_labels_for_sites(config, raws_data)
    
    # Merge features and labels
    logger.info("🔗 Merging features and labels...")
    merged_data = pd.merge(raws_data, fire_labels, on=['site', 'date'], how='inner')
    
    # Engineer features
    logger.info("⚙️ Engineering features...")
    feature_data = engineer_all_features(merged_data, config)
    
    # Split data
    logger.info("✂️ Splitting data chronologically...")
    splits = create_chronological_pipeline(feature_data, config)
    
    # Train ANN model
    logger.info("🧠 Training ANN model...")
    model = create_ann_model(config)
    
    with timer() as train_timer:
        history = train_ann_model(
            model, 
            splits['X_train'], 
            splits['y_train'],
            splits['X_val'],
            splits['y_val'],
            config
        )
    
    # Get predictions
    logger.info("🔮 Generating predictions...")
    y_pred_proba = model.predict(splits['X_test'])
    y_pred_proba = y_pred_proba.flatten()
    
    # Calculate curves
    logger.info("📈 Calculating precision-recall and ROC curves...")
    precision, recall, pr_thresholds = precision_recall_curve(splits['y_test'], y_pred_proba)
    fpr, tpr, roc_thresholds = roc_curve(splits['y_test'], y_pred_proba)
    
    # Threshold sweep
    logger.info("🎯 Performing threshold sweep...")
    thresholds = np.linspace(0.01, 0.99, 99)
    results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(splits['y_test'], y_pred).ravel()
        
        # Calculate metrics
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        results.append({
            'threshold': threshold,
            'precision': precision_val,
            'recall': recall_val,
            'f1_score': f1_val,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })
    
    results_df = pd.DataFrame(results)
    
    # Save results
    results_path = output_dir / "threshold_sweep_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"💾 Results saved to {results_path}")
    
    # Find optimal thresholds
    f1_optimal_idx = results_df['f1_score'].idxmax()
    f1_optimal = results_df.loc[f1_optimal_idx]
    
    # Find precision >= 0.8 operating point
    high_precision = results_df[results_df['precision'] >= 0.8]
    if not high_precision.empty:
        precision_optimal_idx = high_precision['recall'].idxmax()
        precision_optimal = results_df.loc[precision_optimal_idx]
    else:
        precision_optimal = f1_optimal
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Precision-Recall Curve
    axes[0, 0].plot(recall, precision, 'b-', linewidth=2, label='PR Curve')
    axes[0, 0].scatter(f1_optimal['recall'], f1_optimal['precision'], 
                       color='red', s=100, zorder=5, label=f'F1 Optimal (τ={f1_optimal["threshold"]:.3f})')
    if not high_precision.empty:
        axes[0, 0].scatter(precision_optimal['recall'], precision_optimal['precision'], 
                           color='orange', s=100, zorder=5, label=f'Precision≥0.8 (τ={precision_optimal["threshold"]:.3f})')
    axes[0, 0].set_xlabel('Recall')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision-Recall Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ROC Curve
    axes[0, 1].plot(fpr, tpr, 'g-', linewidth=2, label='ROC Curve')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    axes[0, 1].scatter(f1_optimal['fp'] / (f1_optimal['fp'] + f1_optimal['tn']), 
                       f1_optimal['tp'] / (f1_optimal['tp'] + f1_optimal['fn']), 
                       color='red', s=100, zorder=5, label=f'F1 Optimal')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. F1 Score vs Threshold
    axes[0, 2].plot(results_df['threshold'], results_df['f1_score'], 'r-', linewidth=2)
    axes[0, 2].scatter(f1_optimal['threshold'], f1_optimal['f1_score'], 
                       color='red', s=100, zorder=5, label=f'Optimal F1={f1_optimal["f1_score"]:.3f}')
    axes[0, 2].set_xlabel('Threshold')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].set_title('F1 Score vs Threshold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Precision vs Threshold
    axes[1, 0].plot(results_df['threshold'], results_df['precision'], 'b-', linewidth=2, label='Precision')
    axes[1, 0].plot(results_df['threshold'], results_df['recall'], 'g-', linewidth=2, label='Recall')
    axes[1, 0].axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='Precision ≥ 0.8')
    axes[1, 0].scatter(f1_optimal['threshold'], f1_optimal['precision'], 
                       color='red', s=100, zorder=5, label=f'F1 Optimal')
    if not high_precision.empty:
        axes[1, 0].scatter(precision_optimal['threshold'], precision_optimal['precision'], 
                           color='orange', s=100, zorder=5, label=f'Precision≥0.8')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision & Recall vs Threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Confusion Matrix at F1 Optimal
    cm_f1 = np.array([[f1_optimal['tn'], f1_optimal['fp']], 
                      [f1_optimal['fn'], f1_optimal['tp']]])
    sns.heatmap(cm_f1, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    axes[1, 1].set_title(f'Confusion Matrix (τ={f1_optimal["threshold"]:.3f})')
    
    # 6. Metrics Summary
    summary_text = f"""F1 Optimal Threshold: {f1_optimal['threshold']:.3f}
Precision: {f1_optimal['precision']:.3f}
Recall: {f1_optimal['recall']:.3f}
F1 Score: {f1_optimal['f1_score']:.3f}
Accuracy: {f1_optimal['accuracy']:.3f}

Precision≥0.8 Threshold: {precision_optimal['threshold']:.3f}
Precision: {precision_optimal['precision']:.3f}
Recall: {precision_optimal['recall']:.3f}
F1 Score: {precision_optimal['f1_score']:.3f}"""
    
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 2].set_title('Threshold Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "threshold_sweep.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Plot saved to {plot_path}")
    
    # Print summary
    logger.info("\n🎯 Threshold Sweep Summary:")
    logger.info("=" * 50)
    logger.info(f"F1 Optimal Threshold: {f1_optimal['threshold']:.3f}")
    logger.info(f"  Precision: {f1_optimal['precision']:.3f}")
    logger.info(f"  Recall: {f1_optimal['recall']:.3f}")
    logger.info(f"  F1 Score: {f1_optimal['f1_score']:.3f}")
    
    if not high_precision.empty:
        logger.info(f"\nPrecision≥0.8 Operating Point:")
        logger.info(f"  Threshold: {precision_optimal['threshold']:.3f}")
        logger.info(f"  Precision: {precision_optimal['precision']:.3f}")
        logger.info(f"  Recall: {precision_optimal['recall']:.3f}")
        logger.info(f"  F1 Score: {precision_optimal['f1_score']:.3f}")
    
    # Save optimal thresholds to config
    config['optimal_thresholds'] = {
        'f1_optimal': float(f1_optimal['threshold']),
        'precision_optimal': float(precision_optimal['threshold']) if not high_precision.empty else float(f1_optimal['threshold'])
    }
    
    return results_df, config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run threshold sweep experiment")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    
    args = parser.parse_args()
    
    try:
        results, config = run_threshold_sweep_experiment(args.config)
        print(f"\n✅ Threshold sweep experiment completed successfully!")
        print(f"Results saved to: artifacts/figures/threshold_sweep_results.csv")
        print(f"Plot saved to: artifacts/figures/threshold_sweep.png")
        print(f"\n🎯 Optimal thresholds identified:")
        print(f"  F1 Optimal: {config['optimal_thresholds']['f1_optimal']:.3f}")
        print(f"  Precision≥0.8: {config['optimal_thresholds']['precision_optimal']:.3f}")
        
    except Exception as e:
        logging.error(f"❌ Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
