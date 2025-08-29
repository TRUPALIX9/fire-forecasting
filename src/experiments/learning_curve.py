"""
Learning curve experiment: PR-AUC vs training years.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from config import load_config, set_random_seed
from utils import timer, ensure_dir
from data_sources.raws import fetch_all_sites_raws_data
from data_sources.firms import create_fire_labels_for_sites
from features.engineer import engineer_all_features
from splits.chrono import create_chronological_pipeline
from models.ann import create_ann_model, train_ann_model, evaluate_ann_model


def run_learning_curve_experiment(config_path: str):
    """Run learning curve experiment with varying training years."""
    
    # Load configuration
    config = load_config(config_path)
    set_random_seed(config['project']['seed'])
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🔥 Starting Learning Curve Experiment")
    logger.info(f"Project: {config['project']['area']}")
    
    # Ensure output directories
    output_dir = Path("artifacts/figures")
    ensure_dir(output_dir)
    
    # Fetch base data
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
    
    # Learning curve experiment
    years_range = range(config['project']['min_years'], config['project']['preferred_years'] + 1)
    results = []
    
    for years in years_range:
        logger.info(f"📈 Training with {years} years of data...")
        
        # Filter data by years
        end_date = pd.Timestamp.now().normalize()
        start_date = end_date - pd.DateOffset(years=years)
        
        filtered_data = feature_data[
            (feature_data['date'] >= start_date) & 
            (feature_data['date'] <= end_date)
        ].copy()
        
        if len(filtered_data) > config['project']['max_rows']:
            logger.warning(f"Data exceeds max_rows limit: {len(filtered_data)} > {config['project']['max_rows']}")
            # Downsample train set only
            train_data = filtered_data[filtered_data['date'] < filtered_data['date'].quantile(0.7)]
            test_val_data = filtered_data[filtered_data['date'] >= filtered_data['date'].quantile(0.7)]
            filtered_data = pd.concat([train_data, test_val_data])
        
        # Split data
        splits = create_chronological_pipeline(filtered_data, config)
        
        # Train ANN model
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
        
        # Evaluate model
        metrics = evaluate_ann_model(model, splits['X_test'], splits['y_test'])
        
        results.append({
            'years': years,
            'rows_total': len(filtered_data),
            'train_rows': len(splits['X_train']),
            'val_rows': len(splits['X_val']),
            'test_rows': len(splits['X_test']),
            'pr_auc': metrics['pr_auc'],
            'roc_auc': metrics['roc_auc'],
            'f1_score': metrics['f1_score'],
            'train_time_s': train_timer.elapsed,
            'positives': splits['y_train'].sum() + splits['y_val'].sum() + splits['y_test'].sum()
        })
        
        logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}, Rows: {len(filtered_data)}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_path = output_dir / "learning_curve_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"💾 Results saved to {results_path}")
    
    # Create learning curve plot
    plt.figure(figsize=(12, 8))
    
    # PR-AUC vs Years
    plt.subplot(2, 2, 1)
    plt.plot(results_df['years'], results_df['pr_auc'], 'o-', linewidth=2, markersize=8)
    plt.xlabel('Training Years')
    plt.ylabel('PR-AUC')
    plt.title('PR-AUC vs Training Years')
    plt.grid(True, alpha=0.3)
    
    # ROC-AUC vs Years
    plt.subplot(2, 2, 2)
    plt.plot(results_df['years'], results_df['roc_auc'], 's-', linewidth=2, markersize=8, color='orange')
    plt.xlabel('Training Years')
    plt.ylabel('ROC-AUC')
    plt.title('ROC-AUC vs Training Years')
    plt.grid(True, alpha=0.3)
    
    # F1 Score vs Years
    plt.subplot(2, 2, 3)
    plt.plot(results_df['years'], results_df['f1_score'], '^-', linewidth=2, markersize=8, color='green')
    plt.xlabel('Training Years')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Training Years')
    plt.grid(True, alpha=0.3)
    
    # Training Time vs Years
    plt.subplot(2, 2, 4)
    plt.plot(results_df['years'], results_df['train_time_s'], 'd-', linewidth=2, markersize=8, color='red')
    plt.xlabel('Training Years')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time vs Training Years')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "learning_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Plot saved to {plot_path}")
    
    # Print summary
    logger.info("\n📊 Learning Curve Summary:")
    logger.info("=" * 50)
    for _, row in results_df.iterrows():
        logger.info(f"Years: {row['years']:2d} | "
                   f"PR-AUC: {row['pr_auc']:.4f} | "
                   f"Rows: {row['rows_total']:5d} | "
                   f"Time: {row['train_time_s']:5.1f}s")
    
    # Find optimal configuration
    best_idx = results_df['pr_auc'].idxmax()
    best_config = results_df.loc[best_idx]
    
    logger.info(f"\n🏆 Best Configuration:")
    logger.info(f"Years: {best_config['years']}")
    logger.info(f"PR-AUC: {best_config['pr_auc']:.4f}")
    logger.info(f"Total Rows: {best_config['rows_total']}")
    logger.info(f"Training Time: {best_config['train_time_s']:.1f}s")
    
    return results_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run learning curve experiment")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    
    args = parser.parse_args()
    
    try:
        results = run_learning_curve_experiment(args.config)
        print(f"\n✅ Learning curve experiment completed successfully!")
        print(f"Results saved to: artifacts/figures/learning_curve_results.csv")
        print(f"Plot saved to: artifacts/figures/learning_curve.png")
        
    except Exception as e:
        logging.error(f"❌ Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
