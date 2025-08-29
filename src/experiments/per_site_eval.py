"""
Per-site evaluation experiment: individual site performance analysis.
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


def run_per_site_evaluation(config_path: str):
    """Run per-site evaluation to analyze individual site performance."""
    
    # Load configuration
    config = load_config(config_path)
    set_random_seed(config['project']['seed'])
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🏘️ Starting Per-Site Evaluation Experiment")
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
    
    # Get unique sites
    sites = feature_data['site'].unique()
    logger.info(f"📍 Found {len(sites)} sites for evaluation")
    
    # Per-site evaluation
    site_results = []
    site_predictions = {}
    
    for site in sites:
        logger.info(f"🔍 Evaluating site: {site}")
        
        # Filter data for this site
        site_data = feature_data[feature_data['site'] == site].copy()
        
        if len(site_data) < 100:  # Skip sites with too little data
            logger.warning(f"  Skipping {site}: insufficient data ({len(site_data)} rows)")
            continue
        
        # Split data chronologically
        splits = create_chronological_pipeline(site_data, config)
        
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
        
        # Get predictions for sparklines
        y_pred_proba = model.predict(splits['X_test'])
        y_pred_proba = y_pred_proba.flatten()
        
        # Store results
        site_results.append({
            'site': site,
            'rows_total': len(site_data),
            'train_rows': len(splits['X_train']),
            'val_rows': len(splits['X_val']),
            'test_rows': len(splits['X_test']),
            'pr_auc': metrics['pr_auc'],
            'roc_auc': metrics['roc_auc'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'train_time_s': train_timer.elapsed,
            'positives': splits['y_test'].sum(),
            'positives_total': splits['y_train'].sum() + splits['y_val'].sum() + splits['y_test'].sum()
        })
        
        # Store predictions for visualization
        site_predictions[site] = {
            'dates': splits['test_dates'],
            'y_true': splits['y_test'],
            'y_pred_proba': y_pred_proba
        }
        
        logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}, F1: {metrics['f1_score']:.4f}, "
                   f"Rows: {len(site_data)}, Positives: {splits['y_test'].sum()}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(site_results)
    
    # Save results
    results_path = output_dir / "per_site_evaluation.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"💾 Results saved to {results_path}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. PR-AUC by Site
    axes[0, 0].barh(results_df['site'], results_df['pr_auc'], color='skyblue')
    axes[0, 0].set_xlabel('PR-AUC')
    axes[0, 0].set_title('PR-AUC by Site')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # 2. F1 Score by Site
    axes[0, 1].barh(results_df['site'], results_df['f1_score'], color='lightgreen')
    axes[0, 1].set_xlabel('F1 Score')
    axes[0, 1].set_title('F1 Score by Site')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # 3. Data Coverage by Site
    axes[0, 2].barh(results_df['site'], results_df['rows_total'], color='salmon')
    axes[0, 2].set_xlabel('Total Rows')
    axes[0, 2].set_title('Data Coverage by Site')
    axes[0, 2].grid(True, alpha=0.3, axis='x')
    
    # 4. Fire Events by Site
    axes[1, 0].barh(results_df['site'], results_df['positives_total'], color='orange')
    axes[1, 0].set_xlabel('Total Fire Events')
    axes[1, 0].set_title('Fire Events by Site')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 5. Training Time by Site
    axes[1, 1].barh(results_df['site'], results_df['train_time_s'], color='purple')
    axes[1, 1].set_xlabel('Training Time (s)')
    axes[1, 1].set_title('Training Time by Site')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    # 6. Performance vs Data Size Scatter
    scatter = axes[1, 2].scatter(results_df['rows_total'], results_df['pr_auc'], 
                                 c=results_df['f1_score'], s=100, alpha=0.7, cmap='viridis')
    axes[1, 2].set_xlabel('Total Rows')
    axes[1, 2].set_ylabel('PR-AUC')
    axes[1, 2].set_title('Performance vs Data Size')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1, 2])
    cbar.set_label('F1 Score')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "per_site_evaluation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Plot saved to {plot_path}")
    
    # Create sparkline plots for top performing sites
    top_sites = results_df.nlargest(6, 'pr_auc')['site'].tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, site in enumerate(top_sites):
        if site in site_predictions:
            pred_data = site_predictions[site]
            
            # Create sparkline
            axes[i].plot(pred_data['dates'], pred_data['y_pred_proba'], 'b-', linewidth=1, alpha=0.7)
            axes[i].scatter(pred_data['dates'][pred_data['y_true'] == 1], 
                           pred_data['y_pred_proba'][pred_data['y_true'] == 1], 
                           color='red', s=20, alpha=0.8, label='Actual Fires')
            
            axes[i].set_title(f"{site}\nPR-AUC: {results_df[results_df['site'] == site]['pr_auc'].iloc[0]:.3f}")
            axes[i].set_ylabel('Fire Probability')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
            
            if i == 0:
                axes[i].legend()
    
    plt.tight_layout()
    
    # Save sparklines
    sparkline_path = output_dir / "per_site_sparklines.png"
    plt.savefig(sparkline_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Sparklines saved to {sparkline_path}")
    
    # Print summary
    logger.info("\n🏘️ Per-Site Evaluation Summary:")
    logger.info("=" * 60)
    
    # Top performers
    top_performers = results_df.nlargest(5, 'pr_auc')
    logger.info("🏆 Top 5 Performers (by PR-AUC):")
    for _, row in top_performers.iterrows():
        logger.info(f"  {row['site']:25s} | PR-AUC: {row['pr_auc']:.4f} | "
                   f"F1: {row['f1_score']:.4f} | Rows: {row['rows_total']:4d}")
    
    # Bottom performers
    bottom_performers = results_df.nsmallest(5, 'pr_auc')
    logger.info("\n📉 Bottom 5 Performers (by PR-AUC):")
    for _, row in bottom_performers.iterrows():
        logger.info(f"  {row['site']:25s} | PR-AUC: {row['pr_auc']:.4f} | "
                   f"F1: {row['f1_score']:.4f} | Rows: {row['rows_total']:4d}")
    
    # Overall statistics
    logger.info(f"\n📊 Overall Statistics:")
    logger.info(f"  Average PR-AUC: {results_df['pr_auc'].mean():.4f} ± {results_df['pr_auc'].std():.4f}")
    logger.info(f"  Average F1 Score: {results_df['f1_score'].mean():.4f} ± {results_df['f1_score'].std():.4f}")
    logger.info(f"  Total Sites Evaluated: {len(results_df)}")
    logger.info(f"  Total Rows: {results_df['rows_total'].sum():,}")
    logger.info(f"  Total Fire Events: {results_df['positives_total'].sum():,}")
    
    # Save detailed results
    detailed_results = {
        'summary': {
            'total_sites': len(results_df),
            'avg_pr_auc': float(results_df['pr_auc'].mean()),
            'avg_f1_score': float(results_df['f1_score'].mean()),
            'total_rows': int(results_df['rows_total'].sum()),
            'total_fire_events': int(results_df['positives_total'].sum())
        },
        'top_performers': top_performers.to_dict('records'),
        'bottom_performers': bottom_performers.to_dict('records'),
        'all_sites': results_df.to_dict('records')
    }
    
    # Save as JSON
    import json
    json_path = output_dir / "per_site_evaluation.json"
    with open(json_path, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    logger.info(f"💾 Detailed results saved to {json_path}")
    
    return results_df, site_predictions


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run per-site evaluation experiment")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    
    args = parser.parse_args()
    
    try:
        results, predictions = run_per_site_evaluation(args.config)
        print(f"\n✅ Per-site evaluation experiment completed successfully!")
        print(f"Results saved to: artifacts/figures/per_site_evaluation.csv")
        print(f"Plot saved to: artifacts/figures/per_site_evaluation.png")
        print(f"Sparklines saved to: artifacts/figures/per_site_sparklines.png")
        print(f"Detailed results saved to: artifacts/figures/per_site_evaluation.json")
        print(f"\n🏆 Top performer: {results.loc[results['pr_auc'].idxmax(), 'site']}")
        print(f"   PR-AUC: {results['pr_auc'].max():.4f}")
        
    except Exception as e:
        logging.error(f"❌ Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
