#!/usr/bin/env python3
"""
Fire Forecasting ML Pipeline
End-to-end pipeline for training wildfire prediction models
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from src.config import load_config, setup_logging
from src.utils import timer, ensure_dir, save_json, save_csv
from src.data_sources.raws import fetch_all_sites_raws_data, get_site_coordinates
from src.data_sources.firms import load_firms_data
from src.data_sources.frap import fetch_frap_data
from src.features.engineer import engineer_all_features, get_feature_columns
from src.labeling.make_labels import create_labels_pipeline
from src.splits.chrono import create_chronological_pipeline, validate_splits
from src.models.baselines import evaluate_baseline_models
from src.models.ann import train_ann_model, evaluate_ann_model, save_ann_model, plot_training_history

def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='Fire Forecasting ML Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path')
    parser.add_argument('--output-dir', type=str, default='artifacts', help='Output directory')
    parser.add_argument('--force-retrain', action='store_true', help='Force retraining even if artifacts exist')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Fire Forecasting ML Pipeline")
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        ensure_dir(output_dir)
        
        # Run pipeline
        run_pipeline(config, output_dir, args.force_retrain)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

def run_pipeline(config: dict, output_dir: Path, force_retrain: bool = False):
    """Run the complete pipeline"""
    
    # Get configuration
    project_config = config.get('project', {})
    data_config = config.get('data', {})
    train_config = config.get('train', {})
    
    # Check if we should skip training
    if not force_retrain and (output_dir / 'models' / 'ann_model.h5').exists():
        print("Model already exists. Use --force-retrain to retrain.")
        return
    
    # Step 1: Data Fetching
    print("\n" + "="*60)
    print("STEP 1: DATA FETCHING")
    print("="*60)
    
    # Calculate date range
    end_year = project_config.get('end_year', 2024)
    preferred_years = project_config.get('preferred_years', 8)
    start_year = end_year - preferred_years + 1
    
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Get sites
    sites = [site['name'] for site in data_config.get('sites', [])]
    print(f"Processing {len(sites)} sites")
    
    # Fetch RAWS data
    with timer("RAWS data fetch"):
        raws_df = fetch_all_sites_raws_data(sites, start_date, end_date)
    
    # Fetch FIRMS data
    with timer("FIRMS data fetch"):
        # Load real FIRMS data from DL_FIRE_* subdirectories
        firms_df = load_firms_data(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            min_confidence="n"  # Use nominal confidence for real data
        )
    
    # Fetch FRAP data
    with timer("FRAP data fetch"):
        frap_gdf = fetch_frap_data(start_year, end_year)
    
    # Step 2: Feature Engineering
    print("\n" + "="*60)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*60)
    
    with timer("Feature engineering"):
        # Get site coordinates
        site_coords = get_site_coordinates()
        
        # Engineer features
        features_df = engineer_all_features(raws_df, config, site_coords)
    
    # Step 3: Label Creation
    print("\n" + "="*60)
    print("STEP 3: LABEL CREATION")
    print("="*60)
    
    with timer("Label creation"):
        # Ensure materialized 'date' column in features prior to labeling
        import pandas as pd
        if "date" not in features_df.columns:
            if isinstance(features_df.index, pd.DatetimeIndex):
                features_df["date"] = features_df.index.normalize()
            else:
                features_df["date"] = pd.to_datetime(features_df.index, errors="coerce").normalize()
                features_df = features_df.dropna(subset=["date"])
        labeled_df = create_labels_pipeline(features_df, firms_df, config)
    
    # Check row budget
    max_rows = project_config.get('max_rows', 12000)
    if len(labeled_df) > max_rows:
        print(f"Dataset exceeds row budget ({len(labeled_df)} > {max_rows})")
        print("Reducing data size...")
        
        # Reduce years if possible
        min_years = project_config.get('min_years', 5)
        if end_year - start_year > min_years:
            # Reduce years
            years_to_remove = (len(labeled_df) - max_rows) // (len(labeled_df) // (end_year - start_year + 1))
            new_start_year = start_year + years_to_remove
            print(f"Reducing years: {start_year}-{end_year} -> {new_start_year}-{end_year}")
            
            # Filter data
            labeled_df = labeled_df[
                labeled_df['date'] >= datetime(new_start_year, 1, 1)
            ]
        else:
            # Downsample training data only
            print("Downsampling training data...")
            train_frac = max_rows / len(labeled_df)
            labeled_df = labeled_df.sample(frac=train_frac, random_state=42)
    
    print(f"Final dataset size: {len(labeled_df)} rows")
    
    # Step 4: Data Splitting
    print("\n" + "="*60)
    print("STEP 4: DATA SPLITTING")
    print("="*60)
    
    with timer("Data splitting"):
        # Get feature columns
        feature_cols = get_feature_columns(labeled_df)
        print(f"Feature columns: {len(feature_cols)}")
        
        # Create chronological splits and fit scaler on train only
        from src.splits.chrono import create_chronological_pipeline, validate_splits
        split_result = create_chronological_pipeline(labeled_df, feature_cols, target_col='fire_tomorrow')
        validate_splits(split_result['splits'])
    
    # Step 5: Model Training
    print("\n" + "="*60)
    print("STEP 5: MODEL TRAINING")
    print("="*60)
    
    # Get prepared data
    prepared_data = split_result['prepared_data']
    class_weights = split_result['class_weights']
    
    X_train = prepared_data['train']['X']
    y_train = prepared_data['train']['y']
    X_val = prepared_data['val']['X']
    y_val = prepared_data['val']['y']
    X_test = prepared_data['test']['X']
    y_test = prepared_data['test']['y']
    
    # Train baseline models
    with timer("Baseline model training"):
        baseline_results = evaluate_baseline_models(
            X_train, y_train, X_val, y_val, X_test, y_test,
            class_weight=train_config.get('class_weight', True)
        )
    
    # Train ANN model
    model_type = train_config.get('model', 'ann')
    if model_type == 'ann':
        with timer("ANN model training"):
            ann_results = train_ann_model(
                X_train, y_train, X_val, y_val, config, class_weights
            )
            
            # Evaluate on test set
            test_results = evaluate_ann_model(ann_results['model'], X_test, y_test)
            ann_results['test_results'] = test_results
    
    # Step 6: Save Results
    print("\n" + "="*60)
    print("STEP 6: SAVING RESULTS")
    print("="*60)
    
    with timer("Saving results"):
        # Save models
        models_dir = output_dir / 'models'
        ensure_dir(models_dir)
        
        if model_type == 'ann':
            # Save Keras model in .keras format for backend
            save_ann_model(ann_results['model'], models_dir / 'model.keras')
        
        # Save metrics
        metrics_dir = output_dir / 'metrics'
        ensure_dir(metrics_dir)
        
        # Save required metrics for backend
        # Global metrics: prefer ANN test metrics, else best baseline
        global_metrics = {}
        if model_type == 'ann':
            global_metrics = ann_results['test_results']['metrics']
        else:
            # pick best baseline by pr_auc
            best_baseline = max(baseline_results.items(), key=lambda x: x[1]['test_metrics']['pr_auc'])
            global_metrics = best_baseline[1]['test_metrics']
        save_json(global_metrics, metrics_dir / 'global_metrics.json')

        # Per-site metrics CSV (basic aggregates if available)
        try:
            import pandas as pd
            test_split_df = split_result['splits']['test']
            site_col = 'site' if 'site' in test_split_df.columns else None
            if site_col and model_type == 'ann':
                site_df = test_split_df.copy()
                site_df['true_label'] = y_test
                site_df['predicted_probability'] = ann_results['test_results']['predictions']
                per_site = site_df.groupby(site_col).agg(
                    n_samples=('true_label', 'size'),
                    positives=('true_label', 'sum'),
                    positive_rate=('true_label', 'mean'),
                    avg_pred=('predicted_probability', 'mean')
                ).reset_index()
                save_csv(per_site, metrics_dir / 'per_site_metrics.csv')
            else:
                # write an empty stub if site info not available
                empty = pd.DataFrame(columns=['site','n_samples','positives','positive_rate','avg_pred'])
                save_csv(empty, metrics_dir / 'per_site_metrics.csv')
        except Exception:
            # Ensure file exists to satisfy backend even if aggregation fails
            import pandas as pd
            empty = pd.DataFrame(columns=['site','n_samples','positives','positive_rate','avg_pred'])
            save_csv(empty, metrics_dir / 'per_site_metrics.csv')
        
        # Save figures expected by backend
        figures_dir = output_dir / 'figures'
        ensure_dir(figures_dir)
        if model_type == 'ann':
            import matplotlib.pyplot as plt
            # PR Curve
            pr = ann_results['metrics']['precision_recall_curve']
            plt.figure()
            plt.plot(pr['recall'], pr['precision'], label='PR')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.grid(True)
            plt.savefig(figures_dir / 'pr_curve.png', dpi=150, bbox_inches='tight')
            plt.close()

            # ROC Curve
            roc = ann_results['metrics']['roc_curve']
            plt.figure()
            plt.plot(roc['fpr'], roc['tpr'], label='ROC')
            plt.plot([0,1],[0,1],'k--', alpha=0.3)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.grid(True)
            plt.savefig(figures_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
            plt.close()

            # Confusion Matrix at optimal threshold
            cm = ann_results['metrics']['confusion_matrix']
            import numpy as np
            cm_arr = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
            plt.figure()
            plt.imshow(cm_arr, cmap='Blues')
            for (i, j), val in np.ndenumerate(cm_arr):
                plt.text(j, i, int(val), ha='center', va='center')
            plt.xticks([0,1], ['Pred 0','Pred 1'])
            plt.yticks([0,1], ['True 0','True 1'])
            plt.title('Confusion Matrix')
            plt.colorbar()
            plt.savefig(figures_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # Save test predictions
        predictions_dir = output_dir / 'predictions'
        ensure_dir(predictions_dir)
        
        if model_type == 'ann':
            test_pred_df = pd.DataFrame({
                'site': split_result['splits']['test']['site'],
                'date': split_result['splits']['test']['date'],
                'true_label': y_test,
                'predicted_probability': test_results['predictions']
            })
            save_csv(test_pred_df, predictions_dir / 'ann_test_predictions.csv')
        
        # Save FRAP data
        geo_dir = output_dir / 'geo'
        ensure_dir(geo_dir)
        # Save FRAP geojson with expected filename for backend
        try:
            frap_gdf.to_file(geo_dir / 'frap_fire_perimeters.geojson', driver='GeoJSON')
        except Exception:
            pass
        
        # Save site coordinates
        # Save sites as GeoJSON
        import json
        coords = get_site_coordinates()
        features = []
        for site, (lat, lon) in coords.items():
            features.append({
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [float(lon), float(lat)]},
                'properties': {'site': site}
            })
        with open(geo_dir / 'sites.geojson', 'w') as f:
            json.dump({'type': 'FeatureCollection', 'features': features}, f)
        
        # Save summary
        summary = create_pipeline_summary(config, baseline_results, ann_results if model_type == 'ann' else None)
        save_json(summary, output_dir / 'pipeline_summary.json')
        
        # Create README
        create_summary_readme(summary, output_dir)
    
    print("\nPipeline completed successfully!")
    print(f"Results saved to: {output_dir}")

def create_pipeline_summary(config: dict, baseline_results: dict, ann_results: dict = None) -> dict:
    """Create pipeline summary"""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'baseline_results': {},
        'ann_results': None,
        'summary_stats': {}
    }
    
    # Add baseline results
    for model_name, results in baseline_results.items():
        summary['baseline_results'][model_name] = {
            'validation_metrics': {
                'pr_auc': results['metrics']['pr_auc'],
                'roc_auc': results['metrics']['roc_auc'],
                'f1_at_threshold': results['metrics']['f1_at_threshold']
            },
            'test_metrics': {
                'pr_auc': results['test_metrics']['pr_auc'],
                'roc_auc': results['test_metrics']['roc_auc'],
                'f1_at_threshold': results['test_metrics']['f1_at_threshold']
            },
            'training_time': results['metrics']['train_time_s']
        }
    
    # Add ANN results
    if ann_results:
        summary['ann_results'] = {
            'validation_metrics': {
                'pr_auc': ann_results['metrics']['pr_auc'],
                'roc_auc': ann_results['metrics']['roc_auc'],
                'f1_at_threshold': ann_results['metrics']['f1_at_threshold']
            },
            'test_metrics': {
                'pr_auc': ann_results['test_results']['metrics']['pr_auc'],
                'roc_auc': ann_results['test_results']['metrics']['roc_auc'],
                'f1_at_threshold': ann_results['test_results']['metrics']['f1_at_threshold']
            },
            'training_time': ann_results['metrics']['train_time_s'],
            'training_epochs': ann_results['metrics']['training_epochs']
        }
    
    # Add summary stats
    best_baseline = max(
        baseline_results.items(),
        key=lambda x: x[1]['metrics']['pr_auc']
    )
    
    summary['summary_stats'] = {
        'best_baseline_model': best_baseline[0],
        'best_baseline_pr_auc': best_baseline[1]['metrics']['pr_auc'],
        'best_baseline_roc_auc': best_baseline[1]['metrics']['roc_auc'],
        'ann_pr_auc': ann_results['metrics']['pr_auc'] if ann_results else None,
        'ann_roc_auc': ann_results['metrics']['roc_auc'] if ann_results else None,
        'improvement_over_baseline': {
            'pr_auc': (ann_results['metrics']['pr_auc'] - best_baseline[1]['metrics']['pr_auc']) if ann_results else None,
            'roc_auc': (ann_results['metrics']['roc_auc'] - best_baseline[1]['metrics']['roc_auc']) if ann_results else None
        }
    }
    
    return summary

def create_summary_readme(summary: dict, output_dir: Path):
    """Create summary README file"""
    readme_content = f"""# Fire Forecasting Pipeline Results

Generated on: {summary['timestamp']}

## Configuration
- **Area**: {summary['config']['project']['area']}
- **Date Range**: {summary['config']['project']['end_year'] - summary['config']['project']['preferred_years'] + 1} to {summary['config']['project']['end_year']}
- **Model Type**: {summary['config']['train']['model']}
- **Labeling Method**: {summary['config']['labeling']['label_method']}

## Results Summary

### Baseline Models
"""
    
    for model_name, results in summary['baseline_results'].items():
        readme_content += f"""
**{model_name.replace('_', ' ').title()}**
- Validation PR-AUC: {results['validation_metrics']['pr_auc']:.3f}
- Validation ROC-AUC: {results['validation_metrics']['roc_auc']:.3f}
- Test PR-AUC: {results['test_metrics']['pr_auc']:.3f}
- Test ROC-AUC: {results['test_metrics']['roc_auc']:.3f}
- Training Time: {results['training_time']:.2f}s
"""
    
    if summary['ann_results']:
        readme_content += f"""
### Neural Network (ANN)
- Validation PR-AUC: {summary['ann_results']['validation_metrics']['pr_auc']:.3f}
- Validation ROC-AUC: {summary['ann_results']['validation_metrics']['roc_auc']:.3f}
- Test PR-AUC: {summary['ann_results']['test_metrics']['pr_auc']:.3f}
- Test ROC-AUC: {summary['ann_results']['test_metrics']['roc_auc']:.3f}
- Training Time: {summary['ann_results']['training_time']:.2f}s
- Training Epochs: {summary['ann_results']['training_epochs']}

### Performance Comparison
- Best Baseline: {summary['summary_stats']['best_baseline_model']} (PR-AUC: {summary['summary_stats']['best_baseline_pr_auc']:.3f})
- ANN Improvement: +{summary['summary_stats']['improvement_over_baseline']['pr_auc']:.3f} PR-AUC
"""
    
    readme_content += f"""
## Files Generated

### Models
- `models/` - Trained model files

### Metrics
- `metrics/` - Model performance metrics

### Curves
- `curves/` - PR and ROC curve data

### Predictions
- `predictions/` - Test set predictions

### Geographic Data
- `geo/` - Site coordinates and FRAP fire perimeters

## Usage

To use the trained model:

```python
from src.models.ann import load_ann_model
model = load_ann_model('artifacts/models/ann_model.h5')
predictions = model.predict(features)
```
"""
    
    # Write README
    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme_content)

if __name__ == "__main__":
    main()
