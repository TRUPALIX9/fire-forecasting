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

from config import load_config, setup_logging
from utils import timer, ensure_dir, save_json, save_csv
from data_sources.raws import fetch_all_sites_raws_data, get_site_coordinates
from data_sources.firms import fetch_firms_data, create_fire_labels_for_sites
from data_sources.frap import fetch_frap_data
from features.engineer import engineer_all_features, get_feature_columns
from labeling.make_labels import create_labels_pipeline
from splits.chrono import create_chronological_pipeline, validate_splits
from models.baselines import evaluate_baseline_models
from models.ann import train_ann_model, evaluate_ann_model, save_ann_model, plot_training_history

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
        firms_df = fetch_firms_data(start_date, end_date)
    
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
        
        # Create chronological splits
        split_result = create_chronological_pipeline(
            labeled_df, feature_cols, 'fire_tomorrow'
        )
        
        # Validate splits
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
            save_ann_model(ann_results['model'], models_dir / 'ann_model.h5')
        
        # Save metrics
        metrics_dir = output_dir / 'metrics'
        ensure_dir(metrics_dir)
        
        # Save baseline metrics
        for model_name, results in baseline_results.items():
            save_json(results['metrics'], metrics_dir / f'{model_name}_metrics.json')
            save_json(results['test_metrics'], metrics_dir / f'{model_name}_test_metrics.json')
        
        # Save ANN metrics
        if model_type == 'ann':
            save_json(ann_results['metrics'], metrics_dir / 'ann_metrics.json')
            save_json(test_results['metrics'], metrics_dir / 'ann_test_metrics.json')
        
        # Save curves data
        curves_dir = output_dir / 'curves'
        ensure_dir(curves_dir)
        
        # Save PR/ROC curves for ANN
        if model_type == 'ann':
            # PR curve
            pr_data = pd.DataFrame({
                'precision': ann_results['metrics']['precision_recall_curve']['precision'],
                'recall': ann_results['metrics']['precision_recall_curve']['recall'],
                'thresholds': ann_results['metrics']['precision_recall_curve']['thresholds']
            })
            save_csv(pr_data, curves_dir / 'ann_pr_curve.csv')
            
            # ROC curve
            roc_data = pd.DataFrame({
                'fpr': ann_results['metrics']['roc_curve']['fpr'],
                'tpr': ann_results['metrics']['roc_curve']['tpr'],
                'thresholds': ann_results['metrics']['roc_curve']['thresholds']
            })
            save_csv(roc_data, curves_dir / 'ann_roc_curve.csv')
        
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
        frap_gdf.to_file(geo_dir / 'frap_tri.geojson', driver='GeoJSON')
        
        # Save site coordinates
        sites_data = pd.DataFrame([
            {'site': site, 'lat': lat, 'lon': lon}
            for site, (lat, lon) in get_site_coordinates().items()
        ])
        save_csv(sites_data, geo_dir / 'sites.csv')
        
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
