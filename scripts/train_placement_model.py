from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_model_pipeline import train_baseline
import json

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train the ATLAS placement-outlook Logistic Regression baseline.')
    parser.add_argument('--data-dir', default=str(ROOT/'data'))
    parser.add_argument('--output-dir', default=str(ROOT/'outputs'))
    args = parser.parse_args()
    model, metrics, coef_df, _ = train_baseline(args.data_dir, args.output_dir)
    print(json.dumps(metrics, indent=2))
    print('\nLargest standardized model associations (not causal):')
    print(coef_df.head(12).to_string(index=False))
