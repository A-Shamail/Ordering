"""
Example usage of the FeatureAnalyzer class.
"""

import pandas as pd
import os
from datetime import datetime
from feature_analyzer import FeatureAnalyzer

def main():
    # Load example dataset

  
    #X1 = np.random.uniform(-1, 1, n_samples)
    #X2 = X1**3
    #X3 = np.random.uniform(-1, 1, n_samples)
    #X4 = np.random.uniform(-1, 1, n_samples)
    #X5 = np.random.uniform(-1, 1, n_samples)
    #X6 = np.random.uniform(-1, 1, n_samples)
    
    #Y = np.sin(X2) + (X3*X4) + (X5/X6)
    
  

    #csv_file = '../datasets/dependent_dataset_multiplicative.csv'
    # csv_file = '../data/redundant_dataset_polynomial.csv'
    csv_file = '../data/test_tri_5000_new.csv'
    df = pd.read_csv(csv_file)
    
    # Extract dataset name from the CSV file path
    dataset_name = csv_file.split('/')[-1].replace('.csv', '')
    # Create one main timestamped folder for this run
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    main_output_dir = f'outputs/{dataset_name}_comparison_run_{current_time}______'
    os.makedirs(main_output_dir, exist_ok=True)
    
    print("=== Feature Analysis Comparison ===")
    print(f"Dataset: {dataset_name}")
    print(f"All outputs will be saved to: {main_output_dir}\n")
    
    # Example 1: Combinations approach with AS L-score method
    print("1. Running combinations approach with AS L-score method...")
    analyzer1 = FeatureAnalyzer(
        approach='walks',
        l_score_method='CDM',
        model_type='rf',
        n_paths=50,  
        output_dir=f'{main_output_dir}/walks_CDM'
    )
    results1 = analyzer1.analyze_features(df)
    analyzer1.create_visualizations(results1, csv_file)
    print("   Complete!\n")
    
    # Example 2: Random walks approach with CDM L-score method
    # print("2. Running random walks approach with CDM L-score method...")
    # analyzer2 = FeatureAnalyzer(
    #     approach='walks',
    #     l_score_method='CDM',
    #     model_type='rf',
    #     n_paths=50,
    #     output_dir=f'{main_output_dir}/walks_CDM'
    # )
    # results2 = analyzer2.analyze_features(df)
    # analyzer2.create_visualizations(results2, csv_file)
    # print("   Complete!\n")
    
    # Example 3: Cross-comparison - Random walks approach with AS L-scores
    print("3. Running random walks approach with AS L-score method...")
    # analyzer3 = FeatureAnalyzer(
    #    approach='walks',
    #    l_score_method='AS',
    #    model_type='rf',
    #    n_paths=50,
    #    output_dir=f'{main_output_dir}/walks_AS'
    # )
    #results3 = analyzer3.analyze_features(df)
    #analyzer3.create_visualizations(results3, csv_file)
    #print("   Complete!\n")
    
    #    Example 4: Combinations approach with CDM L-scores
    #print("4. Running combinations approach with CDM L-score method...")
    #analyzer4 = FeatureAnalyzer(
    #    approach='combos',
    #    l_score_method='CDM',
    #    model_type='rf',
    #    output_dir=f'{main_output_dir}/combos_CDM'
    #)
    #results4 = analyzer4.analyze_features(df)
    #analyzer4.create_visualizations(results4, csv_file)
    #print("   Complete!\n")
    
    print("=== Analysis Complete! ===")
    print(f"Dataset: {dataset_name}")
    print(f"All results saved in: {main_output_dir}")
    print("\nFolder structure:")
    print("├── combos_AS/     (Exhaustive combinations + AS L-scores)")
    print("├── walks_CDM/     (Random walks + CDM L-scores)")  
    print("├── walks_AS/      (Random walks + AS L-scores)")
    print("└── combos_CDM/    (Exhaustive combinations + CDM L-scores)")
    print("\nEach subfolder contains:")
    print("  - L_score_matrix.png & .csv")
    print("  - all_pairs_scatterplot.png")
    print("  - pairs/ (individual scatterplots)")
    print("  - run_config.json")

if __name__ == "__main__":
    main() 