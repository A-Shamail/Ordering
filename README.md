# Feature Ordering Analysis

Analyzes feature importance and ordering using random walks and L-score methods.

## Quick Start

```bash
python example_usage.py
```

That's it! Results will be saved to `outputs/` with timestamped folders.

## What It Does

- Generates random feature orderings (paths)
- Trains models progressively adding features
- Calculates L-scores to identify feature relationships
- Creates visualizations (L-score matrices, pair plots)

## Configuration

Edit `example_usage.py` to customize:

```python
analyzer = FeatureAnalyzer(
    approach='walks',      # 'walks' or 'combos'
    l_score_method='CDM',  # 'CDM' or 'AS'
    model_type='rf',       # 'rf', 'linear', 'mlp'
    n_paths=50             # Number of random paths
)
```

## Data

- Input: CSV files from `../data/` directory
- Output: Timestamped folders in `outputs/`

## Output Files

Each run creates:
- `L_score_matrix.png/csv` - Feature relationship matrix
- `all_pairs_scatterplot.png` - Combined pairwise plots
- `pairs/*.pdf` - Individual feature pair plots
- `run_config.json` - Configuration used

## Performance Tips

**Slow?** Reduce `n_paths` from 50 to 10-20 for faster runs.

## Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tqdm
```

---


