# NASA_battery_federated_learning
ML model trained utilizing the Flower AI Federated Learning Framework

## Reproduce Current Results

Run the full benchmark set from the repo root:

```bash
python run_all_results.py
```

This writes centralized Random Forest, federated Random Forest, and cyclic
federated boosted-tree results to `benchmark_outputs_latest/`, including JSON
reports, round curves, a comparison chart, and `summary.md`.

To include the Flower XGBoost cyclic benchmark, install the optional XGBoost
dependency and add `--include-xgboost`:

```bash
pip install -r requirements.txt
python run_all_results.py --include-xgboost
```
