<<<<<<< HEAD
# NASA Battery Federated Learning

This repo benchmarks State-of-Health (SOH) regression for NASA battery discharge
data using centralized and federated learning workflows.

The current result pipeline compares:

- centralized Random Forest
- federated Random Forest one-shot ensemble
- cyclic federated Gradient Boosted Trees
- optional cyclic federated XGBoost

SOH labels are normalized per battery by maximum positive discharge capacity.
Rows with missing or zero capacity are excluded from supervised training.

## Repository Layout

```text
NASA_battery_federated_learning/
  Traditional/
    benchmark_traditional.py       Centralized Random Forest benchmark
  Federated/soh_federated/
    simulate.py                    Federated Random Forest baseline
    boosted_simulate.py            Cyclic sklearn GradientBoosting benchmark
    xgboost_simulate.py            Cyclic Flower XGBoost benchmark
    sohfed/task.py                 Data loading, features, metrics, RF helpers
    sohfed/benchmarks.py           Shared benchmark report schema
  run_all_results.py               Runs the full benchmark pipeline
  rerender_existing_results.py     Rebuilds graphs from existing JSON reports
  run_benchmark_suite.py           Legacy traditional-vs-federated RF dashboard
```

## Install Dependencies

From the repo root:

```powershell
C:\Code\flower\Virt311\Scripts\python.exe -m pip install -r requirements.txt
```

`xgboost` is required only when running the optional XGBoost benchmark.

## Recreate Results

Run the complete benchmark set with XGBoost included:

```powershell
C:\Code\flower\Virt311\Scripts\python.exe run_all_results.py --include-xgboost
```

Outputs are written to `benchmark_outputs_latest/`:

- `all_results.json`: combined configuration, summaries, and per-round reports
- `summary.md`: compact metric table and notes
- `model_comparison.png`: comparison across model families
- `federated_rf_rounds.png`: federated RF round metrics
- `federated_boosted_rounds.png`: sklearn boosted-tree round metrics
- `federated_xgboost_rounds.png`: XGBoost round metrics, when enabled
- subdirectories containing each model's `benchmark_report.json`

## Rerender Graphs Without Training

If JSON reports already exist, rebuild the graphs without rerunning models:

```powershell
C:\Code\flower\Virt311\Scripts\python rerender_existing_results.py --output-dir benchmark_outputs_latest
```

Use this after graph style changes or when you need fresh PNGs from saved
results.

## Useful Focused Runs

Run only the cyclic sklearn boosted-tree benchmark:

```powershell
C:\Code\flower\Virt311\Scripts\python.exe Federated\soh_federated\boosted_simulate.py --data-path C:\Code\flower\cleaned_dataset\data --metadata-path C:\Code\flower\cleaned_dataset\metadata.csv --output-dir benchmark_outputs_latest\federated_boosted_25 --num-clients 5 --num-rounds 25 --local-estimators 5 --partition-strategy by_battery
```

Run only the cyclic XGBoost benchmark:

```powershell
C:\Code\flower\Virt311\Scripts\python.exe Federated\soh_federated\xgboost_simulate.py --data-path C:\Code\flower\cleaned_dataset\data --metadata-path C:\Code\flower\cleaned_dataset\metadata.csv --output-dir benchmark_outputs_latest\federated_xgboost --num-clients 5 --num-rounds 25 --local-rounds 5 --partition-strategy by_battery
```

## Interpreting Federated Results

`by_battery` partitioning is intentionally difficult. Each client owns different
batteries, so the local data distributions are non-IID. A weak federated score
under `by_battery` does not necessarily mean the model cannot learn; it often
means the global model is struggling to generalize across heterogeneous battery
families.

For diagnosis, compare:

```powershell
C:\Code\flower\Virt311\Scripts\python.exe run_all_results.py --include-xgboost --partition-strategy iid
C:\Code\flower\Virt311\Scripts\python.exe run_all_results.py --include-xgboost --partition-strategy dirichlet --dirichlet-alpha 5
C:\Code\flower\Virt311\Scripts\python.exe run_all_results.py --include-xgboost --partition-strategy by_battery
```

If IID performs well and `by_battery` performs poorly, the main issue is client
heterogeneity rather than the feature extractor alone.
=======
# NASA_battery_federated_learning
ML model trained utilizing the Flower AI Federated Learning Framework

Steps to run:
- Clone repo onto your work environment
  - git clone git@github.com:mikeyags1016/NASA_battery_federated_learning.git
- Set up virtual environment for project:
  - python3.11 -m venv venv
  - .\venv\Scripts\activate or source venv/bin/activate
- Inside project: python -m pip install --upgrade pip
  - pip install -r requirements.txt

Benchmark code:
- python .\run_benchmark_suite.py --data-path ..\cleaned_dataset\data --metadata-path ..\nasa-battery-dataset\cleaned_dataset\metadata.csv --num-satellites 5 --fed-rounds 3 --traditional-estimators 200 --federated-estimators 40 --fed-max-depth 10 --fed-min-samples-leaf 2 --fed-max-features sqrt --output-dir \benchmark_outputs_improved
>>>>>>> main
