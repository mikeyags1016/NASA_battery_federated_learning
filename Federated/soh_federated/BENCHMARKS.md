# Federated SOH Benchmark Workflows

This directory contains the current federated SOH benchmark implementations.
All workflows use the shared data and metric utilities in `sohfed/task.py` and
`sohfed/benchmarks.py`.

## Shared Data Pipeline

`sohfed/task.py` builds one row per discharge cycle.

Inputs:

- discharge CSV files from `cleaned_dataset/data`
- `metadata.csv`

Target:

- `SOH = Capacity / max_positive_capacity_for_that_battery`
- rows with missing or zero capacity are removed

Features:

- voltage summary statistics
- voltage percentiles and start/end shape information
- voltage-time area
- current and absolute-current statistics
- temperature statistics
- discharge duration and slope features

Partition strategies:

- `iid`: random equal split of cycles
- `dirichlet`: heterogeneous probabilistic split by battery ID
- `by_battery`: each client owns whole batteries; this is the hardest and most
  realistic split

## Federated Random Forest

Entry point:

```powershell
python Federated\soh_federated\simulate.py
```

The Random Forest path is a one-shot federated ensemble baseline. Each client
trains a local `RandomForestRegressor`, sends the pickled forest to the server,
and the server averages predictions across client forests.

Important limitation:

- More rounds do not perform true optimization. Extra rounds add independently
  seeded local forests, but clients do not refine a shared set of weights.

Use this model mainly as a communication and privacy-preserving baseline.

## Cyclic sklearn Gradient Boosted Trees

Entry point:

```powershell
python Federated\soh_federated\boosted_simulate.py
```

This benchmark uses `GradientBoostingRegressor` with `warm_start=True`.
The global boosted model is passed client-to-client, and each round adds more
trees on one client's data.

Why it exists:

- It gives a dependency-light boosted-tree baseline.
- It creates a real round curve, unlike the Random Forest ensemble path.
- It is fast enough for local experimentation.

Main knobs:

- `--num-rounds`: total federated communication rounds
- `--local-estimators`: trees added by each selected client
- `--learning-rate`
- `--max-depth`
- `--partition-strategy`

## Cyclic XGBoost

Entry point:

```powershell
python Federated\soh_federated\xgboost_simulate.py
```

This implementation follows the Flower XGBoost pattern using the Flower
`FedXgbCyclic` strategy available in the installed Flower version. One client
updates the global booster each round, then the updated model is sent back
through Flower as raw model bytes.

Dependency:

```powershell
pip install xgboost>=2.0.0
```

Main knobs:

- `--num-rounds`: communication rounds
- `--local-rounds`: XGBoost boosting rounds per selected client
- `--eta`: learning rate
- `--max-depth`
- `--subsample`
- `--partition-strategy`

## Full Pipeline

From the repo root, run:

```powershell
python run_all_results.py --include-xgboost
```

This runs centralized RF, federated RF, cyclic sklearn boosting, and cyclic
XGBoost, then writes combined outputs to `benchmark_outputs_latest/`.

## Rerendering Existing Results

Graph changes do not require retraining. If JSON reports already exist:

```powershell
python rerender_existing_results.py --output-dir benchmark_outputs_latest
```

This regenerates comparison and round plots from saved JSON reports.

## Metrics

Each benchmark report contains:

- `final_global_mae`
- `final_global_rmse`
- `final_global_r2`
- `final_global_accuracy_1pct`
- upload/download/total communication in MB
- average round time
- average client CPU time
- average client peak memory
- per-round metrics in `rounds`

`accuracy_1pct` is the fraction of predictions within `0.01` absolute SOH.
