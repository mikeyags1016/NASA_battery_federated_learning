# Federated SOH Estimation with Flower

A complete federated learning system for **State-of-Health (SOH) estimation** of lithium-ion batteries, built with [Flower](https://flower.ai), scikit-learn, and PyTorch-free NumPy transport.

---

## Project Structure

```
soh_federated/
├── sohfed/
│   ├── __init__.py
│   ├── task.py          # Model, data loading, train/eval utilities
│   ├── client_app.py    # Flower ClientApp
│   ├── server_app.py    # Flower ServerApp + custom strategy
│   └── benchmarks.py   # Benchmark data structures & report
├── simulate.py          # Standalone simulation runner (recommended)
├── pyproject.toml       # Flower app + dependency config
└── README.md
```

---

## How It Works

### Federated Learning Design

Because `RandomForestRegressor` is not a gradient-based model, classical FedAvg
(which averages gradients) is replaced with **prediction-averaging**:

1. Each client trains its own `RandomForestRegressor` on its local discharge data.
2. The fitted forests are serialised (pickled → uint8 NumPy array) and sent to the server.
3. The server reconstructs a **`FederatedForest`** — an ensemble that averages predictions
   across all client forests.
4. The aggregated `FederatedForest` is sent back to clients for evaluation.

This is equivalent to training a large ensemble whose trees are distributed across clients.

### Data Partitioning

Discharge files listed in `metadata.csv` are split evenly across `N` clients.
Each client sees an independent 80/20 train-test split of its local partition.

---

## Benchmark Metrics Collected

| Metric                     | Where collected |
|----------------------------|-----------------|
| Data transmitted (KB/round)| Server — bytes received from clients + bytes sent back |
| Communication rounds       | Flower round counter |
| Client training time (s)   | `time.perf_counter()` inside `train()` |
| Client peak RAM (KB)       | `tracemalloc` inside `train()` |
| Global MAE                 | Server evaluates `FederatedForest` on a held-out test set |
| Global RMSE                | Same |
| Accuracy (<1 % abs error)  | Fraction of predictions within 0.01 absolute SOH error |
| Round wall time (s)        | Server measures end-to-end round duration |

All metrics are saved to `results/benchmark_report.json` and visualised in
`results/benchmark_dashboard.png`.

---

## Quickstart

### 1. Install dependencies

```bash
# Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install Flower + dependencies
pip install -e .
```

### 2. Run the simulation (recommended)

```bash
python simulate.py \
    --data-path     /path/to/cleaned_dataset/data \
    --metadata-path /path/to/cleaned_dataset/metadata.csv \
    --num-clients   5 \
    --num-rounds    3 \
    --n-estimators  100 \
    --output-dir    ./results
```

### 3. Run via Flower CLI (advanced)

First update `pyproject.toml` with your dataset paths, then:

```bash
# Edit data-base-path and metadata-path in pyproject.toml first
flwr run . --stream
```

Override config on the fly:
```bash
flwr run . --stream --run-config "num-server-rounds=5 n-estimators=200"
```

Scale to more clients:
```bash
flwr run . --stream --federation-config "num-supernodes=10"
```

---

## Output Files

After a run you'll find in `./results/`:

- **`benchmark_report.json`** — full per-round metrics + summary
- **`benchmark_dashboard.png`** — 8-panel dark-theme dashboard:
  - MAE per round
  - RMSE per round
  - Accuracy (<1 % error) per round
  - Data transmitted per round (KB)
  - Avg client training time (s)
  - Avg client peak memory (KB)
  - Round wall time (s)
  - Summary text panel

---

## Configuration Reference (`pyproject.toml`)

| Key                    | Default | Description                            |
|------------------------|---------|----------------------------------------|
| `data-base-path`       | —       | Directory containing discharge CSVs   |
| `metadata-path`        | —       | Path to `metadata.csv`                |
| `num-server-rounds`    | 3       | Number of FL rounds                    |
| `n-estimators`         | 100     | Trees per client RandomForest          |
| `test-size`            | 0.2     | Fraction held out for local testing   |
| `fraction-fit`         | 1.0     | Fraction of clients trained per round |
| `fraction-evaluate`    | 1.0     | Fraction of clients evaluated/round   |
| `benchmark-output`     | `benchmark_report.json` | Output file name         |

---

## Extending the System

- **Different model**: Replace `RandomForestRegressor` in `task.py` with any sklearn estimator; the serialisation helpers (`model_to_bytes` / `bytes_to_model`) work for any picklable object.
- **Gradient-based model**: Swap to a PyTorch MLP and use standard FedAvg weight averaging — the `ArrayRecord` transport layer supports this directly.
- **More features**: Add `V_range`, `dV/dt` statistics, or temperature features to `extract_voltage_features()` in `task.py`.
- **Non-IID splits**: Replace the equal-split partitioning in `load_data()` with e.g. `flwr-datasets` `DirichletPartitioner` for realistic heterogeneous data distributions.
