# NASA Battery Federated Learning

This project benchmarks State-of-Health (SOH) regression for the NASA battery
dataset using centralized and federated learning.

The main result pipeline compares:

- centralized Random Forest
- federated Random Forest
- federated Gradient Boosted Trees
- federated XGBoost

This guide assumes Windows PowerShell.

## 1. Install Required Software

Install these first:

- Git: https://git-scm.com/downloads
- Python 3.11: https://www.python.org/downloads/release/python-311/
- A GitHub account with SSH access configured
- A Kaggle account

Confirm Git and Python are available:

```powershell
git --version
py -3.11 --version
```

If `py -3.11 --version` fails, install Python 3.11 and make sure it is added to
PATH.

## 2. Configure GitHub SSH

If you already use SSH with GitHub, skip this step.

Create an SSH key:

```powershell
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Press Enter to accept the default file location. Then print the public key:

```powershell
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
```

Copy that output and add it to GitHub:

```text
GitHub -> Settings -> SSH and GPG keys -> New SSH key
```

Test the connection:

```powershell
ssh -T git@github.com
```

GitHub should respond with a successful authentication message.

## 3. Create the Workspace Folder

Create and enter the workspace:

```powershell
mkdir C:\Code\flower
cd C:\Code\flower
```

## 4. Clone the Repository

Clone with SSH:

```powershell
git clone git@github.com:mikeyags1016/NASA_battery_federated_learning.git
cd NASA_battery_federated_learning
```

After cloning, the repo should look like this:

```text
C:\Code\flower\NASA_battery_federated_learning
  Federated\
  Traditional\
  README.md
  requirements.txt
  run_all_results.py
  run_benchmark_suite.py
  rerender_existing_results.py
```

## 5. Download the Dataset

Download the dataset from Kaggle:

```text
www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset
```

After downloading and extracting it, arrange the files so the project can find:

```text
C:\Code\flower\cleaned_dataset\metadata.csv
C:\Code\flower\cleaned_dataset\data\
```

The `data` folder should contain many numbered CSV files, for example:

```text
C:\Code\flower\cleaned_dataset\data\00001.csv
C:\Code\flower\cleaned_dataset\data\00002.csv
C:\Code\flower\cleaned_dataset\data\00003.csv
```

The final folder layout should be:

```text
C:\Code\flower
  cleaned_dataset\
    metadata.csv
    data\
      00001.csv
      00002.csv
      ...
  NASA_battery_federated_learning\
    Federated\
    Traditional\
    run_all_results.py
    requirements.txt
```

## 6. Create a Python 3.11 Virtual Environment

From the repo root:

```powershell
cd C:\Code\flower\NASA_battery_federated_learning
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Confirm Python 3.11 is active:

```powershell
python --version
```

You should see Python `3.11.x`.

## 7. Install Dependencies

Upgrade `pip`:

```powershell
python -m pip install --upgrade pip
```

Install project dependencies:

```powershell
python -m pip install -r requirements.txt
```

This installs Flower, scikit-learn, matplotlib, XGBoost, and the other required
libraries.

## 8. Run the Full Benchmark Pipeline

Run all models, including XGBoost:

```powershell
python run_all_results.py --include-xgboost
```

This may take a while because it trains several models and runs Flower
simulations.

Results are written to:

```text
benchmark_outputs_latest\
```

Important output files:

```text
benchmark_outputs_latest\summary.md
benchmark_outputs_latest\all_results.json
benchmark_outputs_latest\model_comparison.png
benchmark_outputs_latest\federated_rf_rounds.png
benchmark_outputs_latest\federated_boosted_rounds.png
benchmark_outputs_latest\federated_xgboost_rounds.png
```

## 9. View the Results

Open:

```text
benchmark_outputs_latest\summary.md
```

This file gives the compact metric comparison.

Also inspect:

```text
benchmark_outputs_latest\model_comparison.png
```

This chart compares the model families.

## 10. Rerender Graphs Without Retraining

If the JSON result files already exist and you only want to regenerate the
graphs:

```powershell
python rerender_existing_results.py --output-dir benchmark_outputs_latest
```

This does not retrain any models.

## Optional Runs

Run only the federated Gradient Boosted Trees benchmark:

```powershell
python Federated\soh_federated\boosted_simulate.py --data-path C:\Code\flower\cleaned_dataset\data --metadata-path C:\Code\flower\cleaned_dataset\metadata.csv --output-dir benchmark_outputs_latest\federated_boosted_25 --num-clients 5 --num-rounds 25 --local-estimators 5 --partition-strategy by_battery
```

Run only the federated XGBoost benchmark:

```powershell
python Federated\soh_federated\xgboost_simulate.py --data-path C:\Code\flower\cleaned_dataset\data --metadata-path C:\Code\flower\cleaned_dataset\metadata.csv --output-dir benchmark_outputs_latest\federated_xgboost --num-clients 5 --num-rounds 25 --local-rounds 5 --partition-strategy by_battery
```

Run the full pipeline with an easier IID split:

```powershell
python run_all_results.py --include-xgboost --partition-strategy iid
```

Run the full pipeline with moderate non-IID Dirichlet partitioning:

```powershell
python run_all_results.py --include-xgboost --partition-strategy dirichlet --dirichlet-alpha 5
```

## What the Main Scripts Do

```text
run_all_results.py
  Runs centralized RF, federated RF, federated sklearn boosting, and optional
  federated XGBoost.

rerender_existing_results.py
  Rebuilds charts from existing JSON files without retraining.

Federated\soh_federated\simulate.py
  Federated Random Forest baseline.

Federated\soh_federated\boosted_simulate.py
  Federated cyclic Gradient Boosted Trees.

Federated\soh_federated\xgboost_simulate.py
  Federated cyclic XGBoost using Flower's XGBoost strategy.
```

## Troubleshooting

If Python 3.11 is not found:

```powershell
py --list
```

Install Python 3.11 if it is missing.

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If the dataset is not found, confirm these paths exist:

```powershell
Test-Path C:\Code\flower\cleaned_dataset\metadata.csv
Test-Path C:\Code\flower\cleaned_dataset\data
```

If XGBoost is missing:

```powershell
python -m pip install xgboost
```

If graphs need to be regenerated but training already finished:

```powershell
python rerender_existing_results.py --output-dir benchmark_outputs_latest
```

## Notes

- The default `by_battery` split is intentionally hard because each client owns
  different batteries.
- Weak federated performance under `by_battery` often means the data is highly
  non-IID, not necessarily that the code failed.
- For more details on the federated benchmarks, see:

```text
Federated\soh_federated\BENCHMARKS.md
```
