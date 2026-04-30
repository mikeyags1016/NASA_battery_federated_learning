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
