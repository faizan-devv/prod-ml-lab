"""Log a single training run with your train_and_log() - Lab 02, Step 2."""

from fraudlab.baseline import NAIVE_FEATURES, load_dataset, split_dataset
from fraudlab.experiments import setup_mlflow, train_and_log

setup_mlflow()
train, test = split_dataset(load_dataset(), "temporal")
run_id = train_and_log({"learning_rate": 0.1}, train, test, NAIVE_FEATURES, run_name="try-one-run")
print(f"\nrun id: {run_id}\nopen http://localhost:5000 -> experiment 'fraud-detection'")
