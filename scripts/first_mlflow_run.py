"""
Standalone MLflow demo — the "Hello World" of experiment tracking.

Run this once to verify MLflow is working before running the full train.py pipeline.
It logs a simple sine wave experiment so you can see runs appear in the MLflow UI.
"""

import math
import mlflow

# Point at our local MLflow server (started via docker-compose)
mlflow.set_tracking_uri('http://localhost:5000')

# Experiments group related runs — like folders for your ML experiments
mlflow.set_experiment('hello-mlflow')

print('Starting first MLflow run...')

# mlflow.start_run() creates a new run and returns a context manager
# Everything logged inside the `with` block belongs to this run
with mlflow.start_run(run_name='sine-wave-demo'):

    # Log a parameter — any hyperparameter or config value worth tracking
    amplitude = 1.0
    frequency = 2.0
    mlflow.log_param('amplitude', amplitude)
    mlflow.log_param('frequency', frequency)

    # Log metrics at each step — MLflow records the full history so you can plot them
    for step in range(20):
        x = step * 0.1
        y = amplitude * math.sin(frequency * x)   # simple sine wave
        mlflow.log_metric('sine_value', y, step=step)

    # Log an artifact — any file you want to store alongside this run
    # Here we create a tiny text summary and attach it
    summary = (
        f'Sine wave demo\n'
        f'  amplitude = {amplitude}\n'
        f'  frequency = {frequency}\n'
        f'  steps     = 20\n'
    )
    with open('sine_summary.txt', 'w') as f:
        f.write(summary)
    mlflow.log_artifact('sine_summary.txt')  # uploads the file to the run's artifact store

    print('Run complete! Open http://localhost:5000 to see your experiment.')
    print('Look for experiment "hello-mlflow" → run "sine-wave-demo".')
