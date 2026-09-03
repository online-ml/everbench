# everbench

https://everbench-production.up.railway.app

This is a platform to host live, never-ending benchmarks. The goal is to evaluate machine learning models on streaming data tasks, be it regression, classification, clustering, anomaly detection, etc.

Each task has a worker that collects events. Every model assigned to a task makes a prediction for each event. Another worker collects labels once the ground truth is available. This allows updating each model, and updating evaluation metrics. The system stores only what it has to, while events and labels are archived into Parquet files.
