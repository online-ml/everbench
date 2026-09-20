# Everbench

https://everbench-production.up.railway.app

This is a platform to host live, never-ending benchmarks. The goal is to evaluate machine learning models on streaming data tasks, be it regression, classification, clustering, anomaly detection, etc.

Each task defines its sources, prediction target and metrics. Sources collect observations and resolve earlier targets; models predict first, then learn and update their metrics when the targets become available. Missing targets are recorded explicitly and excluded from learning and scoring. Completed observations are archived into weekly Parquet files, and backtests and autonomous research use the same delayed replay.

The [Citi Bike benchmark](tasks/citibike/README.md) polls every 15 minutes and predicts available bikes at each NYC station 30 minutes ahead, with one shared model across all stations. See [CONTRIBUTING.md](CONTRIBUTING.md) for development and upgrade instructions.
