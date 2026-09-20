# Citi Bike

Predict available bikes at each NYC station 30 minutes ahead. Each regressor is one shared model trained across all stations. MAE and RMSE are measured in bikes; MAE determines ranking.

The collector polls [Citi Bike's official GBFS feed](https://citibikenyc.com/system-data) every 15 minutes and refreshes station metadata hourly. It includes NYC and Bronx stations, excluding New Jersey, closed stations and reports older than two minutes. Events preserve the raw station fields and actual collection time.

The target is the first fresh snapshot collected between +30 and +32 minutes, normally two polls ahead. Its station report may be up to two minutes old. Each poll both saves new observations and resolves matching earlier forecasts in one transaction. Pending forecasts survive restarts. A maintenance loop checks deadlines every minute; after the matching window and one extra minute for ingestion, it records any missing target as unavailable. These observations remain in archives but do not train or score models. Zero bikes is a valid target.

Four examples are included: persistence, scaled linear regression, a Hoeffding tree and a five-tree adaptive forest. Learned models predict a correction to current availability using location, capacity, bike/dock counts and the target's New York local time. River's tree memory budgets are 8 MiB for the standalone tree and 2 MiB per forest tree, with checks every 10,000 weighted observations. Predictions are nonnegative. Capacity is not a hard bound because valet stations can exceed it.

With the usual database settings configured, start the API:

```sh
uv run everbench api
```

In another terminal, export and upload the models, then start collection:

```sh
uv run python tasks/citibike/examples/regressors.py --output-dir /tmp/citibike-models --api-url http://127.0.0.1:8000 --owner max
uv run everbench debug worker tasks/citibike/task.py
```

Uploads require `EVERBENCH_API_KEY` and `EVERBENCH_MODEL_SIGNING_KEY`. Omit `--api-url` and `--owner` to export pickles only. Existing model IDs cannot be replaced with different artifacts.

`worker-all` discovers this task automatically. On Railway, set `EVERBENCH_TASK_NAMES="wiki-liftwing citibike"` on the web and worker services to run both benchmarks in the existing worker process. Model registration is separate from deploying the collector.
