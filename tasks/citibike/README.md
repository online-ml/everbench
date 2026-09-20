# Citi Bike

Predict available bikes at each NYC station one hour ahead. Each regressor is one shared model trained across all stations. MAE and RMSE are measured in bikes; MAE determines ranking.

The collector polls [Citi Bike's official GBFS feed](https://citibikenyc.com/system-data) every 15 minutes and refreshes station metadata hourly. It includes NYC and Bronx stations, excluding New Jersey and closed stations. A feed more than five minutes old is skipped; individual station report ages do not filter otherwise valid stations. Events preserve the raw station fields and actual collection time.

The target is the first snapshot collected between +55 and +65 minutes, normally four polls ahead. The target is the station's reported available-bike count in that snapshot. Pending forecasts survive restarts. A maintenance loop checks deadlines every minute; after +66 minutes, it records any missing target as unavailable. These observations remain in archives but do not train or score models. Zero bikes is a valid target.

Each example has its own file: [persistence](examples/persistence.py), [linear regression](examples/linear_regression.py), [Hoeffding tree](examples/hoeffding_tree.py) and [adaptive forest](examples/adaptive_forest.py). Each exposes `build_model()`. The learned models share [feature extraction and target scaling](examples/shared.py): bike/dock counts and target changes are divided by station capacity, and predictions are converted back to bikes. Features include location, capacity, New York local time, 15/30/60-minute occupancy lags, recent changes and a rolling mean. Predictions are nonnegative; capacity is not a hard bound because valet stations can exceed it.

The collector retains only four compact snapshots and embeds each station's available history in its event. Delayed training and archive replay therefore see exactly the history available at forecast time. Missing lags use current occupancy with an explicit availability flag; history warms up again after a worker restart. River's tree memory budgets remain 8 MiB for the standalone tree and 2 MiB per forest tree, checked every 10,000 weighted observations. Examples contain model definitions only, with no export or upload CLI.

With the usual database settings configured, start the API:

```sh
uv run everbench api
```

In another terminal, start collection:

```sh
uv run everbench debug worker tasks/citibike/task.py
```

Register signed model artifacts through the regular API. Uploads require `EVERBENCH_API_KEY` and `EVERBENCH_MODEL_SIGNING_KEY`; existing model IDs cannot be replaced with different artifacts. When serializing imported examples, embed their local modules with `cloudpickle.register_pickle_by_value` so the artifact remains self-contained.

`worker-all` discovers this task automatically. On Railway, set `EVERBENCH_TASK_NAMES="wiki-liftwing citibike"` on the web and worker services to run both benchmarks in the existing worker process. Model registration is separate from deploying the collector.
