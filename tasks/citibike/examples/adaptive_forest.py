"""One five-tree forest for all stations, with a 2 MiB budget per tree."""

from river import forest

from tasks.citibike.examples.shared import SharedRegressor


def build_model() -> SharedRegressor:
    return SharedRegressor(
        model=forest.ARFRegressor(
            n_models=5, leaf_prediction="mean", max_depth=12, max_size=2, memory_estimate_period=10_000, seed=42
        )
    )
