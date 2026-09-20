"""One 8 MiB online tree for all stations, with normalized lag features."""

from river import tree

from tasks.citibike.examples.shared import SharedRegressor


def build_model() -> SharedRegressor:
    return SharedRegressor(
        model=tree.HoeffdingTreeRegressor(
            leaf_prediction="mean", max_depth=12, max_size=8, memory_estimate_period=10_000
        )
    )
