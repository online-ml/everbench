"""One online linear regressor for all stations, with normalized lag features."""

from river import linear_model, optim

from tasks.citibike.examples.shared import SharedRegressor


def build_model() -> SharedRegressor:
    return SharedRegressor(model=linear_model.LinearRegression(optimizer=optim.SGD(0.005), l2=0.001, clip_gradient=1))
