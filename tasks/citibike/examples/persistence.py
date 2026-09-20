"""Predict that current station availability persists for the next hour."""


class PersistenceRegressor:
    def predict_one(self, *, event_id: str, event: dict) -> float:
        return float(event["station_status"]["num_bikes_available"])


def build_model() -> PersistenceRegressor:
    return PersistenceRegressor()
