from datetime import UTC, datetime, timedelta

from everbench.replay import ArchiveExample, replay


def test_delayed_replay_preserves_ties_and_does_not_learn_unavailable_targets() -> None:
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    trace = []

    def predict(*, event_id, event):
        trace.append(("predict", event_id))
        event["changed"] = True
        return 0

    def score(*, target, prediction):
        trace.append(("score", target))

    def learn(*, event_id, event, label):
        assert "changed" not in event
        trace.append(("learn", event_id))

    result = replay(
        observations=iter(
            [
                ArchiveExample(
                    event_id="first",
                    sequence=1,
                    payload={},
                    target=1,
                    available_at=origin,
                    resolved_at=origin + timedelta(minutes=30),
                ),
                ArchiveExample(
                    event_id="missing",
                    sequence=2,
                    payload={},
                    target=None,
                    available_at=origin + timedelta(minutes=15),
                    resolved_at=origin + timedelta(minutes=45),
                ),
                ArchiveExample(
                    event_id="at-target",
                    sequence=3,
                    payload={},
                    target=2,
                    available_at=origin + timedelta(minutes=30),
                    resolved_at=origin + timedelta(minutes=60),
                ),
                ArchiveExample(
                    event_id="after-target",
                    sequence=4,
                    payload={},
                    target=3,
                    available_at=origin + timedelta(minutes=45),
                    resolved_at=origin + timedelta(minutes=75),
                ),
            ]
        ),
        predict=predict,
        score=score,
        learn=learn,
    )

    assert result.predictions == 4
    assert result.labels == 3
    assert trace == [
        ("predict", "first"),
        ("predict", "missing"),
        ("score", 1),
        ("learn", "first"),
        ("predict", "at-target"),
        ("predict", "after-target"),
        ("score", 2),
        ("learn", "at-target"),
        ("score", 3),
        ("learn", "after-target"),
    ]
