from app.metrics import LatencyRecorder, collect


def test_empty_recorder_has_no_statistics():
    snapshot = LatencyRecorder().snapshot()

    assert snapshot["total"] == 0
    assert snapshot["p50_ms"] is None
    assert snapshot["recent_ms"] == []


def test_percentiles_use_nearest_rank():
    recorder = LatencyRecorder()
    for ms in range(1, 101):
        recorder.record(ms)

    snapshot = recorder.snapshot()

    assert snapshot["p50_ms"] == 50
    assert snapshot["p95_ms"] == 95
    assert snapshot["max_ms"] == 100
    assert snapshot["last_ms"] == 100


def test_average_is_rounded_to_whole_milliseconds():
    recorder = LatencyRecorder()
    for ms in (10, 20, 33):
        recorder.record(ms)

    assert recorder.snapshot()["avg_ms"] == 21


def test_window_keeps_only_the_latest_samples_but_counts_everything():
    recorder = LatencyRecorder(size=3)
    for ms in (10, 20, 30, 40):
        recorder.record(ms)

    snapshot = recorder.snapshot()

    assert snapshot["total"] == 4
    assert snapshot["recent_ms"] == [20, 30, 40]


def test_per_minute_counts_recent_samples_only(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("app.metrics.time.time", lambda: clock[0])
    recorder = LatencyRecorder()
    recorder.record(5)
    clock[0] += 90
    recorder.record(5)
    recorder.record(5)

    assert recorder.snapshot()["per_minute"] == 2


def test_collect_without_a_router_reports_no_models():
    class Bare:
        pass

    report = collect(Bare())

    assert report["models"] == []
    assert report["performance"] is None
