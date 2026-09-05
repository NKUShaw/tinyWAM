from pathlib import Path

from slim.evaluation.libero_plus.analyze import default_classification_path, render_summary


def test_default_classification_uses_libero_plus_home(monkeypatch):
    monkeypatch.delenv("LIBERO_PLUS_CLASSIFICATION", raising=False)
    monkeypatch.setenv("SLIM_PLUS_HOME", "/tmp/libero-plus")

    assert default_classification_path() == Path(
        "/tmp/libero-plus/libero/libero/benchmark/task_classification.json"
    )


def test_render_summary_contains_full_breakdown():
    rows = [
        {
            "suite": "libero_10",
            "success": True,
            "category": "Camera Viewpoints",
            "difficulty": "easy",
        },
        {
            "suite": "libero_goal",
            "success": False,
            "category": "Sensor Noise",
            "difficulty": "hard",
        },
    ]

    report = render_summary(rows, model_name="checkpoint")

    assert "LIBERO-Plus evaluation summary" in report
    assert "model: checkpoint" in report
    assert "coverage: 2/10030" in report
    assert "rollouts:  50.00% (1/2)" in report
    assert "=== By suite ===" in report
    assert "=== By category ===" in report
    assert "=== By difficulty ===" in report
    assert "=== Leaderboard row ===" in report

    suite_report = render_summary(rows[:1], model_name="checkpoint", expected_rollouts=2519)
    assert "coverage: 1/2519" in suite_report
