from slim.evaluation.summarize import summarize


def test_rollout_summary_deduplicates_episode(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    name = "rollout_libero_goal_task3_episode7_success.txt"
    (first / name).write_text("success", encoding="utf-8")
    (second / name).write_text("duplicate", encoding="utf-8")
    (first / "rollout_libero_goal_task3_episode8_failure.txt").write_text(
        "failure", encoding="utf-8"
    )

    report = summarize(tmp_path)
    assert "libero_goal: 1/2 = 50.00%" in report
    assert "overall: 1/2 = 50.00%" in report
