"""Summarize episode-level rollout markers."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


MARKER = re.compile(
    r"^rollout_(?P<suite>libero_(?:10|spatial|object|goal))_"
    r"task(?P<task>\d+)_episode(?P<episode>\d+)_(?P<result>success|failure)\.txt$"
)


def summarize(root: Path) -> str:
    episodes: dict[tuple[str, int, int], bool] = {}
    for path in root.rglob("rollout_*.txt"):
        match = MARKER.match(path.name)
        if match is None:
            continue
        key = (
            match.group("suite"),
            int(match.group("task")),
            int(match.group("episode")),
        )
        episodes[key] = match.group("result") == "success"

    by_suite: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for (suite, _, _), success in episodes.items():
        by_suite[suite][0] += int(success)
        by_suite[suite][1] += 1

    lines = []
    for suite in ("libero_10", "libero_spatial", "libero_object", "libero_goal"):
        success, total = by_suite.get(suite, (0, 0))
        rate = 100.0 * success / total if total else 0.0
        lines.append(f"{suite}: {success}/{total} = {rate:.2f}%")
    success = sum(int(value) for value in episodes.values())
    total = len(episodes)
    rate = 100.0 * success / total if total else 0.0
    lines.append(f"overall: {success}/{total} = {rate:.2f}%")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.results_root)
    print(report, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
