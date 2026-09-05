from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable



def default_classification_path() -> Path:
    explicit = os.environ.get("LIBERO_PLUS_CLASSIFICATION")
    if explicit:
        return Path(explicit)
    libero_plus_home = os.environ.get("SLIM_PLUS_HOME")
    if libero_plus_home:
        return Path(libero_plus_home) / "libero/libero/benchmark/task_classification.json"
    return Path("libero/libero/benchmark/task_classification.json")


DEFAULT_CLASSIFICATION = default_classification_path()
DEFAULT_SUITES = ("libero_10", "libero_spatial", "libero_object", "libero_goal")
EXPECTED_ROLLOUTS = {
    "libero_10": 2519,
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
}
LEADERBOARD_COLUMNS = (
    ("Camera", "Camera Viewpoints"),
    ("Robot", "Robot Initial States"),
    ("Language", "Language Instructions"),
    ("Light", "Light Conditions"),
    ("Background", "Background Textures"),
    ("Noise", "Sensor Noise"),
    ("Layout", "Objects Layout"),
)
ROLLOUT_RE = re.compile(
    r"^rollout_(?P<suite>libero_(?:10|spatial|object|goal))_"
    r"task(?P<task_id>\d+)_episode(?P<episode>\d+)_(?P<result>success|failure)\.txt$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LIBERO-plus rollout markers by perturbation category.")
    parser.add_argument(
        "--results-root",
        nargs="+",
        type=Path,
        required=True,
        help="One or more result roots to scan, e.g. <run_dir>/results_libero_plus.",
    )
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES), choices=list(DEFAULT_SUITES))
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output for per-rollout records.")
    parser.add_argument("--output", type=Path, default=None, help="Optional text summary output.")
    parser.add_argument("--model-name", type=str, default="", help="Model name used in the leaderboard row.")
    return parser.parse_args()


def load_classification(path: Path) -> dict[str, dict[int, dict]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[int, dict]] = {}
    for suite, items in raw.items():
        by_id: dict[int, dict] = {}
        for item in items:
            try:
                by_id[int(item["id"]) - 1] = item
            except (KeyError, TypeError, ValueError):
                continue
        result[suite] = by_id
    return result


def scan_rollouts(
    roots: Iterable[Path],
    suites: set[str],
    classification: dict[str, dict[int, dict]],
) -> list[dict]:
    records: dict[tuple[str, int, int], dict] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.txt"):
            match = ROLLOUT_RE.match(path.name)
            if match is None:
                continue
            suite = match.group("suite")
            if suite not in suites:
                continue
            task_id = int(match.group("task_id"))
            episode = int(match.group("episode"))
            stat = path.stat()
            key = (suite, task_id, episode)
            old = records.get(key)
            if old is not None and old["mtime"] >= stat.st_mtime:
                continue
            meta = classification.get(suite, {}).get(task_id, {})
            records[key] = {
                "suite": suite,
                "task_id": task_id,
                "episode": episode,
                "success": match.group("result") == "success",
                "result": match.group("result"),
                "category": meta.get("category", "UNKNOWN"),
                "difficulty": str(meta.get("difficulty_level", "UNKNOWN")),
                "name": meta.get("name", ""),
                "path": str(path),
                "mtime": stat.st_mtime,
                "mtime_text": dt.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            }
    return sorted(records.values(), key=lambda row: (row["suite"], row["task_id"], row["episode"]))


def aggregate(rows: Iterable[dict], key_name: str) -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        key = str(row[key_name])
        counts[key][0] += int(row["success"])
        counts[key][1] += 1
    return {key: (val[0], val[1]) for key, val in counts.items()}


def format_rate(success: int, total: int) -> str:
    if total == 0:
        return "NA"
    return f"{100.0 * success / total:6.2f}% ({success}/{total})"


def print_table(title: str, counts: dict[str, tuple[int, int]]) -> None:
    print(f"\n=== {title} ===")
    if not counts:
        print("NA")
        return
    width = max(len(key) for key in counts)
    for key, (success, total) in sorted(counts.items(), key=lambda item: (item[1][0] / max(item[1][1], 1), item[0])):
        print(f"{key:<{width}}  {format_rate(success, total)}")


def maybe_write_csv(path: Path | None, rows: list[dict]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["suite", "task_id", "episode", "success", "result", "category", "difficulty", "name", "path", "mtime_text"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def leaderboard_rate(counts: dict[str, tuple[int, int]], category: str) -> str:
    success, total = counts.get(category, (0, 0))
    if total == 0:
        return "NA"
    return f"{100.0 * success / total:.2f}"


def render_summary(
    rows: list[dict],
    model_name: str = "model",
    expected_rollouts: int | None = None,
) -> str:
    lines: list[str] = []

    def add_table(title: str, counts: dict[str, tuple[int, int]]) -> None:
        lines.append("")
        lines.append(f"=== {title} ===")
        if not counts:
            lines.append("NA")
            return
        width = max(len(key) for key in counts)
        for key, (success, total) in sorted(
            counts.items(),
            key=lambda item: (item[1][0] / max(item[1][1], 1), item[0]),
        ):
            lines.append(f"{key:<{width}}  {format_rate(success, total)}")

    total = len(rows)
    success = sum(int(row["success"]) for row in rows)
    expected = expected_rollouts if expected_rollouts is not None else sum(EXPECTED_ROLLOUTS.values())
    lines.append("LIBERO-Plus evaluation summary (from episode markers)")
    lines.append(f"model: {model_name}")
    lines.append(f"coverage: {total}/{expected} ({100.0 * total / expected:.2f}%)")
    lines.append(f"rollouts: {format_rate(success, total)}")
    add_table("By suite", aggregate(rows, "suite"))
    category_counts = aggregate(rows, "category")
    add_table("By category", category_counts)
    add_table("By difficulty", aggregate(rows, "difficulty"))
    lines.append("")
    lines.append("=== Leaderboard row ===")
    headers = ["Model"] + [name for name, _ in LEADERBOARD_COLUMNS] + ["Total"]
    values = [model_name] + [leaderboard_rate(category_counts, category) for _, category in LEADERBOARD_COLUMNS]
    values.append(f"{100.0 * success / total:.2f}" if total else "NA")
    lines.append("\t".join(headers))
    lines.append("\t".join(values))
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    classification = load_classification(args.classification)
    rows = scan_rollouts(args.results_root, set(args.suites), classification)
    model_name = args.model_name or (args.results_root[0].name if args.results_root else "model")
    expected_rollouts = sum(EXPECTED_ROLLOUTS[suite] for suite in args.suites)
    summary = render_summary(rows, model_name=model_name, expected_rollouts=expected_rollouts)
    print(summary, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary, encoding="utf-8")
        print(f"\nText summary written to: {args.output}", file=sys.stderr)
    maybe_write_csv(args.csv, rows)
    if args.csv is not None:
        print(f"\nCSV written to: {args.csv}")


if __name__ == "__main__":
    main()
