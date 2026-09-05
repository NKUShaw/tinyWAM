import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_research_names_are_confined_to_compatibility_docs():
    forbidden = ("starVLA", "XMoT", "DINOXMoT")
    allowed = {
        ROOT / "slim/compat/legacy.py",
        ROOT / "docs/migration.md",
        ROOT / "NOTICE",
        ROOT / "reproducibility/source_manifest.json",
    }
    violations = []
    for path in (ROOT / "slim").rglob("*.py"):
        if path in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                violations.append((str(path.relative_to(ROOT)), token))
    assert not violations


def test_tracked_public_files_have_no_internal_defaults():
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode().split("\0")
    forbidden = (
        "/" + "share/project/",
        "kzz-fudan" + "-university",
        "calvin_" + "xmot_jepa",
    )
    text_suffixes = {
        ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"
    }
    violations = []
    for relative in tracked:
        if not relative:
            continue
        path = ROOT / relative
        if path.suffix not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                violations.append((relative, token))
    assert not violations
