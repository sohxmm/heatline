"""
Builds the official KJSCE course submission directory structure:
MiniProject<ROLL>/
  ├── Report/
  └── Code/
Usage:
    python scripts/package_submission.py --roll 12321218
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
EXCLUDE_FILES = {"PROJECT_SPEC.md", "DECISIONS.md", "walkthru.md", "WALKTHROUGH.md"}


def package_submission(roll_number: str, out_parent: Path = PROJECT_ROOT.parent) -> Path:
    dest_root = out_parent / f"MiniProject{roll_number}"
    code_dir = dest_root / "Code"
    report_dir = dest_root / "Report"

    if dest_root.exists():
        shutil.rmtree(dest_root)

    code_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    for item in PROJECT_ROOT.iterdir():
        if item.name in EXCLUDE_DIRS or item.name in EXCLUDE_FILES or item.name.startswith("MiniProject"):
            continue
        if item.suffix == ".zip":
            continue
        target = code_dir / item.name
        if item.is_dir():
            shutil.copytree(
                item,
                target,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache", "*verification*.md"),
            )
        else:
            shutil.copy2(item, target)

    print(f"Successfully packaged submission at: {dest_root}")
    return dest_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package KJSCE Soft Computing Mini Project Submission.")
    parser.add_argument("--roll", type=str, default="12321218", help="Team leader roll number.")
    args = parser.parse_args()
    package_submission(args.roll)
