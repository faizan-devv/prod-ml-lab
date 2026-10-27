#!/usr/bin/env python3
"""Lab progress + solutions manager (stdlib only, so it runs before `uv sync`).

    python3 scripts/lab.py status [--brief]     progress overview (also run on session start)
    python3 scripts/lab.py start 04             make lab 04 the current lab
    python3 scripts/lab.py step 04 3            record that you're on step 3 of lab 04
    python3 scripts/lab.py check 04             run lab 04's checks; mark complete if green
    python3 scripts/lab.py solution 04 [--yes]  copy lab 04's reference solution in (backs up yours)
    python3 scripts/lab.py diff 04              show how your files differ from the solution
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABS_DIR = ROOT / "labs"
SOLUTIONS_DIR = ROOT / "solutions"
STATE_FILE = ROOT / ".lab" / "progress.json"
BACKUP_DIR = ROOT / ".lab" / "backups"


# --------------------------------------------------------------------------- state
def labs() -> list[tuple[str, str, Path]]:
    """[(id, title, dir)] for every labs/NN-name/ folder, in order."""
    out = []
    for d in sorted(LABS_DIR.iterdir()):
        m = re.match(r"(\d\d)-", d.name)
        if not (d.is_dir() and m):
            continue
        readme = d / "README.md"
        title = d.name
        if readme.exists():
            first = readme.read_text().splitlines()[0]
            title = re.sub(r"^#\s*(Lab \d+:\s*)?", "", first)
        out.append((m.group(1), title, d))
    return out


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"current_lab": "00", "current_step": 1, "labs": {}}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


def lab_dir(lab_id: str) -> Path:
    for i, _, d in labs():
        if i == lab_id:
            return d
    sys.exit(f"unknown lab {lab_id!r} (expected one of {[i for i, _, _ in labs()]})")


def norm(lab_id: str) -> str:
    return f"{int(lab_id):02d}"


# --------------------------------------------------------------------------- commands
def cmd_status(brief: bool = False) -> None:
    state = load_state()
    cur = state.get("current_lab", "00")
    done = [i for i, _, _ in labs() if state["labs"].get(i, {}).get("status") == "complete"]
    if brief:
        title = next((t for i, t, _ in labs() if i == cur), "?")
        print(f"[prod-ml-lab] {len(done)}/{len(labs())} labs complete. "
              f"Current: lab {cur} - {title} (step {state.get('current_step', 1)}). "
              f"Lab guide: {lab_dir(cur).relative_to(ROOT)}/README.md. "
              "Commands: /lab-start /lab-hint /lab-check /lab-quiz /lab-status")
        return
    print(f"\nProduction ML Lab - progress ({len(done)}/{len(labs())} complete)\n")
    for i, title, _ in labs():
        info = state["labs"].get(i, {})
        status = info.get("status", "not started")
        mark = {"complete": "[x]", "in_progress": "[>]"}.get(status, "[ ]")
        extra = f"  (completed {info['completed_at'][:10]})" if status == "complete" else ""
        if i == cur and status != "complete":
            extra = f"  <- you are here (step {state.get('current_step', 1)})"
        print(f"  {mark} {i}  {title}{extra}")
    print()


def cmd_start(lab_id: str) -> None:
    lab_id = norm(lab_id)
    lab_dir(lab_id)
    state = load_state()
    state["current_lab"], state["current_step"] = lab_id, 1
    entry = state["labs"].setdefault(lab_id, {})
    if entry.get("status") != "complete":
        entry["status"] = "in_progress"
        entry.setdefault("started_at", datetime.now().isoformat(timespec="seconds"))
    save_state(state)
    cmd_status(brief=True)


def cmd_step(lab_id: str, step: str) -> None:
    state = load_state()
    state["current_lab"], state["current_step"] = norm(lab_id), int(step)
    save_state(state)
    cmd_status(brief=True)


def cmd_check(lab_id: str, extra: list[str]) -> int:
    lab_id = norm(lab_id)
    tests = sorted((ROOT / "checks").glob(f"test_lab{lab_id}_*.py"))
    if not tests:
        sys.exit(f"no checks found for lab {lab_id}")
    cmd = ["uv", "run", "--quiet", "pytest", *map(str, tests), *extra]
    print("$", " ".join(cmd[3:]), flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    state = load_state()
    entry = state["labs"].setdefault(lab_id, {})
    entry["last_check"] = datetime.now().isoformat(timespec="seconds")
    entry["last_check_passed"] = rc == 0
    if rc == 0:
        entry["status"] = "complete"
        entry.setdefault("completed_at", datetime.now().isoformat(timespec="seconds"))
        print(f"\nLab {lab_id}: all checks passed.")
        ids = [i for i, _, _ in labs()]
        nxt = ids[ids.index(lab_id) + 1] if ids.index(lab_id) + 1 < len(ids) else None
        if nxt and state.get("current_lab") == lab_id:
            print(f"Next up: lab {nxt}. Run /lab-start {nxt} when you're ready.")
    else:
        entry.setdefault("status", "in_progress")
        print(f"\nLab {lab_id}: some checks failed - read the messages above (or ask /lab-hint).")
    save_state(state)
    return rc


def _solution_files(lab_id: str) -> list[Path]:
    src = SOLUTIONS_DIR / f"lab{lab_id}"
    if not src.exists():
        sys.exit(f"lab {lab_id} has no reference solution files (it's a guided/ops lab).")
    return sorted(p for p in src.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def cmd_diff(lab_id: str) -> None:
    lab_id = norm(lab_id)
    base = SOLUTIONS_DIR / f"lab{lab_id}"
    for sol in _solution_files(lab_id):
        rel = sol.relative_to(base)
        mine = ROOT / rel
        a = mine.read_text().splitlines(keepends=True) if mine.exists() else []
        b = sol.read_text().splitlines(keepends=True)
        sys.stdout.writelines(difflib.unified_diff(a, b, f"yours/{rel}", f"solution/{rel}"))


def cmd_solution(lab_id: str, yes: bool) -> None:
    lab_id = norm(lab_id)
    base = SOLUTIONS_DIR / f"lab{lab_id}"
    files = _solution_files(lab_id)
    print(f"Lab {lab_id} reference solution will overwrite:")
    for f in files:
        print("  ", f.relative_to(base))
    if not yes and input("Your current versions are backed up first. Continue? [y/N] ").lower() != "y":
        print("aborted")
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for f in files:
        rel = f.relative_to(base)
        dest = ROOT / rel
        if dest.exists():
            backup = BACKUP_DIR / f"lab{lab_id}-{stamp}" / rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, backup)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
    state = load_state()
    state["labs"].setdefault(lab_id, {})["used_solution"] = True
    save_state(state)
    print(f"done. Backups (if any) in .lab/backups/lab{lab_id}-{stamp}/")


def main(argv: list[str]) -> int:
    os.chdir(ROOT)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(__doc__)
        return 0
    cmd, args = argv[0], argv[1:]
    if cmd == "status":
        cmd_status(brief="--brief" in args)
    elif cmd == "start" and args:
        cmd_start(args[0])
    elif cmd == "step" and len(args) == 2:
        cmd_step(*args)
    elif cmd == "check" and args:
        return cmd_check(args[0], args[1:])
    elif cmd == "solution" and args:
        cmd_solution(args[0], yes="--yes" in args)
    elif cmd == "diff" and args:
        cmd_diff(args[0])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
