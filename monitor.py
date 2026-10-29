#!/usr/bin/env python3
"""Standard-library-only terminal monitor; safe to run in a second SSH session."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def duration(seconds):
    if seconds is None:
        return "measuring..."
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def render(out):
    path = out / "status.json"
    if not path.exists():
        return "Waiting for status.json. Check console.log if startup takes too long.", False
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "Waiting for a valid status snapshot...", False
    now = time.time()
    total, current = state.get("total", 0), state.get("current", 0)
    ratio = min(1, current / total) if total else 0
    bar = "#" * int(30 * ratio) + "-" * (30 - int(30 * ratio))
    lines = [
        "FLOW PATH LAB ROUND 2 | live progress",
        f"Status: {state['status']}    PID: {state['pid']}",
        f"Jobs: {state['job_index']}/{state['jobs_total']} (job count, NOT wall-time percentage)",
        f"Stage: {state['stage']}",
        f"[{bar}] {current}/{total}  {100 * ratio:.1f}%",
        f"Stage ETA: {duration(state.get('stage_eta_seconds'))}",
        f"Session elapsed: {duration(state.get('ended_at', now) - state['started_at'])}",
        f"Heartbeat age: {now - state['updated_at']:.0f}s | last progress: {now - state['progress_at']:.0f}s",
    ]
    if state.get("loss") is not None:
        lines.append(f"Latest training loss: {state['loss']:.6g} (not comparable across paths)")
    if state.get("seconds_per_update"):
        lines.append(f"Measured seconds/update: {state['seconds_per_update']:.4f}")
    if now - state["updated_at"] > 30 and state["status"] == "running":
        lines.append("WARNING: stale heartbeat. Process may have exited or been killed; inspect console.log.")
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=3)
        if gpu.returncode == 0:
            lines.extend(["GPU: name, utilization, memory used/total, temperature", gpu.stdout.strip()])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    if state.get("error"):
        lines.extend(["ERROR (last lines):", "\n".join(state["error"].splitlines()[-8:])])
    if state["status"] == "completed_with_warnings":
        lines.append("FINISHED, but some reference checks failed. Read report/quality_checks.json.")
    lines += [f"Log: {out / 'console.log'}", f"Report after completion: {out / 'report' / 'report.html'}",
              "Ctrl+C closes this monitor only." ]
    return "\n".join(lines), state["status"] in ("completed", "completed_with_warnings", "failed", "interrupted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/round2")
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    try:
        while True:
            text, done = render(Path(args.out))
            if os.isatty(1) and not args.once:
                print("\033[2J\033[H", end="")
            print(text, flush=True)
            if args.once or done:
                break
            time.sleep(max(1, args.interval))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
