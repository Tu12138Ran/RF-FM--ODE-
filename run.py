#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from flowlab.engine import data_baseline, evaluate, make_pairs, train
from flowlab.io_utils import Progress, atomic_json, environment_info, fashion_images


def validate(config):
    if not config.get("tasks"):
        raise ValueError("No tasks selected")
    for task, c in config["tasks"].items():
        if task not in ("toy", "fashion"):
            raise ValueError(f"Unknown task {task}")
        for key in ("steps", "extra_steps", "batch", "width", "pair_count",
                    "eval_samples", "eval_batch", "trace_samples", "probe_samples",
                    "save_every", "log_every"):
            if not isinstance(c[key], int) or c[key] < 1:
                raise ValueError(f"{task}.{key} must be a positive integer")
        if c["reference_max"] <= c["reference_start"] or c["reference_start"] < 1:
            raise ValueError("reference_max must exceed reference_start >= 1")
        if c.get("reference_consecutive", 2) < 1:
            raise ValueError("reference_consecutive must be >= 1")
        if c["reference_max"] < c["reference_start"] * 2**c.get("reference_consecutive", 2):
            raise ValueError("reference_max cannot accommodate the required consecutive checks")
        if not 2*c["reference_start"] <= c.get("reference_min",2*c["reference_start"]) <= c["reference_max"]:
            raise ValueError("reference_min must lie between 2*reference_start and reference_max")
        if c["reference_rtol"] <= 0 or c.get("reference_atol", 1e-4) < 0:
            raise ValueError("Reference tolerances must be positive/nonnegative")
        if c.get("baseline_repeats", 20) < 2:
            raise ValueError("baseline_repeats must be >=2")
        if c["trace_samples"] > c["eval_samples"] or c["probe_samples"] > c["eval_samples"]:
            raise ValueError("trace/probe samples must not exceed eval samples")
        if not c["seeds"] or len(c["seeds"]) != len(set(c["seeds"])):
            raise ValueError("Seeds must be a non-empty list without duplicates")
        if any(not isinstance(n, int) or n < 1 for n in c["nfes"]):
            raise ValueError("NFE must be positive integers")


def acquire_lock(path):
    handle = path.open("a+")
    if os.name == "posix":
        import fcntl
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process is using {path.parent}")
    else:
        import msvcrt
        handle.seek(0)
        handle.write(" ")
        handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError(f"Another process is using {path.parent}")
    return handle


def main():
    parser = argparse.ArgumentParser(description="Diffusion / linear FM / reflow path lab")
    parser.add_argument("--config", default="configs/round2.json")
    parser.add_argument("--out", default="runs/round2")
    parser.add_argument("--data", default="data")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--only", choices=("toy", "fashion"))
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.only:
        config["tasks"] = {args.only: config["tasks"][args.only]}
    validate(config)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run bash setup.sh; inspect nvidia-smi. "
                           "Use --device cpu only for smoke/toy tests.")
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    # Same precision settings for every method; reproducible on the same environment.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.get("training_tf32", True)
        torch.backends.cudnn.allow_tf32 = config.get("training_tf32", True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(out / ".run.lock")
    saved_cfg = out / "config.json"
    if saved_cfg.exists():
        if json.loads(saved_cfg.read_text(encoding="utf-8")) != config:
            raise ValueError("Config differs from existing run. Use a new --out directory.")
    else:
        atomic_json(saved_cfg, config)
    Path(out / "pid").write_text(str(os.getpid()), encoding="ascii")
    atomic_json(out / "environment.json", dict(environment_info(),
                training_tf32=config.get("training_tf32", True),
                evaluation_precision="fp32_no_tf32", round=2))
    source_files = sorted(Path("flowlab").glob("*.py")) + [Path("run.py")]
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    source_record = out / "source_hashes.json"
    if source_record.exists():
        previous = json.loads(source_record.read_text(encoding="utf-8"))
        if previous != source_hashes:
            raise ValueError("Source code differs from this run. Use a new --out directory "
                             "to avoid mixing experimental implementations.")
    else:
        atomic_json(source_record, source_hashes)
    with (out / "pip_freeze.txt").open("w", encoding="utf-8") as f:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f,
                       stderr=subprocess.DEVNULL, check=False)
    jobs = sum(len(c["seeds"]) * (10 if t == "toy" else 9)
               for t, c in config["tasks"].items()) + 1
    progress = Progress(out, jobs)
    index = 0
    def stage(name):
        nonlocal index
        index += 1
        progress.update(job_index=index, phase=name, current=0, total=0,
                        loss=None, seconds_per_update=None, stage_eta_seconds=None)
    def stop_handler(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, stop_handler)
    try:
        for task, cfg in config["tasks"].items():
            images = None
            if task == "fashion":
                if config.get("synthetic_fashion"):
                    generator = torch.Generator().manual_seed(0)
                    images = torch.randint(0, 256, (64, 1, 28, 28),
                                           dtype=torch.uint8, generator=generator)
                    progress.event("warning", message="Synthetic smoke-test images, NOT Fashion-MNIST.")
                else:
                    images = fashion_images(args.data, "train", progress)
                # Entire uint8 dataset is ~47 MB; avoid worker overhead.
                images = images.to(device)
            for seed in cfg["seeds"]:
                if task == "toy":
                    stage(f"toy/{seed}/data_baseline")
                    data_baseline(seed, cfg, out, device, progress)
                paths = {}
                for method in ("diffusion", "linear"):
                    stage(f"{task}/{seed}/{method}/train")
                    paths[method] = train(task, method, seed, cfg, out, device,
                                          progress, images=images)
                    stage(f"{task}/{seed}/{method}/evaluate")
                    evaluate(task, method, seed, cfg, out, paths[method], device, progress)
                stage(f"{task}/{seed}/teacher_pairs")
                pair_path = make_pairs(task, seed, cfg, out, paths["linear"], device, progress)
                for method in ("linear_extra", "reflow"):
                    stage(f"{task}/{seed}/{method}/train")
                    checkpoint = train(task, method, seed, cfg, out, device, progress,
                                       images=images, parent=paths["linear"],
                                       pair_path=pair_path if method == "reflow" else None)
                    stage(f"{task}/{seed}/{method}/evaluate")
                    evaluate(task, method, seed, cfg, out, checkpoint, device, progress)
        stage("report")
        progress.update(stage="Building figures and report", current=0, total=1)
        from flowlab.reporting import build_report
        quality = build_report(out)
        progress.update(current=1, total=1)
        result_status = "completed_with_warnings" if quality["failed_reference_count"] else "completed"
        progress.event("complete", report=str(out / "report" / "report.html"),
                       reference_checks=quality, status=result_status)
        progress.finish(result_status)
    except KeyboardInterrupt:
        progress.event("interrupted", message="Rerun the same command to resume saved checkpoints.")
        progress.finish("interrupted")
        return 130
    except Exception:
        error = traceback.format_exc()
        (out / "error.txt").write_text(error, encoding="utf-8")
        progress.finish("failed", error=error)
        print(error, file=sys.stderr, flush=True)
        return 1
    finally:
        lock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
