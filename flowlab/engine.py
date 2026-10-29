import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .core import (amp_context, analytic_toy_velocity, make_model, mode_metrics,
                   path_sample, sample_toy, sliced_w2, solve, straightness,
                   strict_eval_precision)
from .io_utils import (atomic_json, atomic_torch, load_checkpoint, restore_rng,
                       rng_state, seed_all)


METHODS = ("diffusion", "linear", "linear_extra", "reflow")


class ReferenceConvergenceError(RuntimeError):
    pass


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def model_from_checkpoint(path, task, cfg, device):
    checkpoint = load_checkpoint(path)
    model = make_model(task, cfg).to(device)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    return model


def train(task, method, seed, cfg, run_dir, device, progress, images=None,
          parent=None, pair_path=None):
    folder = Path(run_dir) / task / f"seed_{seed}" / method
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "checkpoint.pt"
    total = cfg["extra_steps"] if method in ("linear_extra", "reflow") else cfg["steps"]
    progress.update(stage=f"{task}/{seed}/{method}: training", current=0, total=total,
                    stage_eta_seconds=None, loss=None, seconds_per_update=None)
    if path.exists():
        ckpt = load_checkpoint(path)
        if ckpt["step"] >= total:
            progress.event("skip_complete_training", task=task, seed=seed, method=method)
            return path
    else:
        ckpt = None

    # Identical initialization for A/B; identical branch initialization for extra/reflow.
    seed_all(seed)
    model = make_model(task, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    step, offset, cumulative_seconds = 0, 0, 0.0
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step, offset = ckpt["step"], ckpt["offset"]
        cumulative_seconds = ckpt["training_seconds"]
        restore_rng(ckpt["rng"])
    elif parent:
        base = load_checkpoint(parent)
        model.load_state_dict(base["model"])
        ema.load_state_dict(base["ema"])
        optimizer.load_state_dict(base["optimizer"])
        offset = base["step"] + base["offset"]
        restore_rng(base["rng"])

    # Move optimizer state tensors after loading CPU checkpoint.
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and key != "step":
                state[key] = value.to(device)
    pairs = load_checkpoint(pair_path) if pair_path else None
    use_amp = bool(cfg.get("amp", True) and task == "fashion"
                   and device.type == "cuda" and torch.cuda.is_bf16_supported())
    start = time.perf_counter()
    start_step = step
    last_log = start
    recent_losses = []
    log = folder / "training.jsonl"
    model.train()

    def save_checkpoint():
        sync(device)
        elapsed = time.perf_counter() - start
        atomic_torch(path, {
            "model": model.state_dict(), "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(), "step": step, "offset": offset,
            "rng": rng_state(), "training_seconds": cumulative_seconds + elapsed,
            "task": task, "method": method, "seed": seed, "config": cfg,
        })

    for step in range(step + 1, total + 1):
        batch = cfg["batch"]
        if pairs is not None:
            ids = torch.randint(len(pairs["noise"]), (batch,))
            noise = pairs["noise"][ids].to(device=device, dtype=torch.float32)
            data = pairs["target"][ids].to(device=device, dtype=torch.float32)
        else:
            if task == "toy":
                data = sample_toy(batch, device)
            else:
                ids = torch.randint(len(images), (batch,), device=images.device)
                data = images[ids].to(device=device, dtype=torch.float32) / 127.5 - 1
                # No augmentation for Fashion-MNIST; same distribution in all groups.
            noise = torch.randn_like(data)
        t = torch.rand(batch, device=device)
        kind = "diffusion" if method == "diffusion" else "linear"
        xt, target = path_sample(noise, data, t, kind)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, use_amp):
            prediction = model(xt, t)
        loss = (prediction.float() - target.float()).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at {task}/{method}/{step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        global_step = offset + step
        lr = cfg["lr"] * min(1.0, global_step / max(1, cfg["warmup"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        # Warm-start EMA prevents a 5k-step run being dominated by initial weights.
        decay = min(cfg["ema_decay"], (1 + global_step) / (10 + global_step))
        with torch.no_grad():
            for ep, p in zip(ema.parameters(), model.parameters()):
                ep.lerp_(p, 1 - decay)
        recent_losses.append(loss.detach())

        now = time.perf_counter()
        if step % cfg["log_every"] == 0 or step == total or now - last_log > 10:
            sync(device)
            now = time.perf_counter()
            elapsed = now - start
            sec_step = elapsed / max(1, step - start_step)
            mean_loss = float(torch.stack(recent_losses).mean())
            recent_losses.clear()
            record = dict(step=step, global_step=global_step, loss=mean_loss,
                          lr=lr, seconds_per_update=sec_step,
                          training_seconds=cumulative_seconds + elapsed)
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            progress.update(current=step, total=total, loss=mean_loss,
                            seconds_per_update=sec_step,
                            stage_eta_seconds=(total - step) * sec_step)
            progress.event("train", task=task, seed=seed, method=method, **record)
            last_log = now
        if step % cfg["save_every"] == 0 or step == total:
            save_checkpoint()
    atomic_json(folder / "training_summary.json", {
        "steps": total, "offset": offset, "training_seconds": cumulative_seconds
        + time.perf_counter() - start, "amp_bf16": use_amp,
        "parameters": sum(p.numel() for p in model.parameters()),
        "ema": "min(configured_decay, (1+global_step)/(10+global_step))",
    })
    return path


@torch.no_grad()
@strict_eval_precision()
def choose_reference_steps(model, probe, cfg, progress, label):
    """Two passing doublings plus RK4 agreement; record every check, never fake a pass."""
    started = time.perf_counter()
    steps = cfg["reference_start"]
    max_steps = cfg["reference_max"]
    needed = cfg.get("reference_consecutive", 2)
    atol = cfg.get("reference_atol", 1e-4)
    minimum = cfg.get("reference_min", 2 * steps)
    consecutive, history = 0, []
    progress.update(stage=label, current=0, total=max_steps, stage_eta_seconds=None)
    low = solve(model, probe, steps, "heun",
                progress=lambda i, n: progress.update(current=i, total=n))
    while True:
        fine = min(2 * steps, max_steps)
        if fine <= steps:
            raise ValueError("reference_max must exceed reference_start")
        progress.update(stage=label + f": Heun {fine} steps", current=0, total=fine)
        high = solve(model, probe, fine, "heun",
                     progress=lambda i, n: progress.update(current=i, total=n))
        rmse = float((high - low).square().mean().sqrt())
        scale = max(float(high.square().mean().sqrt()), 0.1)
        relative = rmse / scale
        threshold = atol + cfg["reference_rtol"] * scale
        doubling_passed = rmse <= threshold
        consecutive = consecutive + 1 if doubling_passed else 0
        cross = None
        eligible = consecutive >= needed and fine >= minimum
        if eligible or fine == max_steps:
            progress.update(stage=label + ": RK4 cross-check", current=0, total=fine // 2)
            cross_steps = max(1, fine // 2)
            rk = solve(model, probe, cross_steps, "rk4",
                       progress=lambda i, n: progress.update(current=i, total=n))
            cross_rmse = float((rk - high).square().mean().sqrt())
            cross = dict(method="rk4", steps=cross_steps, nfe=4 * cross_steps,
                         rmse=cross_rmse, relative_rmse=cross_rmse / scale,
                         passed=cross_rmse <= threshold)
        converged = eligible and cross is not None and cross["passed"]
        sample_rmse = (high - low).flatten(1).square().mean(1).sqrt()
        entry = dict(coarse_steps=steps, steps=fine, doubling_rmse=rmse,
                     relative_rmse=relative, threshold=threshold,
                     sample_rmse_p95=float(torch.quantile(sample_rmse, .95)),
                     sample_rmse_max=float(sample_rmse.max()),
                     doubling_passed=doubling_passed, consecutive_passes=consecutive,
                     cross_check=cross)
        history.append(entry)
        diagnostic = dict(steps=fine, nfe=2 * fine, doubling_rmse=rmse,
                          relative_rmse=relative, converged=converged,
                          tolerance=cfg["reference_rtol"],
                          absolute_tolerance=atol, threshold=threshold,
                          check_samples=len(probe), cross_check=cross,
                          required_consecutive=needed, history=list(history),
                          precision="fp32_no_tf32", method="heun",
                          check_seconds=time.perf_counter() - started)
        progress.event("reference_check", label=label, steps=fine, nfe=2*fine,
                       relative_rmse=relative, doubling_passed=doubling_passed,
                       consecutive_passes=consecutive,
                       rk4_passed=cross["passed"] if cross else None,
                       converged=converged)
        if converged or fine == max_steps:
            if not converged:
                progress.event("warning", message=label + ": reference did not pass tolerance")
            return fine, diagnostic
        steps, low = fine, high


@torch.no_grad()
@strict_eval_precision()
def data_baseline(seed, cfg, run_dir, device, progress):
    """Finite-sample SW2 baseline against exactly the same target/directions as evaluation."""
    path = Path(run_dir) / "toy" / f"seed_{seed}" / "data_baseline.json"
    if path.exists():
        return
    n = cfg["eval_samples"]
    gen = torch.Generator(device=device).manual_seed(seed + 200000)
    torch.randn(n, 2, generator=gen, device=device)  # evaluation's initial noises
    truth = sample_toy(n, device, gen)
    directions = torch.randn(128, 2, generator=gen, device=device)
    directions /= directions.norm(dim=1, keepdim=True)
    other_gen = torch.Generator(device=device).manual_seed(seed + 500000)
    values = []
    count = cfg.get("baseline_repeats", 20)
    for i in range(count):
        progress.update(stage=f"toy/{seed}: data-data SW2 baseline", current=i, total=count,
                        stage_eta_seconds=None)
        values.append(sliced_w2(sample_toy(n, device, other_gen), truth, directions))
    atomic_json(path, dict(seed=seed, samples=n, repeats=count, values=values,
                          mean=float(np.mean(values)), std=float(np.std(values, ddof=1)),
                          description="Independent target draws vs fixed evaluation target; not a model CI",
                          target_valid_fraction=1 - math.exp(-4.5)))
    progress.update(current=count, total=count)


def shape_for(task):
    return (2,) if task == "toy" else (1, 28, 28)


@torch.no_grad()
@strict_eval_precision()
def make_pairs(task, seed, cfg, run_dir, checkpoint, device, progress):
    folder = Path(run_dir) / task / f"seed_{seed}" / "pairs"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "pairs.pt"
    if output.exists():
        return output
    started = time.perf_counter()
    model = model_from_checkpoint(checkpoint, task, cfg, device)
    gen = torch.Generator(device=device).manual_seed(seed + 123000)
    probe = torch.randn((cfg["probe_samples"],) + shape_for(task), generator=gen, device=device)
    steps_file = folder / "teacher_solver.json"
    if steps_file.exists():
        diag = json.loads(steps_file.read_text(encoding="utf-8"))
        steps = diag["steps"]
    else:
        steps, diag = choose_reference_steps(model, probe, cfg, progress,
                                             f"{task}/{seed}: teacher convergence check")
        atomic_json(steps_file, diag)
    if not diag["converged"]:
        raise ReferenceConvergenceError(
            f"Teacher reference did not converge at {steps} Heun steps. "
            f"See {steps_file}. Model checkpoints are saved; no unchecked teacher pairs are generated.")
    total = cfg["pair_count"]
    batch = cfg["eval_batch"]
    chunks = []
    created = 0
    for index, start in enumerate(range(0, total, batch)):
        count = min(batch, total - start)
        part = folder / f"chunk_{index:05d}.pt"
        if not part.exists():
            # Per-chunk seeds make resume independent of how many chunks were skipped.
            generator = torch.Generator(device=device).manual_seed(seed + 130000 + index)
            z = torch.randn((count,) + shape_for(task), generator=generator, device=device)
            progress.update(stage=f"{task}/{seed}: generating reflow pairs",
                            current=start, total=total, stage_eta_seconds=None)
            sync(device)
            chunk_started = time.perf_counter()
            y = solve(model, z, steps, "heun")
            sync(device)
            # Keep FP32: quantization is not an additional experimental factor.
            atomic_torch(part, {"noise": z.cpu(), "target": y.cpu(),
                                "generation_seconds": time.perf_counter() - chunk_started})
            created += count
        chunks.append(load_checkpoint(part))
        elapsed = time.perf_counter() - started
        progress.update(current=start + count, total=total,
                        stage_eta_seconds=(total - start - count) * elapsed / max(created, 1)
                        if created else None)
    atomic_torch(output, {
        "noise": torch.cat([c["noise"] for c in chunks]),
        "target": torch.cat([c["target"] for c in chunks]),
        "teacher_solver": diag, "seed": seed,
    })
    atomic_json(folder / "generation_summary.json", {
        "pair_count": total, "teacher_solver": diag,
        "this_session_seconds": time.perf_counter() - started,
        "new_pairs_this_session": created,
        "summed_chunk_generation_seconds": sum(c.get("generation_seconds", 0) for c in chunks),
        "note": "Session time includes checking/loading; resumed earlier-session time is not included.",
    })
    return output


@torch.no_grad()
@strict_eval_precision()
def evaluate(task, method, seed, cfg, run_dir, checkpoint, device, progress):
    folder = Path(run_dir) / task / f"seed_{seed}" / method
    result_path = folder / "evaluation.json"
    if result_path.exists() and (folder / "samples.npz").exists():
        return
    model = model_from_checkpoint(checkpoint, task, cfg, device)
    gen = torch.Generator(device=device).manual_seed(seed + 200000)
    # Same initial noise/target draws/directions for every method at a given seed.
    z = torch.randn((cfg["eval_samples"],) + shape_for(task), generator=gen, device=device)
    if task == "toy":
        truth = sample_toy(len(z), device, gen)
        directions = torch.randn(128, 2, generator=gen, device=device)
        directions /= directions.norm(dim=1, keepdim=True)
    else:
        truth, directions = None, None
    steps, diag = choose_reference_steps(
        model, z[:cfg["probe_samples"]], cfg, progress,
        f"{task}/{seed}/{method}: reference convergence check")
    atomic_json(folder / "reference_check.json", diag)
    start_time = time.perf_counter()
    pieces = []
    for start in range(0, len(z), cfg["eval_batch"]):
        batch = z[start:start + cfg["eval_batch"]]
        progress.update(stage=f"{task}/{seed}/{method}: reference samples",
                        current=start, total=len(z), stage_eta_seconds=None)
        pieces.append(solve(model, batch, steps, "heun"))
        progress.update(current=min(len(z), start + len(batch)), total=len(z))
    ref = torch.cat(pieces)
    _, trace = solve(model, z[:cfg["trace_samples"]], steps, "heun", trace=True)
    progress.update(stage=f"{task}/{seed}/{method}: trajectory metrics",
                    current=0, total=1, stage_eta_seconds=None)
    s = straightness(model, trace)
    displacement = float((trace[-1] - trace[0]).square().mean())
    normalization = float(trace[0].square().mean() + trace[-1].square().mean()) + 1e-8
    common = dict(straightness=s, normalized_straightness=s / normalization,
                  displacement_mse=displacement,
                  reference_steps=steps, reference_nfe=2 * steps,
                  reference_converged=diag["converged"],
                  reference_precision="fp32_no_tf32",
                  reference_relative_rmse=diag["relative_rmse"],
                  reference_cross_relative_rmse=diag["cross_check"]["relative_rmse"]
                  if diag["cross_check"] else None)
    rows = []
    saved = {"initial": z[:64].cpu().numpy(),
             "reference": ref[:64].cpu().numpy(),
             "trace": trace[:, :32].cpu().numpy()}
    if task == "toy":
        saved["target"] = truth.cpu().numpy()
        saved["reference"] = ref.cpu().numpy()
        saved["initial"] = z.cpu().numpy()

    for nfe in cfg["nfes"]:
        output = []
        for start in range(0, len(z), cfg["eval_batch"]):
            batch = z[start:start + cfg["eval_batch"]]
            progress.update(stage=f"{task}/{seed}/{method}: Euler NFE={nfe}",
                            current=start, total=len(z), stage_eta_seconds=None)
            output.append(solve(model, batch, nfe, "euler"))
            progress.update(current=min(len(z), start + len(batch)))
        samples = torch.cat(output)
        error = float((samples - ref).square().mean())
        row = dict(task=task, seed=seed, method=method, sampler="euler",
                   nfe=nfe, solver_mse=error, **common)
        if task == "toy":
            row.update(sw2=sliced_w2(samples, truth, directions), **mode_metrics(samples))
        rows.append(row)
        saved[f"nfe_{nfe}"] = samples.cpu().numpy() if task == "toy" else samples[:64].cpu().numpy()
    reference_row = dict(task=task, seed=seed, method=method, sampler="heun_reference",
                         nfe=2 * steps, solver_mse=0.0, **common)
    if task == "toy":
        reference_row.update(sw2=sliced_w2(ref, truth, directions), **mode_metrics(ref))
    rows.append(reference_row)

    # For independent endpoint paths, evaluate field error against an exact GMM oracle.
    # Reflow follows a learned coupling and must not be compared with this wrong oracle.
    field = None
    if task == "toy" and method != "reflow":
        times, errors = [], []
        kind = "diffusion" if method == "diffusion" else "linear"
        for time_value in np.linspace(0.01, 0.99, 20):
            t = torch.full((len(z),), float(time_value), device=device)
            xt, _ = path_sample(z, truth, t, kind)
            oracle = analytic_toy_velocity(xt, t, kind)
            errors.append(float((model(xt, t) - oracle).square().mean()))
            times.append(float(time_value))
        field = dict(times=times, mse=errors)
    # Write arrays first; evaluation.json is the completion marker.
    np.savez_compressed(folder / "samples.npz", **saved)
    atomic_json(result_path, dict(rows=rows, reference_check=diag,
                                  field_error=field, sample_count=len(z),
                                  evaluation_seconds=time.perf_counter() - start_time))
    progress.event("evaluation_complete", task=task, seed=seed, method=method,
                   reference_converged=diag["converged"],
                   normalized_straightness=common["normalized_straightness"])
