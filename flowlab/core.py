import math
from contextlib import contextmanager, nullcontext

import torch
from torch import nn
import torch.nn.functional as F


def expand_time(t, x):
    return t.reshape((-1,) + (1,) * (x.ndim - 1))


def path_sample(noise, data, t, kind):
    """Generation time: 0=noise, 1=data. Targets are dX_t/dt, not DDPM v."""
    s = expand_time(t, noise)
    if kind == "diffusion":
        angle = s * (math.pi / 2)
        alpha, sigma = angle.sin(), angle.cos()
        xt = alpha * data + sigma * noise
        target = (math.pi / 2) * (sigma * data - alpha * noise)
    elif kind == "linear":
        xt = s * data + (1 - s) * noise
        target = data - noise
    else:
        raise ValueError(kind)
    return xt, target


class TimeEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.register_buffer("freq", torch.exp(
            torch.linspace(0, math.log(1000), dim // 2)), persistent=True)

    def forward(self, t):
        angles = t.float()[:, None] * self.freq[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class ToyMLP(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.time = TimeEmbedding(32)
        self.net = nn.Sequential(
            nn.Linear(34, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 2))

    def forward(self, x, t):
        return self.net(torch.cat((x, self.time(t)), dim=1))


def norm(c):
    return nn.GroupNorm(math.gcd(c, 8), c)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, temb=128):
        super().__init__()
        self.n1 = norm(cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.time = nn.Linear(temb, cout)
        self.n2 = norm(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.time(t)[:, :, None, None]
        return self.skip(x) + self.c2(F.silu(self.n2(h)))


class SmallUNet(nn.Module):
    """28 -> 14 -> 7, exact-size upsampling; no attention/dropout."""
    def __init__(self, base=32):
        super().__init__()
        self.time = nn.Sequential(TimeEmbedding(64), nn.Linear(64, 128),
                                  nn.SiLU(), nn.Linear(128, 128))
        self.stem = nn.Conv2d(1, base, 3, padding=1)
        self.r0 = ResBlock(base, base)
        self.d0 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.r1 = ResBlock(base * 2, base * 2)
        self.d1 = nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1)
        self.mid = ResBlock(base * 2, base * 2)
        self.u1 = ResBlock(base * 4, base * 2)
        self.u0 = ResBlock(base * 3, base)
        self.out = nn.Sequential(norm(base), nn.SiLU(),
                                 nn.Conv2d(base, 1, 3, padding=1))

    def forward(self, x, t):
        te = self.time(t)
        h0 = self.r0(self.stem(x), te)
        h1 = self.r1(self.d0(h0), te)
        h = self.mid(self.d1(h1), te)
        h = self.u1(torch.cat((F.interpolate(h, size=h1.shape[-2:],
                                            mode="nearest"), h1), 1), te)
        h = self.u0(torch.cat((F.interpolate(h, size=h0.shape[-2:],
                                            mode="nearest"), h0), 1), te)
        return self.out(h)


def make_model(task, cfg):
    return ToyMLP(cfg["width"]) if task == "toy" else SmallUNet(cfg["width"])


def amp_context(device, enabled):
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if enabled and device.type == "cuda" else nullcontext())


@contextmanager
def strict_eval_precision():
    """FP32 evaluation without TF32; restore training settings even on failure."""
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


@torch.no_grad()
def solve(model, z, steps, method="euler", trace=False, progress=None):
    """FP32 states. Actual NFE: Euler=N, Heun=2N, classical RK4=4N."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if method not in ("euler", "heun", "rk4"):
        raise ValueError(method)
    x = z.float().clone()
    dt = 1.0 / steps
    states = [x.clone()] if trace else None
    for i in range(steps):
        t = torch.full((x.shape[0],), i / steps, device=x.device)
        v = model(x, t).float()
        pred = x + dt * v
        if method == "heun":
            tn = torch.full_like(t, (i + 1) / steps)
            pred = x + dt * (v + model(pred, tn).float()) / 2
        elif method == "rk4":
            th = torch.full_like(t, (i + 0.5) / steps)
            tn = torch.full_like(t, (i + 1) / steps)
            k2 = model(x + dt * v / 2, th).float()
            k3 = model(x + dt * k2 / 2, th).float()
            k4 = model(x + dt * k3, tn).float()
            pred = x + dt * (v + 2 * k2 + 2 * k3 + k4) / 6
        x = pred
        if trace:
            states.append(x.clone())
        if progress:
            progress(i + 1, steps)
    if not torch.isfinite(x).all():
        raise FloatingPointError("Non-finite ODE output; inspect training/solver.")
    return (x, torch.stack(states)) if trace else x


@torch.no_grad()
def straightness(model, states):
    """Trapezoidal integral of per-coordinate velocity/chord discrepancy."""
    n = states.shape[0] - 1
    chord = states[-1] - states[0]
    total = torch.zeros((), device=states.device)
    for i in range(n + 1):
        t = torch.full((states.shape[1],), i / n, device=states.device)
        err = (model(states[i], t).float() - chord).square().mean()
        total += err * (0.5 if i in (0, n) else 1.0) / n
    return float(total)


def toy_centers(device="cpu"):
    a = torch.arange(8, device=device) * (2 * math.pi / 8)
    return 2 * torch.stack((a.cos(), a.sin()), dim=1)


def sample_toy(n, device, generator=None):
    centers = toy_centers(device)
    idx = torch.randint(8, (n,), device=device, generator=generator)
    return centers[idx] + 0.15 * torch.randn(
        n, 2, device=device, generator=generator)


@torch.no_grad()
def sliced_w2(x, y, projections):
    """Empirical SW2, equal sample counts, fixed unit projection directions."""
    if x.shape != y.shape:
        raise ValueError("SW2 requires equal shape in this implementation")
    a = (x @ projections.T).sort(dim=0).values
    b = (y @ projections.T).sort(dim=0).values
    return float((a - b).square().mean().sqrt())


@torch.no_grad()
def mode_metrics(x):
    distances = torch.cdist(x, toy_centers(x.device))
    nearest, labels = distances.min(dim=1)
    valid = nearest <= 0.45  # 3 component standard deviations
    counts = torch.bincount(labels[valid], minlength=8)
    minimum = max(1, math.ceil(len(x) * 0.01))
    coverage = int((counts >= minimum).sum())
    return {
        "mode_coverage": coverage,
        "valid_fraction": float(valid.float().mean()),
        "mode_counts": counts.cpu().tolist(),
    }


@torch.no_grad()
def analytic_toy_velocity(x, t, kind):
    """Exact marginal velocity for independent Gaussian -> 8-Gaussian mixture."""
    if kind == "diffusion":
        a = torch.cos(t * math.pi / 2)
        b = torch.sin(t * math.pi / 2)
        da = -math.pi / 2 * b
        db = math.pi / 2 * a
    elif kind == "linear":
        a, b = 1 - t, t
        da, db = -torch.ones_like(t), torch.ones_like(t)
    else:
        raise ValueError("No analytic formula for learned reflow coupling")
    centers = toy_centers(x.device)
    var = a.square() + (0.15 * b).square()
    mean = b[:, None, None] * centers[None, :, :]
    residual = x[:, None, :] - mean
    weights = (-residual.square().sum(-1) / (2 * var[:, None])).softmax(1)
    factor = (a * da + 0.15**2 * b * db) / var
    conditional = db[:, None, None] * centers[None, :, :] + factor[:, None, None] * residual
    return (weights[:, :, None] * conditional).sum(1)
