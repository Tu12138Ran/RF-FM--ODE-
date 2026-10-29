import math
import tempfile
import unittest
from pathlib import Path

import torch

from flowlab.core import (SmallUNet, analytic_toy_velocity, path_sample, sliced_w2,
                          solve, straightness, toy_centers)
from flowlab.engine import train
from flowlab.io_utils import load_checkpoint


class QuietProgress:
    def __init__(self, interrupt_step=None):
        self.interrupt_step = interrupt_step

    def update(self, **kwargs):
        pass

    def event(self, kind, **kwargs):
        if kind == "train" and kwargs["step"] == self.interrupt_step:
            raise KeyboardInterrupt()


class NumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_endpoint_and_derivative(self):
        torch.manual_seed(2)
        noise, data = torch.randn(12, 2, dtype=torch.float64), torch.randn(12, 2, dtype=torch.float64)
        for kind in ("diffusion", "linear"):
            x0, _ = path_sample(noise, data, torch.zeros(12, dtype=torch.float64), kind)
            x1, _ = path_sample(noise, data, torch.ones(12, dtype=torch.float64), kind)
            torch.testing.assert_close(x0, noise)
            torch.testing.assert_close(x1, data)
            t = torch.linspace(.05, .95, 12, dtype=torch.float64)
            _, velocity = path_sample(noise, data, t, kind)
            plus, _ = path_sample(noise, data, t + 1e-5, kind)
            minus, _ = path_sample(noise, data, t - 1e-5, kind)
            torch.testing.assert_close((plus - minus) / 2e-5, velocity, rtol=1e-7, atol=1e-7)

    def test_velocity_is_not_diffusion_v_without_scale(self):
        eps, x = torch.tensor([[1., 2.]]), torch.tensor([[3., 4.]])
        _, u = path_sample(eps, x, torch.zeros(1), "diffusion")
        torch.testing.assert_close(u, math.pi / 2 * x)

    def test_nfe_accounting_and_exact_constant_velocity(self):
        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, x, t):
                self.calls += 1
                return torch.ones_like(x) * 2

        z = torch.zeros(8, 2)
        for method, nfe in (("euler", 5), ("heun", 10)):
            model = Constant()
            y = solve(model, z, 5, method)
            self.assertEqual(model.calls, nfe)
            torch.testing.assert_close(y, z + 2)
        model = Constant()
        _, trace = solve(model, z, 8, "heun", trace=True)
        self.assertLess(straightness(model, trace), 1e-10)

    def test_gaussian_counterexample(self):
        class Gaussian(torch.nn.Module):
            def forward(self, x, t):
                return ((2 * t - 1) / ((1 - t)**2 + t**2))[:, None] * x
        z = torch.randn(32, 2)
        one = solve(Gaussian(), z, 1)
        torch.testing.assert_close(one, torch.zeros_like(z))
        ref = solve(Gaussian(), z, 256, "heun")
        torch.testing.assert_close(ref, z, atol=1e-4, rtol=1e-4)

    def test_sw2_and_unet_shape(self):
        x = torch.randn(64, 2)
        self.assertEqual(sliced_w2(x, x.flip(0), torch.eye(2)), 0.0)
        model = SmallUNet(8)
        images = torch.randn(2, 1, 28, 28)
        y = model(images, torch.rand(2))
        self.assertEqual(y.shape, images.shape)
        y.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_analytic_gmm_field_satisfies_continuity(self):
        # Finite differences validate the Gaussian-mixture oracle independently.
        x = torch.tensor([[.2, -.3], [1.1, .7], [-1.5, .2]], dtype=torch.float64)
        centers = toy_centers().double()
        dt, dx = 1e-5, 1e-5
        for kind in ("linear", "diffusion"):
            def density(points, t):
                if kind == "linear":
                    a, b = 1 - t, t
                else:
                    a, b = math.cos(t * math.pi / 2), math.sin(t * math.pi / 2)
                var = a*a + .15**2 * b*b
                r = points[:, None, :] - b * centers
                return ((-r.square().sum(-1) / (2 * var)).exp() / (2 * math.pi * var)).mean(1)
            t = .43
            dpdt = (density(x, t + dt) - density(x, t - dt)) / (2 * dt)
            div = torch.zeros(len(x), dtype=torch.float64)
            for k in range(2):
                shift = torch.zeros_like(x)
                shift[:, k] = dx
                def flux(points):
                    ts = torch.full((len(x),), t, dtype=torch.float64)
                    return density(points, t) * analytic_toy_velocity(points, ts, kind)[:, k]
                div += (flux(x + shift) - flux(x - shift)) / (2 * dx)
            torch.testing.assert_close(dpdt + div, torch.zeros_like(div), atol=2e-7, rtol=1e-5)

    def test_resume_matches_uninterrupted_training(self):
        cfg = dict(width=16, steps=6, extra_steps=2, batch=16, lr=.001,
                   warmup=2, ema_decay=.99, amp=False, log_every=1, save_every=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = train("toy", "linear", 7, cfg, root / "full", torch.device("cpu"), QuietProgress())
            with self.assertRaises(KeyboardInterrupt):
                train("toy", "linear", 7, cfg, root / "resume", torch.device("cpu"),
                      QuietProgress(interrupt_step=4))
            b = train("toy", "linear", 7, cfg, root / "resume", torch.device("cpu"), QuietProgress())
            ca, cb = load_checkpoint(a), load_checkpoint(b)
            self.assertEqual(ca["step"], cb["step"])
            for component in ("model", "ema"):
                for key in ca[component]:
                    torch.testing.assert_close(ca[component][key], cb[component][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
