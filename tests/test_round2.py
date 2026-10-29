import json
import tempfile
import tarfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from flowlab.core import solve, strict_eval_precision
from flowlab.engine import ReferenceConvergenceError, choose_reference_steps, make_pairs
from flowlab.reporting import aggregate_rows
from test_numerics import QuietProgress
from export_results import export_results


class Constant(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, t):
        self.calls += 1
        return torch.ones_like(x)


class RoundTwoTests(unittest.TestCase):
    def test_export_includes_diagnostics_excludes_weights(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root)/"run"
            for name in ("report/report.html", "toy/seed_0/data_baseline.json",
                         "toy/seed_0/pairs/teacher_solver.json",
                         "toy/seed_0/linear/reference_check.json",
                         "toy/seed_0/linear/checkpoint.pt", "toy/seed_0/pairs/pairs.pt"):
                p = out/name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("test", encoding="utf-8")
            result = export_results(out, Path(root)/"results.tar.gz")
            with tarfile.open(result["path"]) as archive:
                names = archive.getnames()
            self.assertEqual(len(names), 4)
            self.assertIn("toy/seed_0/linear/reference_check.json", names)
            self.assertIn("toy/seed_0/pairs/teacher_solver.json", names)
            self.assertFalse(any(n.endswith(".pt") for n in names))

    def test_rk4_nfe_and_convergence(self):
        model = Constant()
        z = torch.zeros(3, 2)
        torch.testing.assert_close(solve(model, z, 7, "rk4"), z + 1)
        self.assertEqual(model.calls, 28)
        class Exponential(torch.nn.Module):
            def forward(self, x, t):
                return x
        y = solve(Exponential(), torch.ones(2, 2), 8, "rk4")
        torch.testing.assert_close(y, torch.ones_like(y) * torch.exp(torch.tensor(1.)),
                                   atol=1e-5, rtol=1e-5)

    def test_strict_precision_restored_after_exception(self):
        old_a = torch.backends.cuda.matmul.allow_tf32
        old_b = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            with self.assertRaises(RuntimeError):
                with strict_eval_precision():
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    self.assertFalse(torch.backends.cudnn.allow_tf32)
                    raise RuntimeError("intentional test")
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_a
            torch.backends.cudnn.allow_tf32 = old_b

    def test_requires_two_doublings_and_rk4(self):
        cfg=dict(reference_start=2, reference_min=8, reference_max=16,
                 reference_rtol=.001, reference_atol=1e-5, reference_consecutive=2)
        steps, diag=choose_reference_steps(Constant(), torch.zeros(8,2), cfg,
                                           QuietProgress(), "test")
        self.assertEqual(steps,8)
        self.assertEqual(len(diag["history"]),2)
        self.assertTrue(diag["converged"])
        self.assertTrue(diag["cross_check"]["passed"])
        self.assertEqual(diag["cross_check"]["nfe"],diag["nfe"])
        self.assertEqual(diag["precision"],"fp32_no_tf32")

    def test_failed_teacher_does_not_generate_pairs(self):
        class DifficultTimeField(torch.nn.Module):
            def forward(self,x,t):
                return (1000*t**6)[:,None].expand_as(x)
        cfg=dict(reference_start=2, reference_min=8, reference_max=8,
                 reference_rtol=1e-10, reference_atol=0., reference_consecutive=2,
                 probe_samples=4)
        with tempfile.TemporaryDirectory() as root:
            with patch("flowlab.engine.model_from_checkpoint",return_value=DifficultTimeField()):
                with self.assertRaises(ReferenceConvergenceError):
                    make_pairs("toy",0,cfg,root,"unused.pt",torch.device("cpu"),QuietProgress())
            folder=Path(root)/"toy/seed_0/pairs"
            self.assertFalse((folder/"pairs.pt").exists())
            diag=json.loads((folder/"teacher_solver.json").read_text())
            self.assertFalse(diag["converged"])
            self.assertFalse(diag["cross_check"]["passed"])

    def test_adaptive_reference_seeds_are_not_split(self):
        rows=[dict(task="toy",method="linear",sampler="heun_reference",nfe=nfe,
                   seed=s,reference_converged=(s!=1),solver_mse=0.,sw2=v)
              for s,nfe,v in [(0,128,.1),(1,512,.2),(2,256,.3)]]
        summary=aggregate_rows(rows)
        self.assertEqual(len(summary),1)
        self.assertEqual(summary[0]["seed_count"],3)
        self.assertAlmostEqual(summary[0]["sw2_mean"],.2)
        self.assertEqual(summary[0]["reference_failed_seeds"],1)
        self.assertFalse(summary[0]["reference_all_converged"])


if __name__ == "__main__":
    unittest.main()
