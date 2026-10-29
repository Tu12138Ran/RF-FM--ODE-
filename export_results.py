#!/usr/bin/env python3
"""Export report AND small numerical diagnostics; exclude large model/pair files."""
import argparse
import json
import tarfile
from pathlib import Path


def export_results(out, destination):
    out=Path(out).resolve()
    destination=Path(destination).resolve()
    if not (out/"report/report.html").exists():
        raise RuntimeError("Report is not ready. Wait for completion or run report.py first.")
    names={"config.json","environment.json","pip_freeze.txt","source_hashes.json",
           "status.json","events.jsonl","console.log","error.txt","evaluation.json",
           "reference_check.json","teacher_solver.json","training_summary.json",
           "generation_summary.json","data_baseline.json"}
    files=[p for p in out.rglob("*") if p.is_file() and not p.is_symlink()
           and (p.name in names or p.is_relative_to(out/"report")) and p!=destination]
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(destination,"w:gz") as archive:
        for path in sorted(files):
            archive.add(path,arcname=path.relative_to(out).as_posix(),recursive=False)
    return dict(path=str(destination),files=len(files),bytes=destination.stat().st_size)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="runs/round2")
    parser.add_argument("--output",default="flow_path_round2_results.tar.gz")
    args=parser.parse_args()
    print(json.dumps(export_results(args.out,args.output),indent=2))
