#!/usr/bin/env python3
import argparse
from flowlab.reporting import build_report

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="runs/round2")
args = parser.parse_args()
build_report(args.out)
