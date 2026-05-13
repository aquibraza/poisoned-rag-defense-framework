#!/usr/bin/env python3
"""Thin wrapper: run eval_asr from repo root so scripts/compute_asr_from_results.py works."""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import eval_asr

if __name__ == "__main__":
    eval_asr.main()
