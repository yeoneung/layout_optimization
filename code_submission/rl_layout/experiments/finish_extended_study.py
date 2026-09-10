"""Run the frozen test after validation, then independently audit and analyze.

This wrapper does not edit the manuscript or select from test outcomes. It may
be resumed with the same output root after an interrupted test.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from extended_protocol import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int,
                        help="physical CPU workers; default uses available physical core count")
    parser.add_argument("--run-validation", action="store_true",
                        help="run/resume validation before selecting the test configuration")
    args = parser.parse_args()
    root = args.out.resolve()
    import psutil
    workers = args.workers if args.workers is not None else psutil.cpu_count(logical=False)
    if not workers or workers < 1:
        raise ValueError("a positive verified physical worker count is required")
    scripts = Path(__file__).resolve().parent
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    env["PYTHONUTF8"] = "1"
    def call(script, *arguments):
        subprocess.run([sys.executable, "-u", str(scripts / script), *map(str, arguments)],
                       cwd=str(scripts), env=env, check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if args.run_validation:
        call("run_extended_study.py", "--out", root, "--phase", "validation", "--workers", workers)
    validation = json.loads((root / "validation" / "protocol.json").read_text(encoding="utf-8"))
    if validation["settings"]["workers"] != workers:
        raise RuntimeError("validation and test must use the same worker count; use a separate output root")
    call("analyze_extended_study.py", root / "validation")
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    candidates = sorted(k for k in selection["scores"] if k.startswith("ALNS:"))
    chosen = max(candidates, key=lambda k: selection["scores"][k]).split(":", 1)[1]
    if selection["test_results_used"] or chosen != selection["selected_configuration"]:
        raise RuntimeError("invalid validation selection")
    call("run_extended_study.py", "--out", root, "--phase", "test", "--configuration", chosen,
         "--workers", workers)
    atomic_json(root / "status.json", {"status": "auditing", "phase": "test", "updated_unix": time.time()})
    call("analyze_extended_study.py", root / "test")
    atomic_json(root / "status.json", {"status": "complete", "phase": "test", "audit_passed": True,
                                       "finished_unix": time.time(), "report": str(root / "test" / "report.md"),
                                       "manuscript_results_updated": False})
    print("TEST AND AUDIT COMPLETE; the manuscript still requires an evidence-based results update.", flush=True)


if __name__ == "__main__":
    main()
