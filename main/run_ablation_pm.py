"""
Cumulative ablation of ProtoMem on the final seeds (A5 = final configuration = the final_bal runs, not re-run).

  A0  single encoder + subject BN + proxy classifier + exemplar replay (cross-entropy only)
  A1  + logit consistency on exemplars (DER)
  A2  + pseudo-feature replay from the Gaussian memory (one mode per class and subject)
  A3  + multi-mode memory (two modes per class and subject, Ward consolidation)
  A4  + drift compensation of the memory
  A5  + balanced consolidation                      (= final ProtoMem)

Results: <out>/<row>/protomem_9_subjects_seed_<s>/accuracy_matrix.txt; each job also prints
"[job k/N] [ablation A2] ProtoMem seed 8" and the per-subject "test acc:" lines. tools/ablation_table.py builds the table.

    python main/run_ablation_pm.py --data_dir /kaggle/working/data/ --out /kaggle/working/ablation
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import subprocess
import time
import numpy as np

BASE = ['n_samples=100']
OFF = ['distill_reg=0', 'gen_reg=0', 'drift_comp=no', 'consolidate_epochs=0', 'modes_per_subject=1']
ROWS = {
    'A0': BASE + OFF,
    'A1': BASE + OFF + ['distill_reg=1.0'],
    'A2': BASE + OFF + ['distill_reg=1.0', 'gen_reg=0.5'],
    'A3': BASE + OFF + ['distill_reg=1.0', 'gen_reg=0.5', 'modes_per_subject=2'],
    'A4': BASE + OFF + ['distill_reg=1.0', 'gen_reg=0.5', 'modes_per_subject=2', 'drift_comp=yes'],
}


def fmt(seconds):
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h}h{m:02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out', default='/kaggle/working/ablation')
    parser.add_argument('--seeds', type=int, nargs='*', default=[8, 9, 10, 11, 12])
    parser.add_argument('--rows', nargs='*', default=list(ROWS), choices=list(ROWS))
    parser.add_argument('--time_limit_h', type=float, default=11.0)
    flags = parser.parse_args()

    jobs = [(r, s) for r in flags.rows for s in flags.seeds
            if not os.path.exists(f"{flags.out}/{r}/protomem_9_subjects_seed_{s}/accuracy_matrix.txt")]
    print(f"\n{len(jobs)} ablation jobs to run\n", flush=True)
    t0, durations = time.time(), []
    for i, (r, s) in enumerate(jobs):
        if time.time() - t0 > flags.time_limit_h * 3600:
            print(f"Time limit reached: {len(jobs) - i} jobs not run (run the same cell again in this session).")
            break
        eta = fmt(np.mean(durations) * (len(jobs) - i)) if durations else "known after the first job"
        line = f"[job {i + 1}/{len(jobs)}] [ablation {r}] ProtoMem seed {s} | elapsed: {fmt(time.time() - t0)} | estimated time left: {eta}"
        print("\n" + "#" * len(line) + f"\n{line}\n" + "#" * len(line), flush=True)
        start = time.time()
        cmd = ['python', 'main/main_protomem.py', '--config', './configs/protomem.yml', '--n_runs', '1',
               '--checkpoint', f"{flags.out}/{r}", '--set', f"data_dir={flags.data_dir}", f"seed_offset={s}", *ROWS[r]]
        if subprocess.run(cmd).returncode != 0:
            print(f"!!! job failed: {r} seed {s}", flush=True)
        durations.append(time.time() - start)
    print("\nDONE. Download the results: zip the ablation folder (see the notebook instructions).")


if __name__ == '__main__':
    main()
