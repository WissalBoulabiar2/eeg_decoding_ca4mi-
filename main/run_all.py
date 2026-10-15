"""
Everything needed for the paper table, as one resumable job list with progress and time estimates.

  1. ProtoMem tuning + ablation (tune_mp.GRID_PROTOMEM), selected on VALIDATION only
  2. the six baselines on the clean protocol (run_benchmark_clean.py)
  3. CA4MI-MP (phase 5 with the fixes)
  4. final table: baselines + CA4MI-MP + ProtoMem with the validation-selected configuration

One job = one (method/config, seed). Finished jobs are detected from their result files and skipped, so the same
command resumes after an interrupted session. --after_run is executed after every job (push results out of Kaggle).
Progress is printed before every job and written to <out>/progress.txt.

    python main/run_all.py --data_dir /kaggle/working/data/ --out /kaggle/working/all --n_runs 3 \
        --after_run 'bash tools/kaggle_sync.sh push /kaggle/working/all'
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import glob
import subprocess
import time
import numpy as np
import tune_mp

BASELINES = ['finetuning', 'ewc', 'er', 'der', 'mudvi', 'cger']


def fmt(seconds):
    if seconds is None:
        return "?"
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h}h{m:02d}"


def build_jobs(out, data_dir, n_runs, der_set):
    """(kind, name, done, command) for every job, in execution order."""
    jobs = []
    for cfg, extra in tune_mp.GRID_PROTOMEM.items():
        for s in range(n_runs):
            done = os.path.exists(f"{out}/protomem/{cfg}/protomem_9_subjects_seed_{s}/val_accuracy_matrix.txt")
            cmd = ['python', 'main/main_protomem.py', '--config', './configs/protomem.yml', '--n_runs', '1',
                   '--checkpoint', f"{out}/protomem/{cfg}", '--set', f"data_dir={data_dir}", f"seed_offset={s}", *extra]
            jobs.append(('protomem', f"ProtoMem [{cfg}] seed {s}", done, cmd))
    for m in BASELINES:
        for s in range(n_runs):
            done = bool(glob.glob(f"{out}/bench/{m}/**/acc_run_{s}_subject_9.txt", recursive=True))
            cmd = ['python', 'main/run_benchmark_clean.py', '--methods', m, '--data_dir', data_dir, '--n_runs', '1',
                   '--seed_offset', str(s), '--out', f"{out}/bench"]
            if m == 'der' and der_set:
                cmd += ['--set', *der_set]
            jobs.append(('baseline', f"{m} seed {s}", done, cmd))
    for s in range(n_runs):
        done = bool(glob.glob(f"{out}/ca4mi_mp/**/ca4mi_mp_9_subjects_seed_{s}/accuracy_matrix.txt", recursive=True))
        cmd = ['python', 'main/main_mp.py', '--config', './configs/ca4mi_mp.yml', '--phase', '5', '--n_runs', '1',
               '--checkpoint', f"{out}/ca4mi_mp", '--set', f"data_dir={data_dir}", 'wandb_mode=disabled',
               f"seed_offset={s}"]
        jobs.append(('ca4mi_mp', f"CA4MI-MP seed {s}", done, cmd))
    return jobs


def final_table(out):
    print("\n" + "=" * 80 + "\nPROTOMEM: SELECTION ON VALIDATION\n" + "=" * 80)
    tune_mp.GRID = tune_mp.GRID_PROTOMEM
    best = tune_mp.summary(f"{out}/protomem")
    print("\n" + "=" * 80 + f"\nFINAL TABLE (ProtoMem = validation-selected config: {best})\n" + "=" * 80)
    cmd = ['python', 'main/run_benchmark_clean.py', '--summary', '--out', f"{out}/bench",
           '--mp_dir', f"{out}/ca4mi_mp"]
    if best:
        cmd += ['--proto_dir', f"{out}/protomem/{best}"]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description='All runs for the paper table, resumable, with progress')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--out', type=str, default='./checkpoints/all')
    parser.add_argument('--n_runs', type=int, default=3)
    parser.add_argument('--der_epochs', type=int, default=None,
                        help='epochs per subject for DER (its config uses der_epochs=1); default: keep the config')
    parser.add_argument('--time_limit_h', type=float, default=11.0, help='stop launching jobs after this many hours')
    parser.add_argument('--after_run', type=str, default=None)
    parser.add_argument('--table_only', action='store_true')
    flags = parser.parse_args()
    os.makedirs(flags.out, exist_ok=True)

    if not flags.table_only:
        der_set = [f"der_epochs={flags.der_epochs}"] if flags.der_epochs else []
        jobs = build_jobs(flags.out, flags.data_dir, flags.n_runs, der_set)
        todo = [j for j in jobs if not j[2]]
        n_total, n_done = len(jobs), len(jobs) - len(todo)
        print(f"\n{n_total} jobs in total: {n_done} already done, {len(todo)} to run\n", flush=True)
        t0, durations = time.time(), {}
        for i, (kind, name, _, cmd) in enumerate(todo):
            elapsed = time.time() - t0
            if elapsed > flags.time_limit_h * 3600:
                print(f"Time limit of this session reached: {len(todo) - i} jobs left for the next session")
                break
            # remaining time: per-kind average when known, otherwise the overall average
            all_d = [d for v in durations.values() for d in v]
            est = [np.mean(durations[k]) if durations.get(k) else (np.mean(all_d) if all_d else None)
                   for k, _, _, _ in todo[i:]]
            eta = None if any(e is None for e in est) else sum(est)
            line = (f"[job {n_done + i + 1}/{n_total}] {name} | jobs left: {len(todo) - i} | "
                    f"elapsed: {fmt(elapsed)} | estimated time left: {fmt(eta) if eta else 'known after the first job'}")
            print("\n" + "#" * len(line) + f"\n{line}\n" + "#" * len(line), flush=True)
            with open(os.path.join(flags.out, 'progress.txt'), 'w') as f:
                f.write(line + "\n")
            start = time.time()
            r = subprocess.run(cmd)
            durations.setdefault(kind, []).append(time.time() - start)
            if r.returncode != 0:
                print(f"!!! job failed: {name} (it will be retried in the next session)", flush=True)
            if flags.after_run:
                subprocess.run(flags.after_run, shell=True)
    final_table(flags.out)


if __name__ == '__main__':
    main()
