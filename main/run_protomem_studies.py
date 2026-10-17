"""
ProtoMem studies for the paper, with the final configuration (selected on validation, never retuned here):
  - buffer size: exemplars per subject in {0, 20, 50, 200} (100 = the main result, already available)
  - subject order: the random order of the earlier figure (S4, S5, S7, S3, S9, S1, S2, S6, S8)
on the final seeds 3-7. One job = one (study, seed); finished jobs are skipped. Every job prints a header
"[job k/N] [buffer 50] ProtoMem seed 3" or "[job k/N] [order random] ProtoMem seed 3", followed by the per-subject
"test acc:" lines, so the figures can be rebuilt from the saved log alone (tools/plot_buffer_order.py).

    python main/run_protomem_studies.py --data_dir /kaggle/working/data/ --out /kaggle/working/studies
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import subprocess
import time
import numpy as np

FINAL = ['n_samples=100', 'distill_reg=1.0', 'consolidate_epochs=10']   # ProtoMem final configuration
BUFFERS = [0, 20, 50, 200]
RANDOM_ORDER = [4, 5, 7, 3, 9, 1, 2, 6, 8]


def fmt(seconds):
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h}h{m:02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out', default='./checkpoints/studies')
    parser.add_argument('--seeds', type=int, nargs='*', default=[3, 4, 5, 6, 7])
    parser.add_argument('--studies', nargs='*', default=['buffer', 'order'], choices=['buffer', 'order'])
    parser.add_argument('--time_limit_h', type=float, default=11.0)
    flags = parser.parse_args()

    jobs = []
    if 'buffer' in flags.studies:
        for nb in BUFFERS:
            for s in flags.seeds:
                jobs.append((f"buffer {nb}", f"{flags.out}/buffer_{nb}", s, FINAL + [f'n_samples={nb}']))
    if 'order' in flags.studies:
        order = '[' + ','.join(map(str, RANDOM_ORDER)) + ']'
        for s in flags.seeds:
            jobs.append(("order random", f"{flags.out}/order_random", s, FINAL + [f'subject_order={order}']))
    done = lambda ckpt, s: os.path.exists(f"{ckpt}/protomem_9_subjects_seed_{s}/accuracy_matrix.txt")
    todo = [j for j in jobs if not done(j[1], j[2])]
    print(f"\n{len(jobs)} jobs in total: {len(jobs) - len(todo)} already done, {len(todo)} to run\n", flush=True)

    t0, durations = time.time(), []
    for i, (tag, ckpt, s, extra) in enumerate(todo):
        if time.time() - t0 > flags.time_limit_h * 3600:
            print("Time limit reached: run the same cell again to continue.")
            break
        eta = fmt(np.mean(durations) * (len(todo) - i)) if durations else "known after the first job"
        line = (f"[job {len(jobs) - len(todo) + i + 1}/{len(jobs)}] [{tag}] ProtoMem seed {s} | "
                f"elapsed: {fmt(time.time() - t0)} | estimated time left: {eta}")
        print("\n" + "#" * len(line) + f"\n{line}\n" + "#" * len(line), flush=True)
        start = time.time()
        cmd = ['python', 'main/main_protomem.py', '--config', './configs/protomem.yml', '--n_runs', '1',
               '--checkpoint', ckpt, '--set', f"data_dir={flags.data_dir}", f"seed_offset={s}", *extra]
        if subprocess.run(cmd).returncode != 0:
            print(f"!!! job failed: [{tag}] seed {s}", flush=True)
        durations.append(time.time() - start)


if __name__ == '__main__':
    main()
