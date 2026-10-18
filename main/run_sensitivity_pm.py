"""
Stability / plasticity sensitivity of ProtoMem to two hyperparameters (final configuration otherwise):
  gen_reg (weight of the prototype / pseudo-feature replay loss) in {0.1, 0.5, 1.0}
  tau     (temperature of the proxy classifier)                 in {0.05, 0.1, 0.2}
The centre (0.5, 0.1) is the final ProtoMem (final_bal runs) and is not re-run.

Results: <out>/g<gen_reg>_t<tau>/protomem_9_subjects_seed_<s>/accuracy_matrix.txt; tools/plot_sensitivity.py draws
the heatmaps (stability = BWT, plasticity = mean accuracy on each subject right after learning it).

    python main/run_sensitivity_pm.py --data_dir /kaggle/working/data/ --out /kaggle/working/sensitivity
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import subprocess
import time
import numpy as np

FINAL = ['n_samples=100', 'distill_reg=1.0', 'consolidate_epochs=10']
GEN = [0.1, 0.5, 1.0]
TAU = [0.05, 0.1, 0.2]


def fmt(seconds):
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h}h{m:02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out', default='/kaggle/working/sensitivity')
    parser.add_argument('--seeds', type=int, nargs='*', default=[8, 9, 10])
    parser.add_argument('--time_limit_h', type=float, default=11.0)
    flags = parser.parse_args()

    jobs = [(g, t, s) for g in GEN for t in TAU for s in flags.seeds
            if not (g == 0.5 and t == 0.1)
            and not os.path.exists(f"{flags.out}/g{g}_t{t}/protomem_9_subjects_seed_{s}/accuracy_matrix.txt")]
    print(f"\n{len(jobs)} sensitivity jobs to run\n", flush=True)
    t0, durations = time.time(), []
    for i, (g, t, s) in enumerate(jobs):
        if time.time() - t0 > flags.time_limit_h * 3600:
            print(f"Time limit reached: {len(jobs) - i} jobs not run (run the same cell again in this session).")
            break
        eta = fmt(np.mean(durations) * (len(jobs) - i)) if durations else "known after the first job"
        line = (f"[job {i + 1}/{len(jobs)}] [sensitivity gen={g} tau={t}] ProtoMem seed {s} | "
                f"elapsed: {fmt(time.time() - t0)} | estimated time left: {eta}")
        print("\n" + "#" * len(line) + f"\n{line}\n" + "#" * len(line), flush=True)
        start = time.time()
        cmd = ['python', 'main/main_protomem.py', '--config', './configs/protomem.yml', '--n_runs', '1',
               '--checkpoint', f"{flags.out}/g{g}_t{t}", '--set', f"data_dir={flags.data_dir}", f"seed_offset={s}",
               *FINAL, f"gen_reg={g}", f"tau={t}"]
        if subprocess.run(cmd).returncode != 0:
            print(f"!!! job failed: gen={g} tau={t} seed {s}", flush=True)
        durations.append(time.time() - start)
    print("\nDONE. Zip and download the sensitivity folder.")


if __name__ == '__main__':
    main()
