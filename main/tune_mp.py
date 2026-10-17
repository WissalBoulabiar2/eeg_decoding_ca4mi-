"""
Hyperparameter selection for CA4MI-MP on the VALIDATION matrices (val_accuracy_matrix.txt), never on the test.

Each configuration changes one setting of the reference (phase 5 with the fixes) and is run on the same seeds.
Selection rule (fixed before looking at any result): highest mean validation ACC among the configurations whose
validation BWT is not more than 1 point below the reference. The test matrices are only printed for the reference
and the selected configuration.

    python main/tune_mp.py --data_dir /kaggle/working/data/ --out /kaggle/working/tune --n_runs 3
    python main/tune_mp.py --summary --out /kaggle/working/tune
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import glob
import subprocess
import time
import numpy as np

REFERENCE = 'ref'
GRID = {  # name -> extra --set overrides (one change each w.r.t. the reference)
    'ref': [],
    'sel20': ['min_select_epoch=20'],          # no checkpoint before 20 epochs (noisy 22-trial validation)
    'distill': ['replay_distill=yes'],         # replay logit consistency (DER-style)
    'pro0.01': ['pro_loss_reg=0.01'],          # prototype loss 10x stronger
    'mem40': ['n_samples=40'],                 # 40 replayed trials per subject (baselines DER/MUDVI/CGER use 100)
    'modes2': ['modes_per_subject=2'],         # 2 modes/class/subject: Ward consolidation becomes active on 2a
}
# ProtoMem (main/main_protomem.py): tuning grid, each entry is also an ablation row
GRID_PROTOMEM = {
    'ref': [],
    'tau0.05': ['tau=0.05'],                   # sharper proxy logits
    'gen0': ['gen_reg=0'],                     # ablation: no pseudo-feature replay from the memory
    'gen1': ['gen_reg=1.0'],                   # stronger pseudo-feature replay
    'distill0': ['distill_reg=0'],             # ablation: no logit consistency
    'nodrift': ['drift_comp=no'],              # ablation: no drift compensation
    'modes1': ['modes_per_subject=1'],         # ablation: single mode per class and subject
    'mem40': ['n_samples=40'],                 # 40 exemplars per subject
}
# ProtoMem round 2, around the configuration selected in round 1 (mem40); its first entry is the reference
_R2 = ['n_samples=40']
GRID_PROTOMEM2 = {
    'ref40': _R2,
    'mem100': ['n_samples=100'],               # same exemplar budget as DER / MUDVI / CGER
    'tau0.2': _R2 + ['tau=0.2'],
    'lr3e-4': _R2 + ['lr=0.0003'],
    'pat20': _R2 + ['patience=20', 'n_epochs=200'],
    'distill1': _R2 + ['distill_reg=1.0'],
    'proxies5': _R2 + ['n_proxies=5'],
}
# ProtoMem round 3, around the configuration selected in round 2 (distill1 = n_samples 40, distill_reg 1.0)
_R3 = ['n_samples=40', 'distill_reg=1.0']
GRID_PROTOMEM3 = {
    'ref_d1': _R3,
    'mem100': _R3 + ['n_samples=100'],                  # best validation ACC in round 2
    'lr3e-4': _R3 + ['lr=0.0003'],                      # best validation BWT in round 2
    'mem100_lr3e-4': _R3 + ['n_samples=100', 'lr=0.0003'],
    'consol5': _R3 + ['consolidate_epochs=5'],          # balanced consolidation after each subject
    'consol10': _R3 + ['consolidate_epochs=10'],
    'bntrain': _R3 + ['freeze_past_bn=no'],             # past subjects' BN affine trainable through replay
}
# ProtoMem round 4 (last), around the configuration selected in round 3 (mem100): plasticity
_R4 = ['n_samples=100', 'distill_reg=1.0']
GRID_PROTOMEM4 = {
    'ref_m100': _R4,
    'aug': _R4 + ['augment=yes'],                       # EEG data augmentation
    'aug_strong': _R4 + ['augment=yes', 'aug_shift=100', 'aug_noise=0.1'],
    'noEA': _R4 + ['ea=no'],                            # the baselines use no alignment
    'consol10': _R4 + ['consolidate_epochs=10'],        # best validation BWT among the round-3 consolidations
}
# ProtoMem round 5, around consol10 (round 4): consolidation strength and learning rate. Selection seeds 0-2 only;
# the final evaluation (main/run_final.py) uses seeds that were never used for any choice.
_R5 = ['n_samples=100', 'distill_reg=1.0', 'consolidate_epochs=10']
GRID_PROTOMEM5 = {
    'ref_c10': _R5,
    'consol20': _R5 + ['consolidate_epochs=20'],
    'consol10_lr0.3': _R5 + ['consolidate_lr_factor=0.3'],
    'consol10_lr3e-4': _R5 + ['lr=0.0003'],
    'consol20_lr3e-4': _R5 + ['consolidate_epochs=20', 'lr=0.0003'],
}
# ProtoMem round 6: weight averaging (EMA) on top of the round-5 configuration, to reduce run-to-run variance
GRID_PROTOMEM6 = {
    'ref_c10': _R5,
    'ema0.99': _R5 + ['ema_decay=0.99'],
    'ema0.995': _R5 + ['ema_decay=0.995'],
}
# ProtoMem round 7: balanced current/replay batches (plasticity on new subjects), on top of the round-5 config
GRID_PROTOMEM7 = {
    'ref_c10': _R5,
    'balanced': _R5 + ['balanced_replay=yes'],
    'balanced_d0.5': _R5 + ['balanced_replay=yes', 'distill_reg=0.5'],
}
GRIDS = {'ca4mi_mp': GRID, 'protomem': GRID_PROTOMEM, 'protomem2': GRID_PROTOMEM2, 'protomem3': GRID_PROTOMEM3,
         'protomem4': GRID_PROTOMEM4, 'protomem5': GRID_PROTOMEM5, 'protomem6': GRID_PROTOMEM6,
         'protomem7': GRID_PROTOMEM7}


def run_command(method, out, name, data_dir, seed, extra):
    if method.startswith('protomem'):
        return ['python', 'main/main_protomem.py', '--config', './configs/protomem.yml', '--n_runs', '1',
                '--checkpoint', os.path.join(out, name), '--set', f'data_dir={data_dir}', f'seed_offset={seed}', *extra]
    return ['python', 'main/main_mp.py', '--config', './configs/ca4mi_mp.yml', '--phase', '5',
            '--n_runs', '1', '--checkpoint', os.path.join(out, name),
            '--set', f'data_dir={data_dir}', 'wandb_mode=disabled', f'seed_offset={seed}', *extra]


def seed_of(path):
    return os.path.basename(os.path.dirname(path)).rsplit('_', 1)[1]


def matrices(out, name, kind):
    """kind = 'val_accuracy_matrix' or 'accuracy_matrix'; one matrix per seed."""
    files = glob.glob(f"{out}/{name}/**/*_9_subjects_seed_*/{kind}.txt", recursive=True)
    return {seed_of(f): np.loadtxt(f) for f in sorted(files)}


def acc_bwt(a):
    n = len(a)
    return a[-1].mean(), np.sum(a[-1] - np.diag(a)) / (n - 1)


def stats(mats):
    v = np.array([acc_bwt(a) for a in mats.values()])
    return v.mean(0), v.std(0), len(v)


def summary(out):
    REFERENCE = next(iter(GRID))  # the first configuration of the grid is its reference
    rows = {}
    print(f"\nVALIDATION (selection only)\n{'config':10s} {'val ACC':>14s} {'val BWT':>14s}  seeds")
    for name in GRID:
        val = matrices(out, name, 'val_accuracy_matrix')
        if not val:
            print(f"{name:10s} {'(no results)':>14s}")
            continue
        m, s, n = stats(val)
        rows[name] = m
        print(f"{name:10s} {m[0]:6.2f} ± {s[0]:4.2f} {m[1]:6.2f} ± {s[1]:4.2f}  {n}")
    if REFERENCE not in rows:
        print("\nReference not available yet: no selection.")
        return None
    ref_bwt = rows[REFERENCE][1]
    ok = {k: v for k, v in rows.items() if v[1] >= ref_bwt - 1.0}
    best = max(ok, key=lambda k: ok[k][0])
    better = [k for k, v in ok.items() if k != REFERENCE and v[0] > rows[REFERENCE][0]]
    print(f"\nSelected on validation: {best}")
    print(f"Changes that improve validation ACC without hurting validation BWT: {better or 'none'}")
    print(f"\nTEST (reference and selected only)\n{'config':10s} {'ACC':>14s} {'BWT':>14s}  seeds")
    for name in dict.fromkeys([REFERENCE, best]):
        m, s, n = stats(matrices(out, name, 'accuracy_matrix'))
        print(f"{name:10s} {m[0]:6.2f} ± {s[0]:4.2f} {m[1]:6.2f} ± {s[1]:4.2f}  {n}")
    return best


def main():
    global GRID
    parser = argparse.ArgumentParser(description='CA4MI-MP / ProtoMem hyperparameter selection on validation')
    parser.add_argument('--method', choices=list(GRIDS), default='ca4mi_mp')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--out', type=str, default='./checkpoints/tune_mp')
    parser.add_argument('--n_runs', type=int, default=3)
    parser.add_argument('--configs', nargs='*', default=None, help='subset of the grid (default: all)')
    parser.add_argument('--time_limit_h', type=float, default=10.5, help='stop launching runs after this many hours')
    parser.add_argument('--after_run', type=str, default=None,
                        help='shell command run after every finished seed (e.g. push results out of Kaggle)')
    parser.add_argument('--summary', action='store_true')
    flags = parser.parse_args()
    GRID = GRIDS[flags.method]
    flags.configs = flags.configs or list(GRID)
    unknown = set(flags.configs) - set(GRID)
    if unknown:
        parser.error(f"unknown configs for {flags.method}: {sorted(unknown)}")

    t0 = time.time()
    if not flags.summary:
        # One process per (config, seed): a seed already saved is never re-run, so an interrupted
        # session loses at most the seed that was running.
        jobs = [(name, seed) for name in flags.configs for seed in range(flags.n_runs)]
        todo = [(n, s) for n, s in jobs if str(s) not in matrices(flags.out, n, 'val_accuracy_matrix')]
        print(f"\n{len(jobs)} runs in total: {len(jobs) - len(todo)} already done, {len(todo)} to run\n", flush=True)
        durations = []
        for i, (name, seed) in enumerate(todo):
            elapsed = time.time() - t0
            if elapsed > flags.time_limit_h * 3600:
                print(f"time limit reached: {len(todo) - i} runs left for the next session")
                break
            left = len(todo) - i
            eta = f"{np.mean(durations) * left / 3600:.1f} h" if durations else "known after the first run"
            line = (f"[run {len(jobs) - len(todo) + i + 1}/{len(jobs)}] {name} seed {seed} | runs left: {left} | "
                    f"elapsed: {elapsed / 3600:.1f} h | estimated time left: {eta}")
            print("\n" + "#" * len(line) + f"\n{line}\n" + "#" * len(line), flush=True)
            os.makedirs(flags.out, exist_ok=True)
            with open(os.path.join(flags.out, 'progress.txt'), 'w') as f:
                f.write(line + "\n")
            start = time.time()
            r = subprocess.run(run_command(flags.method, flags.out, name, flags.data_dir, seed, GRID[name]))
            durations.append(time.time() - start)
            if r.returncode != 0:
                print(f"!!! run failed: {name} seed {seed} (retried in the next session)", flush=True)
            if flags.after_run:
                subprocess.run(flags.after_run, shell=True, check=False)
    summary(flags.out)


if __name__ == '__main__':
    main()
