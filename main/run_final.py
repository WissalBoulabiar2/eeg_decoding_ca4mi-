"""
Final experiments for the paper, as one resumable job list with progress and time estimates.

  A. ProtoMem round 5 (tune_mp.GRID_PROTOMEM5) on the SELECTION seeds 0-2, selected on validation only.
  B. Final evaluation on NEW seeds (3-7 by default) never used for any choice, for every method:
     Finetuning, EWC, ER, DER, MUDVI, CGER, CA4MI (original code, protocol-corrected), CA4MI-MP, ProtoMem (A's pick).
  C. Final table (mean ± std over the final seeds) + paired tests ProtoMem vs each method:
     paired t-test over seeds (ACC, BWT) and Wilcoxon signed-rank over (seed, subject) final accuracies.

One job = one (method/config, seed); finished jobs are detected from result files and skipped, so the same command
resumes after an interrupted session (restore <out> first). --after_run runs after every job.

    python main/run_final.py --data_dir /kaggle/working/data/ --out /kaggle/working/final_runs
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

BASELINES = ['finetuning', 'ewc', 'er', 'der', 'mudvi', 'cger', 'ca4mi']
NAMES = {'finetuning': 'Finetuning', 'ewc': 'EWC', 'er': 'ER', 'der': 'DER', 'mudvi': 'MUDVI', 'cger': 'CGER',
         'ca4mi': 'CA4MI', 'ca4mi_mp': 'CA4MI-MP', 'protomem': 'ProtoMem (ours)'}


def fmt(seconds):
    if seconds is None:
        return "?"
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h}h{m:02d}"


# ---------------------------------------------------------------------------- result files
def matrix_file(out, method, seed):
    """Final-evaluation accuracy matrix of (method, seed), or None."""
    if method in ('finetuning', 'ewc', 'er', 'der', 'mudvi', 'cger'):
        pat = f"{out}/final/bench/{method}/**/acc_run_{seed}_subject_9.txt"
    elif method == 'ca4mi':
        pat = f"{out}/final/bench/ca4mi/**/ca4mi_9_subjects_seed_{seed}/accuracy_matrix.txt"
    elif method == 'ca4mi_mp':
        pat = f"{out}/final/ca4mi_mp/**/ca4mi_mp_9_subjects_seed_{seed}/accuracy_matrix.txt"
    else:
        pat = f"{out}/final/protomem/**/protomem_9_subjects_seed_{seed}/accuracy_matrix.txt"
    files = sorted(glob.glob(pat, recursive=True))
    return files[-1] if files else None


def f1_file(acc_file):
    return (acc_file.replace('acc_run_', 'f1_run_') if 'acc_run_' in acc_file
            else acc_file.replace('accuracy_matrix', 'f1_macro_matrix'))


# ---------------------------------------------------------------------------- jobs
def protomem_cmd(checkpoint, data_dir, seed, extra):
    return ['python', 'main/main_protomem.py', '--config', './configs/protomem.yml', '--n_runs', '1',
            '--checkpoint', checkpoint, '--set', f"data_dir={data_dir}", f"seed_offset={seed}", *extra]


def selection_jobs(out, data_dir, sel_seeds):
    jobs = []
    for cfg, extra in tune_mp.GRID_PROTOMEM5.items():
        for s in sel_seeds:
            done = os.path.exists(f"{out}/tune5/{cfg}/protomem_9_subjects_seed_{s}/val_accuracy_matrix.txt")
            jobs.append(('protomem', f"[A selection] ProtoMem {cfg} seed {s}", done,
                         protomem_cmd(f"{out}/tune5/{cfg}", data_dir, s, extra)))
    return jobs


def final_jobs(out, data_dir, final_seeds, proto_extra, der_set):
    jobs = []
    for s in final_seeds:
        jobs.append(('protomem', f"[B final] ProtoMem seed {s}", matrix_file(out, 'protomem', s) is not None,
                     protomem_cmd(f"{out}/final/protomem", data_dir, s, proto_extra)))
        for m in BASELINES:
            cmd = ['python', 'main/run_benchmark_clean.py', '--methods', m, '--data_dir', data_dir, '--n_runs', '1',
                   '--seed_offset', str(s), '--out', f"{out}/final/bench"]
            if m == 'der' and der_set:
                cmd += ['--set', *der_set]
            kind = 'ca4mi' if m == 'ca4mi' else 'baseline'
            jobs.append((kind, f"[B final] {NAMES[m]} seed {s}", matrix_file(out, m, s) is not None, cmd))
        jobs.append(('ca4mi', f"[B final] CA4MI-MP seed {s}", matrix_file(out, 'ca4mi_mp', s) is not None,
                     ['python', 'main/main_mp.py', '--config', './configs/ca4mi_mp.yml', '--phase', '5',
                      '--n_runs', '1', '--checkpoint', f"{out}/final/ca4mi_mp", '--set', f"data_dir={data_dir}",
                      'wandb_mode=disabled', f"seed_offset={s}"]))
    return jobs


def run_jobs(jobs, out, t0, time_limit_h, after_run, durations, n_total, n_before):
    """Run the unfinished jobs; returns False if the session time limit stopped it."""
    todo = [j for j in jobs if not j[2]]
    for i, (kind, name, _, cmd) in enumerate(todo):
        elapsed = time.time() - t0
        if elapsed > time_limit_h * 3600:
            print(f"\nTime limit of this session reached: run the same cell again to continue.")
            return False
        all_d = [d for v in durations.values() for d in v]
        est = [np.mean(durations[k]) if durations.get(k) else (np.mean(all_d) if all_d else None)
               for k, _, _, _ in todo[i:]]
        eta = None if any(e is None for e in est) else sum(est)
        k = n_before + (len(jobs) - len(todo)) + i + 1
        line = (f"[job {k}/{n_total}] {name} | elapsed: {fmt(elapsed)} | "
                f"estimated time left (this stage): {fmt(eta) if eta else 'known after the first job'}")
        print("\n" + "#" * len(line) + f"\n{line}\n" + "#" * len(line), flush=True)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, 'progress.txt'), 'w') as f:
            f.write(line + "\n")
        start = time.time()
        r = subprocess.run(cmd)
        durations.setdefault(kind, []).append(time.time() - start)
        if r.returncode != 0:
            print(f"!!! job failed: {name} (retried in the next session)", flush=True)
        if after_run:
            subprocess.run(after_run, shell=True)
    return True


# ---------------------------------------------------------------------------- table + tests
def per_seed(out, method, seeds):
    res = {}
    for s in seeds:
        f = matrix_file(out, method, s)
        if f is None:
            continue
        a, f1 = np.loadtxt(f), np.loadtxt(f1_file(f))
        n = len(a)
        res[s] = {'acc': a[-1].mean(), 'bwt': np.sum(a[-1] - np.diag(a)) / (n - 1),
                  'fgt': np.mean([a[u:n - 1, u].max() - a[-1, u] for u in range(n - 1)]),
                  'f1': f1[-1].mean(), 'final_row': a[-1]}
    return res


def final_table(out, seeds):
    from scipy.stats import ttest_rel, wilcoxon
    methods = BASELINES + ['ca4mi_mp', 'protomem']
    data = {m: per_seed(out, m, seeds) for m in methods}
    proto = data['protomem']
    lines = [f"FINAL TABLE - seeds {list(seeds)} (never used for selection) - mean ± std",
             f"{'Method':18s} {'ACC (%)':>14s} {'BWT (%)':>14s} {'Forgetting':>14s} {'F1 (%)':>14s}  seeds"
             f"   p(ACC)  p(BWT)  p(Wilcoxon)"]
    for m in methods:
        d = data[m]
        if not d:
            lines.append(f"{NAMES[m]:18s} {'(no results)':>14s}")
            continue
        v = {k: np.array([d[s][k] for s in d]) for k in ('acc', 'bwt', 'fgt', 'f1')}
        row = f"{NAMES[m]:18s} " + " ".join(f"{v[k].mean():6.2f} ± {v[k].std():4.2f}"
                                            for k in ('acc', 'bwt', 'fgt', 'f1')) + f"  {len(d):5d}"
        common = sorted(set(d) & set(proto))
        if m != 'protomem' and len(common) >= 2:
            p_acc = ttest_rel([proto[s]['acc'] for s in common], [d[s]['acc'] for s in common]).pvalue
            p_bwt = ttest_rel([proto[s]['bwt'] for s in common], [d[s]['bwt'] for s in common]).pvalue
            diff = np.concatenate([proto[s]['final_row'] - d[s]['final_row'] for s in common])
            p_w = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            row += f"   {p_acc:6.3f}  {p_bwt:6.3f}  {p_w:8.3f}"
        lines.append(row)
    lines.append("p: ProtoMem vs method. Paired t-test over seeds (ACC, BWT); Wilcoxon over (seed, subject) "
                 "final accuracies.")
    text = "\n".join(lines)
    print("\n" + "=" * 100 + "\n" + text + "\n" + "=" * 100)
    with open(os.path.join(out, 'final_table.txt'), 'w') as f:
        f.write(text + "\n")


def main():
    parser = argparse.ArgumentParser(description='Final experiments: selection, evaluation on new seeds, table')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--out', type=str, default='./checkpoints/final_runs')
    parser.add_argument('--selection_seeds', type=int, nargs='*', default=[0, 1, 2])
    parser.add_argument('--final_seeds', type=int, nargs='*', default=[3, 4, 5, 6, 7])
    parser.add_argument('--der_epochs', type=int, default=None, help='DER epochs per subject (config: 1)')
    parser.add_argument('--time_limit_h', type=float, default=11.0)
    parser.add_argument('--after_run', type=str, default=None)
    parser.add_argument('--table_only', action='store_true')
    flags = parser.parse_args()
    os.makedirs(flags.out, exist_ok=True)

    if not flags.table_only:
        t0, durations = time.time(), {}
        sel = selection_jobs(flags.out, flags.data_dir, flags.selection_seeds)
        n_final = len(flags.final_seeds) * (len(BASELINES) + 2)
        n_total = len(sel) + n_final
        n_done = sum(j[2] for j in sel)
        print(f"\n{n_total} jobs in total ({len(sel)} selection + {n_final} final): "
              f"{n_done} selection jobs already done\n", flush=True)
        if not run_jobs(sel, flags.out, t0, flags.time_limit_h, flags.after_run, durations, n_total, 0):
            return
        print("\n" + "=" * 80 + "\nA. PROTOMEM ROUND 5: SELECTION ON VALIDATION (seeds "
              f"{flags.selection_seeds})\n" + "=" * 80)
        tune_mp.GRID = tune_mp.GRID_PROTOMEM5
        best = tune_mp.summary(f"{flags.out}/tune5")
        with open(os.path.join(flags.out, 'selected_config.txt'), 'w') as f:
            f.write(f"{best}: {' '.join(tune_mp.GRID_PROTOMEM5[best])}\n")
        print(f"\nProtoMem final configuration: {best} -> {tune_mp.GRID_PROTOMEM5[best]}")
        der_set = [f"der_epochs={flags.der_epochs}"] if flags.der_epochs else []
        fin = final_jobs(flags.out, flags.data_dir, flags.final_seeds, tune_mp.GRID_PROTOMEM5[best], der_set)
        if not run_jobs(fin, flags.out, t0, flags.time_limit_h, flags.after_run, durations, n_total, len(sel)):
            return
    final_table(flags.out, flags.final_seeds)


if __name__ == '__main__':
    main()
