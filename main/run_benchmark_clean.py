"""
Run the baselines on the clean protocol shared with CA4MI-MP (see dataloaders/dataloader_clean_benchmark.py).
The baseline methods, networks and configs are used unchanged; only data_dir, n_runs and checkpoint are overridden.

    python main/run_benchmark_clean.py --data_dir /kaggle/working/data/ --n_runs 3 --out /kaggle/working/bench
    python main/run_benchmark_clean.py ... --methods er der           # subset of methods
    python main/run_benchmark_clean.py --summary --out /kaggle/working/bench --mp_dir /kaggle/working/bench/ca4mi_mp
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import glob
import numpy as np
from omegaconf import OmegaConf
from trainer import EEGTrainer
import utils

os.environ.setdefault('WANDB_MODE', 'disabled')  # the original CA4MI code calls wandb.init without a mode

METHODS = ['finetuning', 'ewc', 'er', 'der', 'mudvi', 'cger']
NAMES = {'finetuning': 'Finetuning', 'ewc': 'EWC', 'er': 'ER', 'der': 'DER', 'mudvi': 'MUDVI', 'cger': 'CGER',
         'ca4mi': 'CA4MI'}


class CleanBaselineTrainer(EEGTrainer):
    def run_single_experiment(self, run_index):
        # seed_offset: run one given seed in its own process (seed and result files use run_index + offset)
        return super().run_single_experiment(run_index + int(self.args.get('seed_offset', 0)))

    def initialize_dataloader(self):
        from dataloaders import dataloader_clean_benchmark as clean
        if self.args.experiment in ('ewc', 'finetuning'):
            return clean.CleanBaselineStreaming(self.args)
        if self.args.experiment in ('er', 'der', 'mudvi', 'cger', 'ca4mi'):
            return clean.CleanReplayStreaming(self.args)
        raise NotImplementedError(self.args.experiment)


class CleanCA4MITrainer(CleanBaselineTrainer):
    """Original CA4MI algorithm (methods/ca4mi.py, EEGTrainer.run_ca4mi) on the clean protocol.
    Only two things differ from the original code, the same ones as for every other method:
      - data: subject order S1 -> S9, identical trials, replay with true labels in the training split only;
      - evaluation: past subjects are tested with the trained shared encoder. The original passes
        ca4mi.load_current_models(u), a freshly initialised network (main/trainer.py:233).
    Its losses, prototypes, IEA, mixup, checkpoint selection and replay routing are untouched."""

    def evaluate_all_previous_subjects(self, ca4mi, dataset, t, acc, lss, f1_mat, inf_time_mat):
        for u in range(t + 1):
            model = utils.load_model_with_shared_update(
                network_class=ca4mi.network.Cls_HEAD, args=ca4mi.args, current_model=ca4mi.model,
                checkpoint_dir=ca4mi.checkpoint, sub_id=u)
            res = ca4mi.test_previous_subjects(dataset[u]['test'], u, model=model)
            print(f">>> Test on {dataset[u]['name']:15s}: acc={res['acc_cls']:5.2f}%")
            acc[t, u], lss[t, u] = res['acc_cls'], res['loss_cls']
            f1_mat[t, u], inf_time_mat[t, u] = res['f1_macro'], res['avg_inf_time']


def result_exists(out, method, seed, n_subjects=9):
    if method == 'ca4mi':
        return bool(glob.glob(f"{out}/ca4mi/**/ca4mi_{n_subjects}_subjects_seed_{seed}/accuracy_matrix.txt",
                              recursive=True))
    return bool(glob.glob(f"{out}/{method}/**/acc_run_{seed}_subject_{n_subjects}.txt", recursive=True))


def metrics(acc, f1):
    n = len(acc)
    return (acc[-1].mean(), np.sum(acc[-1] - np.diag(acc)) / (n - 1),
            np.mean([acc[u:n - 1, u].max() - acc[-1, u] for u in range(n - 1)]), f1[-1].mean())


def baseline_runs(out, method, n_subjects=9):
    """One result per seed (run index); a seed run twice (interrupted batch) is counted once."""
    files = sorted(glob.glob(f"{out}/{method}/**/acc_run_*_subject_{n_subjects}.txt", recursive=True))
    by_seed = {os.path.basename(f).split('_')[2]: f for f in files}
    return [metrics(np.loadtxt(f), np.loadtxt(f.replace('acc_run_', 'f1_run_'))) for f in by_seed.values()]


def mp_runs(mp_dir, prefix='ca4mi_mp', n_subjects=9):
    files = sorted(glob.glob(f"{mp_dir}/**/{prefix}_{n_subjects}_subjects_seed_*/accuracy_matrix.txt",
                             recursive=True))
    by_seed = {os.path.basename(os.path.dirname(f)).rsplit('_', 1)[1]: f for f in files}
    return [metrics(np.loadtxt(f), np.loadtxt(f.replace('accuracy_matrix', 'f1_macro_matrix')))
            for f in by_seed.values()]


def summary(out, mp_dir=None, proto_dir=None, n_subjects=9):
    rows = [(NAMES[m], baseline_runs(out, m, n_subjects)) for m in METHODS]
    ca4mi = mp_runs(f"{out}/ca4mi", prefix='ca4mi', n_subjects=n_subjects)
    if ca4mi:
        rows.append(('CA4MI', ca4mi))
    if mp_dir:
        rows.append(('CA4MI-MP', mp_runs(mp_dir, n_subjects=n_subjects)))
    if proto_dir:
        rows.append(('ProtoMem (ours)', mp_runs(proto_dir, prefix='protomem', n_subjects=n_subjects)))
    print(f"\n{'Method':18s} {'ACC (%)':>14s} {'BWT (%)':>14s} {'Forgetting':>14s} {'F1 (%)':>14s}  seeds")
    for name, runs in rows:
        if not runs:
            print(f"{name:18s} {'(no results)':>14s}")
            continue
        m = np.array(runs)
        print(f"{name:18s} " + " ".join(f"{m[:, k].mean():6.2f} ± {m[:, k].std():4.2f}" for k in range(4))
              + f"  {len(m)}")


def main():
    parser = argparse.ArgumentParser(description='Baselines on the clean CA4MI-MP protocol')
    parser.add_argument('--methods', nargs='*', default=METHODS, choices=METHODS + ['ca4mi'])
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--n_runs', type=int, default=3)
    parser.add_argument('--out', type=str, default='./checkpoints/bench_clean')
    parser.add_argument('--device', type=str)
    parser.add_argument('--summary', action='store_true', help='only print the results table')
    parser.add_argument('--mp_dir', type=str, help='CA4MI-MP checkpoint dir to include in the table')
    parser.add_argument('--proto_dir', type=str, help='ProtoMem checkpoint dir to include in the table')
    parser.add_argument('--n_subjects', type=int, default=9)
    parser.add_argument('--seed_offset', type=int, default=0, help='run seeds seed_offset .. seed_offset+n_runs-1')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                        help='extra config overrides for every baseline (e.g. a fast test)')
    flags = parser.parse_args()

    if not flags.summary:
        for method in flags.methods:
            seeds = range(flags.seed_offset, flags.seed_offset + flags.n_runs)
            if all(result_exists(flags.out, method, s, flags.n_subjects) for s in seeds):
                print(f"skip {method}: seeds {list(seeds)} already done")
                continue
            args = OmegaConf.load(f'./configs/{method}.yml')
            override = {'data_dir': flags.data_dir, 'n_runs': flags.n_runs, 'n_subjects': flags.n_subjects,
                        'seed_offset': flags.seed_offset, 'checkpoint': os.path.join(flags.out, method)}
            if flags.device:
                override['device'] = flags.device
            args = OmegaConf.merge(args, override, OmegaConf.from_dotlist(flags.set))
            print("\n" + "=" * 80 + f"\nCLEAN BENCHMARK: {method}  (n_runs={args.n_runs})\n" + "=" * 80)
            trainer = CleanCA4MITrainer if method == 'ca4mi' else CleanBaselineTrainer
            trainer(args).run_full_experiment()
    summary(flags.out, flags.mp_dir, flags.proto_dir, flags.n_subjects)


if __name__ == '__main__':
    main()
