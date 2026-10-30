"""
Entry point for ProtoMem (methods/protomem.py). Same data protocol as CA4MI-MP and the clean benchmark:
subject order S1 -> S9, identical train/valid/test trials (dataloaders/dataloader_ca4mi_mp.py).
Euclidean alignment is per subject with the reference computed on that subject's TRAINING trials only.

    python main/main_protomem.py --config ./configs/protomem.yml --n_runs 3 --set data_dir=/path/
Matrices: <checkpoint>/protomem_9_subjects_seed_<s>/{accuracy_matrix,f1_macro_matrix,val_accuracy_matrix}.txt
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import random
import numpy as np
import torch
from omegaconf import OmegaConf
from dataloaders.dataloader_ca4mi_mp import IncrementalDataStreamingMP
from methods.protomem import ProtoMem, enabled


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_arrays(loader):
    xs, ys = [], []
    for batch in loader:
        xs.append(np.asarray(batch[0], dtype=np.float32))
        ys.append(np.asarray(batch[1]))
    return np.concatenate(xs), np.concatenate(ys).astype(np.int64)


def ea_matrix(x):
    """R^{-1/2} with R the mean trial covariance (Euclidean alignment, He & Wu 2019)."""
    xc = x - x.mean(-1, keepdims=True)
    r = np.einsum('nct,ndt->cd', xc, xc) / (x.shape[0] * (x.shape[-1] - 1))
    w, v = np.linalg.eigh(r)
    return (v / np.sqrt(np.clip(w, 1e-10, None))) @ v.T


def load_subject(args, loader, t):
    """Aligned (x_tr, y_tr, x_va, y_va, x_te, y_te) tensors of the t-th subject of the stream."""
    d = loader.load_data(t)[t]
    x_tr, y_tr = to_arrays(d['train'])
    x_va, y_va = to_arrays(d['valid'])
    x_te, y_te = to_arrays(d['test'])
    if enabled(args, 'ea', 'yes'):
        w = ea_matrix(x_tr)
        x_tr, x_va, x_te = (np.einsum('cd,ndt->nct', w, z).astype(np.float32) for z in (x_tr, x_va, x_te))
    return d['name'], tuple(torch.as_tensor(a, device=args.device) for a in (x_tr, y_tr, x_va, y_va, x_te, y_te))


def run_reference(args, seed, mode):
    """Reference points without continual learning (same network, alignment and training schedule):
    mode='isolated': one fresh model per subject, trained and tested on that subject only (cannot forget);
    mode='joint': one model trained once on all subjects' training trials, selected on all validation trials.
    The matrix row r holds the accuracy of subjects 0..r, so ACC is computed exactly as for the incremental runs
    (BWT and forgetting are 0 by construction)."""
    args.seed = seed
    set_seed(seed)
    loader = IncrementalDataStreamingMP(args)
    n = args.n_subjects
    data = [load_subject(args, loader, t)[1] for t in range(n)]
    acc_u, f1_u, val_u = np.zeros(n), np.zeros(n), np.zeros(n)
    if mode == 'isolated':
        for t in range(n):
            set_seed(seed * 100 + t)
            method = ProtoMem(args)
            x_tr, y_tr, x_va, y_va, x_te, y_te = data[t]
            print(f"--- isolated model, subject {t + 1}/{n}")
            method.train_subject(0, x_tr, y_tr, x_va, y_va)
            acc_u[t], f1_u[t] = method.evaluate(x_te, y_te, 0)
            val_u[t] = method.evaluate(x_va, y_va, 0)[0]
    else:
        method = ProtoMem(args)
        cat = lambda k: torch.cat([d[k] for d in data])
        print(f"--- joint model on {n} subjects")
        method.train_subject(0, cat(0), cat(1), cat(2), cat(3))
        for t in range(n):
            acc_u[t], f1_u[t] = method.evaluate(data[t][4], data[t][5], 0)
            val_u[t] = method.evaluate(data[t][2], data[t][3], 0)[0]
    acc, f1, val = (np.tril(np.tile(v, (n, 1))).astype(np.float32) for v in (acc_u, f1_u, val_u))
    out = os.path.join(args.checkpoint, f"{args.experiment}_{n}_subjects_seed_{seed}")
    os.makedirs(out, exist_ok=True)
    np.savetxt(os.path.join(out, 'accuracy_matrix.txt'), acc, '%.6f')
    np.savetxt(os.path.join(out, 'f1_macro_matrix.txt'), f1, '%.6f')
    np.savetxt(os.path.join(out, 'val_accuracy_matrix.txt'), val, '%.6f')
    print(f"seed {seed} [{mode}]: ACC={acc_u.mean():.2f}  -> {out}")
    return acc_u.mean(), 0.0


def run(args, seed):
    mode = args.get('mode', 'incremental')
    if mode in ('isolated', 'joint'):
        return run_reference(args, seed, mode)
    args.seed = seed
    set_seed(seed)
    loader = IncrementalDataStreamingMP(args)
    method = ProtoMem(args)
    n = args.n_subjects
    acc, f1, val = (np.zeros((n, n), dtype=np.float32) for _ in range(3))
    data = {}
    dev = args.device
    for t in range(n):
        d = loader.load_data(t)[t]
        x_tr, y_tr = to_arrays(d['train'])
        x_va, y_va = to_arrays(d['valid'])
        x_te, y_te = to_arrays(d['test'])
        if enabled(args, 'ea', 'yes'):
            w = ea_matrix(x_tr)
            x_tr, x_va, x_te = (np.einsum('cd,ndt->nct', w, z).astype(np.float32) for z in (x_tr, x_va, x_te))
        to = lambda a: torch.as_tensor(a, device=dev)
        data[t] = tuple(map(to, (x_tr, y_tr, x_va, y_va, x_te, y_te)))
        x_tr, y_tr, x_va, y_va, x_te, y_te = data[t]

        print(f"--- subject {t + 1}/{n} ({d['name']})")
        method.train_subject(t, x_tr, y_tr, x_va, y_va)
        method.consolidate(t, x_tr, y_tr)
        method.end_subject(t, x_tr, y_tr)
        for u in range(t + 1):
            acc[t, u], f1[t, u] = method.evaluate(data[u][4], data[u][5], u)
            val[t, u] = method.evaluate(data[u][2], data[u][3], u)[0]
        print("  test acc: " + " ".join(f"{a:5.1f}" for a in acc[t, :t + 1]))

    out = os.path.join(args.checkpoint, f"{args.experiment}_{n}_subjects_seed_{seed}")
    os.makedirs(out, exist_ok=True)
    np.savetxt(os.path.join(out, 'accuracy_matrix.txt'), acc, '%.6f')
    np.savetxt(os.path.join(out, 'f1_macro_matrix.txt'), f1, '%.6f')
    np.savetxt(os.path.join(out, 'val_accuracy_matrix.txt'), val, '%.6f')
    if getattr(method, 'diag', None):       # drift-approximation diagnostic (methods/protomem.py)
        import json
        with open(os.path.join(out, 'drift_diag.json'), 'w') as fh:
            json.dump(method.diag, fh, indent=1)
    if getattr(method, 'mode_diag', None):  # multimodality of the (subject, class) embeddings
        import json
        with open(os.path.join(out, 'mode_diag.json'), 'w') as fh:
            json.dump(method.mode_diag, fh, indent=1)
    bwt = np.sum(acc[-1] - np.diag(acc)) / (n - 1)
    val_bwt = np.sum(val[-1] - np.diag(val)) / (n - 1)
    print(f"seed {seed}: ACC={acc[-1].mean():.2f}  BWT={bwt:.2f}  F1={f1[-1].mean():.2f}  "
          f"valACC={val[-1].mean():.2f}  valBWT={val_bwt:.2f}  -> {out}")
    return acc[-1].mean(), bwt


def main():
    parser = argparse.ArgumentParser(description='ProtoMem: subject-incremental MI-EEG decoding')
    parser.add_argument('--config', type=str, default='./configs/protomem.yml')
    parser.add_argument('--n_runs', type=int)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE')
    flags = parser.parse_args()
    args = OmegaConf.merge(OmegaConf.load(flags.config), OmegaConf.from_dotlist(flags.set))
    if flags.n_runs is not None:
        args.n_runs = flags.n_runs
    if flags.checkpoint is not None:
        args.checkpoint = flags.checkpoint
    print(OmegaConf.to_yaml(args))
    res = [run(args, r + int(args.get('seed_offset', 0))) for r in range(int(args.n_runs))]
    res = np.array(res)
    print(f"\nProtoMem over {len(res)} seeds: ACC={res[:, 0].mean():.2f} ± {res[:, 0].std():.2f}  "
          f"BWT={res[:, 1].mean():.2f} ± {res[:, 1].std():.2f}")


if __name__ == '__main__':
    main()
