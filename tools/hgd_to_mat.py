"""
High-Gamma dataset (Schirrmeister et al., 2017) -> the A<s>.mat format of BCI-IV 2a used by every method:
22 sensorimotor channels, 0.5-100 Hz, 250 Hz, 0-4 s after the cue (1000 samples), left vs right hand (labels 0 / 1),
the first 72 trials of each class of the training session (144 trials per subject, as BCI-IV 2a).

Each raw EDF is downloaded from GIN, converted, and deleted, so only one raw file is on disk at a time.

    python tools/hgd_to_mat.py --out hgd_data
"""
import argparse
import os
import time
import urllib.request
import numpy as np
import scipy.io

URL = "https://gin.g-node.org/robintibor/high-gamma-dataset/raw/master/data/train/{}.edf"
PREF = ['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'FCC3h', 'FCC4h', 'CCP3h', 'CCP4h', 'FCC1h', 'FCC2h', 'CCP1h', 'CCP2h', 'FC5', 'FC6', 'CP5', 'CP6']


def convert(edf, n_per_class=72):
    import mne
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_edf(edf, preload=False)
    names = {c.replace('EEG ', '').strip(): c for c in raw.ch_names}
    chans = [names[c] for c in PREF if c in names][:22]
    assert len(chans) == 22, f"only {len(chans)} motor channels found: {raw.ch_names[:20]}"
    raw.pick(chans)
    raw.load_data()
    raw.filter(0.5, 100.0)
    sf = int(raw.info['sfreq'])
    events, ev_id = mne.events_from_annotations(raw)
    code = {v: (0 if 'left' in k.lower() else 1) for k, v in ev_id.items()
            if 'left' in k.lower() or 'right' in k.lower()}
    assert len(code) == 2, f"left/right events not found: {ev_id}"
    x = raw.get_data() * 1e6                      # volts -> microvolts, as BCI-IV 2a
    n_win, step = 4 * sf, sf // 250
    X, y, count = [], [], {0: 0, 1: 0}
    for onset, _, c in events:                    # chronological order
        if c in code and count[code[c]] < n_per_class and onset + n_win <= x.shape[1]:
            X.append(x[:, onset:onset + n_win:step])
            y.append(code[c])
            count[code[c]] += 1
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64), sf, ev_id, chans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="hgd_data")
    p.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 15)))
    flags = p.parse_args()
    os.makedirs(flags.out, exist_ok=True)
    for s in flags.subjects:
        dst = os.path.join(flags.out, f"A{s}.mat")
        if os.path.exists(dst):
            continue
        t = time.time()
        edf = os.path.join(flags.out, f"_tmp_{s}.edf")
        req = urllib.request.Request(URL.format(s), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(edf, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        X, y, sf, ev_id, chans = convert(edf)
        if s == flags.subjects[0]:
            print("sfreq", sf, "| events", ev_id, "| channels", chans)
        scipy.io.savemat(dst, {'trainX': np.transpose(X, (2, 1, 0)), 'trainY': y[None, :]})
        os.remove(edf)
        print(f"subject {s}: {X.shape} classes {np.bincount(y)} ({time.time() - t:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
