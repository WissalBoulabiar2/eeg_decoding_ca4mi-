"""
Clean benchmark protocol for the baselines: the same data protocol as CA4MI-MP (dataloader_ca4mi_mp).
The baseline methods themselves are not modified; only the data they receive is.

For every method:
  - subject order S1 -> S9 (the original baseline loaders draw a random permutation),
  - identical train / valid / test trials: same loading (2 classes, no shuffling) and the same stratified split
    with random_state = seed + sub_id as dataloader_ca4mi_mp,
  - replay memory (replay methods, use_memory='yes'): true labels, sampled from the current subject's own training
    trials, added to the training split only (the original loader labels every replayed trial 0 and splits the
    memory into the validation and test sets of the current subject).
Each method keeps its own memory size (n_samples) and other settings from its config.
"""
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader
from dataloaders import dataloader_replay as replay
from dataloaders import dataloader_baseline as baseline


class _MemoryDataset(Dataset):
    """Snapshot of the replay memory, same item format as dataloader_replay.EEGData."""

    def __init__(self, memory):
        self.memory = {k: list(v) for k, v in memory.items()}

    def __getitem__(self, i):
        m = self.memory
        return m['x'][i], int(m['y'][i]), m['sub_module_label'][i], m['dis_label'][i]

    def __len__(self):
        return len(self.memory['x'])


class _TwoTuple(Dataset):
    """(data, target) items, the format of dataloader_baseline.EEGData."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, i):
        x, y = self.dataset[i][:2]
        return x, y

    def __len__(self):
        return len(self.dataset)


def _own_subject(streaming, sub_id):
    """Current subject's own trials (no memory, no shuffling) and the shared split."""
    own = replay.EEGData(root=streaming.root, args=streaming.args, sub=streaming.subject_queue[sub_id],
                         memory=None, sub_id=sub_id)
    split = replay.IncrementalDataStreaming.stratified_split(
        streaming, own, streaming.pc_valid, streaming.pc_test, random_state=streaming.seed + sub_id)
    return own, split


def _print_sizes(train, valid, test, n_replay):
    print(f"Training set size:      {len(train)}  ({len(train) - n_replay} own + {n_replay} replayed)")
    print(f"Validation set size:    {len(valid)}  own trials")
    print(f"Test set size:          {len(test)}  own trials")


class CleanReplayStreaming(replay.IncrementalDataStreaming):
    """ER, DER, MUDVI, CGER."""

    def __init__(self, args):
        super().__init__(args)
        self.subject_queue = list(range(1, args.n_subjects + 1))

    def load_data(self, sub_id):
        self.dataloaders[sub_id] = {}
        own, (train_idx, valid_idx, test_idx) = _own_subject(self, sub_id)
        self.train_set[sub_id] = own
        train_own = Subset(own, train_idx)
        self.train_split[sub_id] = train_own
        self.valid_split[sub_id] = Subset(own, valid_idx)
        self.test_set[sub_id] = Subset(own, test_idx)

        memory = _MemoryDataset(self.global_memory) if sub_id > 0 and len(self.global_memory['x']) > 0 else None
        train = ConcatDataset([train_own, memory]) if memory is not None else train_own

        self.dataloaders[sub_id]['train'] = DataLoader(
            train, batch_size=self.batch_size, shuffle=True,
            num_workers=self.n_workers, pin_memory=self.pin_memory, drop_last=False)
        self.dataloaders[sub_id]['valid'] = DataLoader(
            self.valid_split[sub_id], batch_size=int(self.batch_size * self.pc_valid), shuffle=False,
            num_workers=self.n_workers, pin_memory=self.pin_memory, drop_last=False)
        self.dataloaders[sub_id]['test'] = DataLoader(
            self.test_set[sub_id], batch_size=int(self.batch_size), shuffle=False,
            num_workers=self.n_workers, pin_memory=self.pin_memory, drop_last=False)
        self.dataloaders[sub_id]['name'] = f'BCI-subject{self.subject_queue[sub_id]}'
        _print_sizes(train, self.valid_split[sub_id], self.test_set[sub_id], 0 if memory is None else len(memory))

        if self.use_memory == 'yes' and self.n_samples > 0:
            self.update_memory(sub_id)
        return self.dataloaders

    def update_memory(self, sub_id):
        """Same sampling as the original (n_samples random training trials, reservoir), with the true labels."""
        n = self.n_samples // len(self.subject_index[sub_id])
        own_train = self.train_split[sub_id]
        for ind in torch.randperm(len(own_train))[:n]:
            x, y, sub, dis = own_train[int(ind)]
            self.reservoir_sample({'x': x, 'y': int(y), 'sub_module_label': sub, 'dis_label': dis},
                                  self.global_reservoir_size)
        print(f'Global memory updated, retaining {len(self.global_memory["x"])} samples')


class CleanBaselineStreaming(baseline.IncrementalDataStreaming):
    """Finetuning, EWC (no replay)."""

    def __init__(self, args):
        super().__init__(args)
        self.subject_queue = list(range(1, args.n_subjects + 1))

    def load_data(self, sub_id):
        self.dataloaders[sub_id] = {}
        own, (train_idx, valid_idx, test_idx) = _own_subject(self, sub_id)
        own = _TwoTuple(own)
        self.train_set[sub_id] = own
        self.train_split[sub_id] = Subset(own, train_idx)
        self.valid_split[sub_id] = Subset(own, valid_idx)
        self.test_set[sub_id] = Subset(own, test_idx)

        self.dataloaders[sub_id]['train'] = self.create_safe_dataloader(
            self.train_split[sub_id], batch_size=self.batch_size, shuffle=True, drop_last=False, mode='train')
        self.dataloaders[sub_id]['valid'] = self.create_safe_dataloader(
            self.valid_split[sub_id], batch_size=int(self.batch_size * self.pc_valid), mode='valid')
        self.dataloaders[sub_id]['test'] = self.create_safe_dataloader(
            self.test_set[sub_id], batch_size=self.batch_size, mode='test')
        self.dataloaders[sub_id]['name'] = f'BCI-subject{self.subject_queue[sub_id]}'
        _print_sizes(self.train_split[sub_id], self.valid_split[sub_id], self.test_set[sub_id], 0)
        return self.dataloaders
