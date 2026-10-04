from torch.utils.data import DataLoader, Subset
import sys
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split

class EEGData:
    def __init__(self, root, args, sub, sub_id, train=True):
        self.root = root
        self.train = train
        self.sub_id = sub_id
        self.dataset_name = args.dataset_name

        if self.dataset_name == 'BCICompIV2a':
            self._load_bci_comp_iv2a(args, sub)
        elif self.dataset_name == 'BCICompIV2b':
            self._load_bci_comp_iv2b(args, sub)
        elif self.dataset_name == 'OpenBMI':
            self._load_openbmi(args, sub)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}. "
                             "Supported datasets: 'BCICompIV2a', 'BCICompIV2b', 'OpenBMI'")

    def _load_bci_comp_iv2a(self, args, sub):
        data_dict = scipy.io.loadmat(f"{self.root}A{sub}.mat")
        trainX, trainY = np.array(data_dict['trainX']), np.array(data_dict['trainY']).squeeze()
        raw_data = np.transpose(trainX, (2, 1, 0))
        raw_data = raw_data[:, :, :1000]
        raw_targets = list(trainY)

        if args.n_classes == 2:
            self.data, self.targets = self._filter_classes(raw_data, raw_targets, [0, 1])
        elif args.n_classes == 4:
            self.data, self.targets = raw_data, raw_targets
        else:
            raise ValueError("Unsupported class type for BCICompIV2a. Only 2 and 4 classes are supported.")

        idx = np.arange(len(self.data))
        np.random.shuffle(idx)
        self.data = self.data[idx]
        self.targets = np.array(self.targets)[idx]

    def _load_bci_comp_iv2b(self, args, sub):
        data_dict = scipy.io.loadmat(f"{self.root}B{sub}.mat")
        trainX, trainY = np.array(data_dict['trainX']), np.array(data_dict['trainY']).squeeze()
        self.data = np.transpose(trainX, (2, 1, 0))
        self.targets = list(trainY)

    def _load_openbmi(self, args, sub):
        data_dict = scipy.io.loadmat(f"{self.root}s{sub}.mat")
        trainX = np.array(data_dict['X'])
        trainY = np.array(data_dict['Y']).squeeze()

        mask = np.isin(trainY, [0, 1])
        self.data = trainX[mask]
        self.targets = trainY[mask]

    def _filter_classes(self, data, targets, classes):
        mask = [target in classes for target in targets]
        filtered_data = data[mask]
        filtered_targets = [targets[i] for i, m in enumerate(mask) if m]
        return filtered_data, filtered_targets

    def __getitem__(self, index):
        return (
            self.data[index],
            int(self.targets[index])
        )

    def __len__(self):
        return len(self.data)


class IncrementalDataStreaming(object):
    def __init__(self, args):
        super(IncrementalDataStreaming, self).__init__()
        self.args = args
        self.use_memory = args.use_memory

        self.n_workers = args.n_workers
        self.pin_memory = True
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.pc_valid = args.pc_valid
        self.root = args.data_dir
        self.latent_dim = args.latent_dim
        self.n_samples = args.n_samples
        self.pc_valid = args.pc_valid
        self.pc_test = args.pc_test

        self.sub_cls = [[s, args.n_classes] for s in range(args.n_subjects)]
        # self.subject_queue = np.arange(1, args.n_subjects + 1).tolist()
        # randomize the order of subjects
        self.subject_queue = np.random.permutation(np.arange(1, args.n_subjects + 1)).tolist()
        self.subject_index = [[s] for s in range(args.n_subjects)]
        self.global_reservoir_size = args.global_reservoir_size
        self.dataloaders = {}
        self.train_set = {}
        self.test_set = {}
        self.train_split = {}
        self.valid_split = {}

    def create_stratified_splits(self, dataset, sub_id):
        all_targets = [dataset[i][1] for i in range(len(dataset))]
        all_indices = np.arange(len(dataset))

        unique_labels, counts = np.unique(all_targets, return_counts=True)

        min_samples_needed = 3
        insufficient_classes = []
        for label, count in zip(unique_labels, counts):
            if count < min_samples_needed:
                insufficient_classes.append((label, count))

        if insufficient_classes:
            return self.create_manual_balanced_splits(all_targets, all_indices, sub_id)

        try:
            temp_indices, test_indices, temp_targets, test_targets = train_test_split(
                all_indices, all_targets,
                test_size=self.pc_test,
                stratify=all_targets,
                random_state=self.seed + sub_id
            )

            adjusted_valid_ratio = self.pc_valid / (1 - self.pc_test)
            train_indices, valid_indices, train_targets, valid_targets = train_test_split(
                temp_indices, temp_targets,
                test_size=adjusted_valid_ratio,
                stratify=temp_targets,
                random_state=self.seed + sub_id + 1
            )

            return train_indices, valid_indices, test_indices

        except ValueError as e:
            return self.create_manual_balanced_splits(all_targets, all_indices, sub_id)

    def create_manual_balanced_splits(self, all_targets, all_indices, sub_id):
        from collections import defaultdict

        class_indices = defaultdict(list)
        for idx, target in zip(all_indices, all_targets):
            class_indices[target].append(idx)

        train_indices, valid_indices, test_indices = [], [], []

        for label, indices in class_indices.items():
            indices = np.array(indices)
            np.random.seed(self.seed + sub_id)
            np.random.shuffle(indices)

            n_samples = len(indices)

            n_test = max(1, int(n_samples * self.pc_test))
            n_valid = max(1, int(n_samples * self.pc_valid))
            n_train = n_samples - n_test - n_valid

            if n_train < 1:
                n_train = 1
                remaining = n_samples - n_train
                n_test = max(1, remaining // 2)
                n_valid = remaining - n_test

            test_indices.extend(indices[:n_test])
            valid_indices.extend(indices[n_test:n_test + n_valid])
            train_indices.extend(indices[n_test + n_valid:])

        np.random.seed(self.seed + sub_id)
        np.random.shuffle(train_indices)
        np.random.shuffle(valid_indices)
        np.random.shuffle(test_indices)

        return train_indices, valid_indices, test_indices

    def create_safe_dataloader(self, subset, batch_size, shuffle=False, drop_last=False, mode='train'):
        dataset_size = len(subset)

        if mode in ['test', 'valid']:
            effective_batch_size = min(batch_size, dataset_size)
            drop_last = False
            shuffle = False
        else:
            effective_batch_size = batch_size
            drop_last = drop_last and (dataset_size > batch_size)

        return DataLoader(
            subset,
            batch_size=effective_batch_size,
            shuffle=shuffle,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last
        )

    def load_data(self, sub_id):
        self.dataloaders[sub_id] = {}
        sys.stdout.flush()

        self.train_set[sub_id] = EEGData(
            root=self.root,
            args=self.args,
            sub=self.subject_queue[sub_id],
            sub_id=sub_id
        )

        train_indices, valid_indices, test_indices = self.create_stratified_splits(
            self.train_set[sub_id], sub_id
        )

        train_split = Subset(self.train_set[sub_id], train_indices)
        valid_split = Subset(self.train_set[sub_id], valid_indices)
        test_split = Subset(self.train_set[sub_id], test_indices)

        self.train_split[sub_id] = train_split
        self.valid_split[sub_id] = valid_split
        self.test_set[sub_id] = test_split

        train_loader = self.create_safe_dataloader(
            train_split,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            mode='train'
        )

        valid_loader = self.create_safe_dataloader(
            valid_split,
            batch_size=int(self.batch_size * self.pc_valid),
            shuffle=False,
            drop_last=False,
            mode='valid'
        )

        test_loader = self.create_safe_dataloader(
            test_split,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            mode='test'
        )

        self.dataloaders[sub_id]['train'] = train_loader
        self.dataloaders[sub_id]['valid'] = valid_loader
        self.dataloaders[sub_id]['test'] = test_loader
        self.dataloaders[sub_id]['name'] = f'BCI-subject{self.subject_queue[sub_id]}'

        shape = self.train_set[sub_id].data.shape[:]
        print(f"Training set size:      {len(train_loader.dataset)}  EEG signals of shape {shape}")
        print(f"Validation set size:    {len(valid_loader.dataset)}  EEG signals of shape {shape}")
        print(f"Test set size:          {len(test_loader.dataset)}  EEG signals of shape {shape}")

        return self.dataloaders