import scipy.io
import torch.utils.data
import random
import torch
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np


class EEGData_ca4mi:
    def __init__(self, root, args, sub, memory, sub_id, train=True):
        """
        Initialize the unified EEG dataset loader for CA4MI.

        Args:
            root (str): Root directory of the dataset.
            args (Namespace): Arguments object with dataset_name and n_classes attributes.
            sub (int): Subject number.
            memory (dict): Memory object for subject-specific data.
            sub_id (int): Subject ID for labeling.
            train (bool, optional): Whether to load training data. Defaults to True.
        """
        self.root = root
        self.train = train
        self.sub_id = sub_id
        self.dataset_name = args.dataset_name

        # Load data based on dataset name
        if self.dataset_name == 'BCICompIV2a':
            self._load_bci_comp_iv2a(args, sub)
        elif self.dataset_name == 'BCICompIV2b':
            self._load_bci_comp_iv2b(args, sub)
        elif self.dataset_name == 'OpenBMI':
            self._load_openbmi(args, sub)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}. "
                             "Supported datasets: 'BCICompIV2a', 'BCICompIV2b', 'OpenBMI'")

        # Initialize labels
        self.sub_module_label = [self.sub_id] * len(self.data)

        self.dis_label = [self.sub_id + 1] * len(self.data)

        # If training, integrate memory data if available
        if train and memory is not None:
            self._append_memory(memory)


    def _load_bci_comp_iv2a(self, args, sub):
        """Load BCI Competition IV 2a dataset."""
        # Adjusted path for your data directory
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

    def _load_bci_comp_iv2b(self, args, sub):
        """Load BCI Competition IV 2b dataset."""
        data_dict = scipy.io.loadmat(f"{self.root}B{sub}.mat")
        trainX, trainY = np.array(data_dict['trainX']), np.array(data_dict['trainY']).squeeze()
        self.data = np.transpose(trainX, (2, 1, 0))  # Rearrange dimensions
        self.targets = list(trainY)

    def _load_openbmi(self, args, sub):
        """Load OpenBMI dataset."""
        data_dict = scipy.io.loadmat(f"{self.root}s{sub}.mat")
        trainX = np.array(data_dict['X'])
        trainY = np.array(data_dict['Y']).squeeze()

        # Only keep data with labels 0 and 1
        mask = np.isin(trainY, [0, 1])
        self.data = trainX[mask]
        self.targets = trainY[mask].tolist()  # Convert to list for consistency

    def _filter_classes(self, data, targets, classes):
        """Filters data and targets to include only specified classes."""
        mask = [target in classes for target in targets]
        filtered_data = data[mask]
        filtered_targets = [targets[i] for i, m in enumerate(mask) if m]
        return filtered_data, filtered_targets

    def _append_memory(self, memory):
        """Appends memory data to the dataset."""
        for i in range(len(memory['x'])):
            self.data = np.append(self.data, [memory['x'][i]], axis=0)
            self.targets.append(memory['y'][i])
            self.sub_module_label.append(memory['sub_module_label'][i])
            self.dis_label.append(memory['dis_label'][i])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the data point.

        Returns:
            tuple: (data, target, sub_module_label, dis_label) for the given index.
        """
        return (
            self.data[index],
            int(self.targets[index]),
            self.sub_module_label[index],
            self.dis_label[index],
        )

    def __len__(self):
        """Returns the total number of data points."""
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

        self.progressive_config = {
            'enabled': getattr(args, 'progressive_batch', True),
            'min_batch_size': getattr(args, 'min_batch_size', 8),
            'max_batch_size': getattr(args, 'max_batch_size', 128),
            'strategy': getattr(args, 'batch_strategy', 'exponential'),  # 'linear', 'exponential', 'staged'
            'warmup_subjects': getattr(args, 'warmup_subjects', 2),
            'stable_subjects': getattr(args, 'stable_subjects', 3),
        }

        self.sub_cls = [[s, args.n_classes] for s in range(args.n_subjects)]
        # randomize the order of subjects
        # self.subject_queue = np.random.permutation(np.arange(1, args.n_subjects + 1)).tolist()
        self.subject_queue = np.arange(1, args.n_subjects + 1).tolist()
        self.subject_index = [[s] for s in range(args.n_subjects)]
        self.global_reservoir_size = args.global_reservoir_size
        self.dataloaders = {}
        self.train_set = {}
        self.test_set = {}
        self.train_split = {}
        self.valid_split = {}
        self.global_memory = {'x': [], 'y': [], 'sub_module_label': [], 'dis_label': []}

    def get_progressive_batch_size(self, sub_id):
        """
        Calculate progressive batch size based on subject ID
        """
        if not self.progressive_config['enabled']:
            return self.batch_size

        total_subjects = len(self.subject_queue)
        min_batch = self.progressive_config['min_batch_size']
        max_batch = self.progressive_config['max_batch_size']
        base_batch = self.batch_size
        strategy = self.progressive_config['strategy']

        # Ensure that the batch size is within a reasonable range.
        min_batch = max(4, min(min_batch, base_batch))
        max_batch = max(base_batch, max_batch)

        if strategy == 'linear':
            # Linear growth: from min_batch to max_batch
            progress = sub_id / max(total_subjects - 1, 1)
            batch_size = int(min_batch + (max_batch - min_batch) * progress)

        elif strategy == 'exponential':
            # Exponential growth: smoother transition
            if sub_id == 0:
                batch_size = min_batch
            else:
                # Calculate index factors
                max_exp = np.log2(max_batch / min_batch)
                current_exp = (sub_id / max(total_subjects - 1, 1)) * max_exp
                batch_size = int(min_batch * (2 ** current_exp))

        elif strategy == 'staged':
            # Phased strategy
            warmup_subjects = self.progressive_config['warmup_subjects']
            stable_subjects = self.progressive_config['stable_subjects']

            if sub_id < warmup_subjects:
                # Warm-up phase: Use the smallest batch size
                batch_size = min_batch
            elif sub_id < warmup_subjects + stable_subjects:
                # Stable phase: Use the basic batch size.
                batch_size = base_batch
            else:
                #  Acceleration phase: Use large batch sizes
                remaining_subjects = total_subjects - warmup_subjects - stable_subjects
                if remaining_subjects > 0:
                    progress = (sub_id - warmup_subjects - stable_subjects) / remaining_subjects
                    batch_size = int(base_batch + (max_batch - base_batch) * progress)
                else:
                    batch_size = max_batch

        elif strategy == 'sqrt':
            progress = np.sqrt(sub_id / max(total_subjects - 1, 1))
            batch_size = int(min_batch + (max_batch - min_batch) * progress)

        else:  # 'fixed'
            batch_size = base_batch

        # Ensure that the batch size is a reasonable value.
        batch_size = max(min_batch, min(batch_size, max_batch))

        if getattr(self.args, 'power_of_2_batch', False):
            batch_size = 2 ** int(np.log2(batch_size))

        return batch_size

    def get_adaptive_valid_batch_size(self, train_batch_size):
        """
        Adaptively calculate the validation batch size based on the training batch size.
        """
        valid_multiplier = getattr(self.args, 'valid_batch_multiplier', 1.5)
        valid_batch_size = int(train_batch_size * valid_multiplier)

        max_valid_batch = int(self.batch_size * self.pc_valid * 2)
        valid_batch_size = min(valid_batch_size, max_valid_batch)

        return max(4, valid_batch_size)

    def stratified_split(self, dataset, pc_valid, pc_test, random_state=42):
        """
        Use stratified sampling for data partitioning to ensure balanced category distribution in each split.
        """
        labels = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if len(sample) == 2:
                _, label = sample
            elif len(sample) == 3:
                _, label, _ = sample
            elif len(sample) == 4:
                _, label, _, _ = sample
            else:
                label = sample[1]
            labels.append(label)

        labels = np.array(labels)
        indices = np.arange(len(dataset))

        test_size = pc_test
        valid_size = pc_valid / (1 - pc_test)

        train_valid_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            stratify=labels,
            random_state=random_state
        )

        train_valid_labels = labels[train_valid_indices]
        train_indices, valid_indices = train_test_split(
            train_valid_indices,
            test_size=valid_size,
            stratify=train_valid_labels,
            random_state=random_state
        )

        return train_indices, valid_indices, test_indices

    def load_data(self, sub_id):
        """
             Load data for the specified subject using progressive batch size.
        """
        self.dataloaders[sub_id] = {}

        if sub_id == 0:
            memory = None
        else:
            memory = self.global_memory

        self.train_set[sub_id] = EEGData_ca4mi(
            root=self.root,
            args=self.args,
            sub=self.subject_queue[sub_id],
            memory=memory,
            sub_id=sub_id
        )

        train_indices, valid_indices, test_indices = self.stratified_split(
            self.train_set[sub_id],
            self.pc_valid,
            self.pc_test,
            random_state=self.seed + sub_id
        )

        train_split = Subset(self.train_set[sub_id], train_indices)
        valid_split = Subset(self.train_set[sub_id], valid_indices)
        test_split = Subset(self.train_set[sub_id], test_indices)

        self.train_split[sub_id] = train_split
        self.valid_split[sub_id] = valid_split
        self.test_set[sub_id] = test_split

        dynamic_batch_size = self.get_progressive_batch_size(sub_id)
        valid_batch_size = self.get_adaptive_valid_batch_size(dynamic_batch_size)

        max_train_batch = len(train_split) // 2
        max_valid_batch = len(valid_split) // 2 if len(valid_split) > 0 else 1
        max_test_batch = len(test_split) // 2 if len(test_split) > 0 else 1

        dynamic_batch_size = min(dynamic_batch_size, max(1, max_train_batch))
        valid_batch_size = min(valid_batch_size, max(1, max_valid_batch))
        test_batch_size = min(dynamic_batch_size, max(1, max_test_batch))

        train_loader = DataLoader(
            train_split,
            batch_size=dynamic_batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )

        valid_loader = DataLoader(
            valid_split,
            batch_size=valid_batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )

        test_loader = DataLoader(
            test_split,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )

        self.dataloaders[sub_id]['train'] = train_loader
        self.dataloaders[sub_id]['valid'] = valid_loader
        self.dataloaders[sub_id]['test'] = test_loader
        self.dataloaders[sub_id]['name'] = f'BCI-subject{self.subject_queue[sub_id]}'
        self.dataloaders[sub_id]['batch_size'] = dynamic_batch_size


        shape = self.train_set[sub_id].data.shape[:]
        print(f"  Training set size:      {len(train_loader.dataset)}  EEG signals of shape {shape}")
        print(f"  Validation set size:    {len(valid_loader.dataset)}  EEG signals of shape {shape}")
        print(f"  Test set size:          {len(test_loader.dataset)}  EEG signals of shape {shape}")


        if self.use_memory == 'yes' and self.n_samples > 0:
            self.update_memory(sub_id)

        return self.dataloaders

    def update_memory(self, sub_id):
        """
        Update global memory using reservoir sampling to retain knowledge across subjects.
        """
        num_samples_per_subject = self.n_samples // len(self.subject_index[sub_id])
        mem_class_mapping = {i: i for i in range(len(self.subject_index[sub_id]))}
        reservoir_size = self.global_reservoir_size

        if 'x' not in self.global_memory:
            self.global_memory = {'x': [], 'y': [], 'sub_module_label': [], 'dis_label': []}

        data_loader = torch.utils.data.DataLoader(self.train_split[sub_id], batch_size=1,
                                                  num_workers=self.n_workers, pin_memory=self.pin_memory)

        rand_indices = torch.randperm(len(data_loader.dataset))[:num_samples_per_subject]

        for i, ind in enumerate(rand_indices):
            sample = {
                'x': data_loader.dataset[ind][0],
                'y': mem_class_mapping.get(i, 0),
                'sub_module_label': data_loader.dataset[ind][2],
                'dis_label': data_loader.dataset[ind][3]
            }
            self.reservoir_sample(sample, reservoir_size)

        print(f'  Global memory updated: {len(self.global_memory["x"])} samples retained')

    def reservoir_sample(self, sample, reservoir_size):
        """
        Helper function to perform reservoir sampling on global memory.

        Args:
            sample (dict): The sample to consider for adding to memory.
            reservoir_size (int): The maximum size of the reservoir memory.
        """
        if len(self.global_memory['x']) < reservoir_size:
            for key in sample:
                self.global_memory[key].append(sample[key])
        else:
            j = random.randint(0, len(self.global_memory['x']) - 1)
            if random.random() < reservoir_size / (reservoir_size + len(self.global_memory['x'])):
                for key in sample:
                    self.global_memory[key][j] = sample[key]
