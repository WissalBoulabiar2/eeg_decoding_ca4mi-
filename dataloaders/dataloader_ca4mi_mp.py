"""
Dataloader for CA4MI-MP with replay_fix='yes'.

Same data, subject order and stratified split as dataloader_ca4mi, but each subject's dataset contains only
that subject's own trials. The original loader appends the replay memory to the dataset *before* splitting,
so replayed trials (all labelled 0, see README_MP.md) end up in the current subject's validation and test sets.
Here the replay buffer is handled by EEGTrainerMP after IEA alignment and is added to the training split only.
"""
from torch.utils.data import Subset, DataLoader
from dataloaders.dataloader_ca4mi import EEGData_ca4mi, IncrementalDataStreaming


class IncrementalDataStreamingMP(IncrementalDataStreaming):

    def __init__(self, args):
        super().__init__(args)
        # subject_order: optional list of subject numbers (e.g. [4, 5, 7, 3, 9, 1, 2, 6, 8]); default S1 -> SN
        order = args.get('subject_order', None)
        if order:
            order = [int(s) for s in order]
            assert sorted(order) == list(range(1, args.n_subjects + 1)), f"invalid subject_order {order}"
            self.subject_queue = order
            print(f"  Subject order: {['S%d' % s for s in order]}")

    def load_data(self, sub_id):
        self.dataloaders[sub_id] = {}
        self.train_set[sub_id] = EEGData_ca4mi(
            root=self.root, args=self.args, sub=self.subject_queue[sub_id], memory=None, sub_id=sub_id)

        train_indices, valid_indices, test_indices = self.stratified_split(
            self.train_set[sub_id], self.pc_valid, self.pc_test, random_state=self.seed + sub_id)

        for split, indices in [('train', train_indices), ('valid', valid_indices), ('test', test_indices)]:
            # Unshuffled: EEGTrainer.align_data rebuilds the loaders (and shuffles the training one)
            self.dataloaders[sub_id][split] = DataLoader(
                Subset(self.train_set[sub_id], indices), batch_size=self.batch_size, shuffle=False,
                num_workers=self.n_workers, pin_memory=self.pin_memory)
        self.dataloaders[sub_id]['name'] = f'BCI-subject{self.subject_queue[sub_id]}'
        self.dataloaders[sub_id]['batch_size'] = self.batch_size

        print(f"  Training set size:      {len(train_indices)}  own trials")
        print(f"  Validation set size:    {len(valid_indices)}  own trials")
        print(f"  Test set size:          {len(test_indices)}  own trials")
        return self.dataloaders
