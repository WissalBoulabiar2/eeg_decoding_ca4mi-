import torch
import torch.nn as nn
from .backbones import EEGNet
from .backbones import DeepConvNet as DCN
from .backbones import ShallowConvNet as SCN


class FeatureExtractor(torch.nn.Module):
    def __init__(self, args):
        super(FeatureExtractor, self).__init__()
        _, nchans, ntimes = args.inputsize
        latent_dim = args.latent_dim
        if args.model == 'EEGNet':
            self.model = EEGNet(args)
        elif args.model == 'SCN':
            self.model = SCN(args)
        elif args.model == 'DCN':
            self.model = DCN(args, latent_dim=args.latent_dim, kernel_1=64, kernel_2=16,
                             dropout=args.dropout, block_out_channels=[25, 25, 50, 100, 200])
        else:
            raise ValueError(f"Unknown model type: {args.model}")

    def forward(self, x):
        features = self.model(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)

        return features



class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        _, nchans, ntimes = args.inputsize

        if hasattr(args, 'task_cls'):
            self.task_cls = args.task_cls
        elif hasattr(args, 'n_subjects') and hasattr(args, 'n_classes'):
            self.task_cls = [[s, args.n_classes] for s in range(args.n_subjects)]
        else:
            num_subjects = getattr(args, 'n_subjects', 9)
            num_classes = getattr(args, 'n_classes', 2)
            self.task_cls = [[s, num_classes] for s in range(num_subjects)]

        self.latent_dim = args.latent_dim
        self.num_tasks = len(self.task_cls)
        self.hidden1 = args.hidden_dim[0]
        self.hidden2 = args.hidden_dim[1]

        self.FeatureExtractor = FeatureExtractor(args)

        self.feature_dropout = nn.Dropout(p=getattr(args, 'feature_dropout', 0.2))

        self.head = torch.nn.ModuleList()
        for i in range(self.num_tasks):
            if args.model == 'DCN':
                input_dim = self.latent_dim * 1
            else:
                input_dim = self.latent_dim

            self.head.append(
                torch.nn.Sequential(
                    torch.nn.Linear(input_dim, self.hidden1),
                    torch.nn.ELU(inplace=True),
                    torch.nn.Dropout(),
                    torch.nn.Linear(self.hidden1, self.hidden2),
                    torch.nn.ELU(inplace=True),
                    torch.nn.Linear(self.hidden2, self.task_cls[i][1])
                ))

    def forward(self, x, task_id=None):
        """
        Forward pass for MuDvi usage
        """
        features = self.FeatureExtractor(x)
        features = features.view(features.size(0), -1)
        features = self.feature_dropout(features)

        if task_id is not None:
            if task_id >= len(self.head):
                raise ValueError(f"Task {task_id} head not available. Only {len(self.head)} heads exist.")

            logits = self.head[task_id](features)
            return {
                'features': features,
                'logits': logits
            }
        else:
            # return features only
            return features


    def print_model_size(self):
        count_F = sum(p.numel() for p in self.FeatureExtractor.parameters() if p.requires_grad)
        count_H = sum(p.numel() for p in self.head.parameters() if p.requires_grad)

        print('Num parameters in F       = %s,  per subject = %s ' % (
            self.pretty_print(count_F), self.pretty_print(count_F / self.num_tasks)))
        print('Num parameters in H       = %s,  per subject = %s ' % (
            self.pretty_print(count_H), self.pretty_print(count_H / self.num_tasks)))
        print('Num parameters in F+H     = %s ' % self.pretty_print(count_F + count_H))
        print('-------------------------->   Total architecture size: %s parameters (%sB)' %
              (self.pretty_print(count_F + count_H), self.pretty_print(4 * (count_F + count_H))))

    def pretty_print(self, num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

    def get_encoded_ftrs(self, x_s, x_p=None, task_id=None):
        """Extract features for compatibility"""
        features = self.FeatureExtractor(x_s)
        features = features.view(features.size(0), -1)
        return features, None

