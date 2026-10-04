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
            self.model = DCN(args, latent_dim=latent_dim, kernel_1=64, kernel_2=16,
                             dropout=args.dropout, block_out_channels=[25, 25, 50, 100, 200])
        if args.dataset == "bci-competition-IV2a":
            self.fc = nn.Linear(800, 64)
        elif args.dataset == "bci-competition-IV2b":
            self.fc = nn.Linear(600, 64)
        elif args.dataset == "openBMI":
            self.fc = nn.Linear(800, 64)

    def forward(self, x):

        if isinstance(self.model, DCN):
            x = x.float()
            x = self.model.first_conv_block(x)
            for block in self.model.deep_block:
                x = block(x)

            x = x.view(x.size(0), -1)

            if x.size(1) != self.fc.in_features:
                raise ValueError(f"Input to fc has shape {x.shape}, expected {self.fc.in_features}")

            x = self.fc(x)
            return x

        features = self.model(x)

        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        return features


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        ncha, size, _ = args.inputsize
        self.task_cls = args.task_cls
        self.latent_dim = args.latent_dim
        self.num_tasks = args.n_subjects
        self.hidden1 = args.hidden_dim[0]
        self.hidden2 = args.hidden_dim[1]

        self.FeatureExtractor = FeatureExtractor(args)
        if args.model == 'EEGNet' or args.model == 'SCN':
            self.head = torch.nn.ModuleList()
            for i in range(self.num_tasks):
                self.head.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(self.latent_dim * 2, self.hidden1),
                        torch.nn.ELU(inplace=True),
                        torch.nn.Dropout(),
                        torch.nn.Linear(self.hidden1, self.hidden2),
                        torch.nn.ELU(inplace=True),
                        torch.nn.Linear(self.hidden2, self.task_cls[i][1])
                    ))
        elif args.model == 'DCN':
            self.head = torch.nn.ModuleList()
            for i in range(self.num_tasks):
                self.head.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(self.latent_dim * 1, self.hidden1),
                        torch.nn.ELU(inplace=True),
                        torch.nn.Dropout(),
                        torch.nn.Linear(self.hidden1, self.hidden2),
                        torch.nn.ELU(inplace=True),
                        torch.nn.Linear(self.hidden2, self.task_cls[i][1])
                    ))

    def forward(self, x, task_id):
        features = self.FeatureExtractor(x)
        features = features.view(features.size(0), -1)

        return self.head[task_id](features)

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

