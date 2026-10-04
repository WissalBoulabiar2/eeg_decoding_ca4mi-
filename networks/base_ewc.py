import torch
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
            self.model = DCN(args, latent_dim=latent_dim, kernel_1=64, kernel_2=16, dropout=args.dropout,
                             block_out_channels=[25, 25, 50, 100, 200])

    def forward(self, x):
        return self.model(x)


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        _, nchans, ntimes = args.inputsize
        self.task_cls = args.task_cls
        self.latent_dim = args.latent_dim
        self.num_tasks = args.n_subjects
        self.hidden1 = args.hidden_dim[0]
        self.hidden2 = args.hidden_dim[1]
        self.num_classes = args.n_classes

        self.FeatureExtractor = FeatureExtractor(args)
        self.classifiers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.latent_dim * 1, self.hidden1),
                torch.nn.ELU(inplace=True),
                torch.nn.Dropout(),
                torch.nn.Linear(self.hidden1, self.hidden2),
                torch.nn.ELU(inplace=True),
                torch.nn.Linear(self.hidden2, self.num_classes)
            ) for _ in range(self.num_tasks)
        ])

        if args.model == 'EEGNet' or args.model == 'SCN':
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.latent_dim * 2, self.hidden1),
                torch.nn.ELU(inplace=True),
                torch.nn.Dropout(),
                torch.nn.Linear(self.hidden1, self.hidden2),
                torch.nn.ELU(inplace=True),
                torch.nn.Linear(self.hidden2, self.num_classes)
            )
        elif args.model == 'DCN':
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.latent_dim * 1, self.hidden1),
                torch.nn.ELU(inplace=True),
                torch.nn.Dropout(),
                torch.nn.Linear(self.hidden1, self.hidden2),
                torch.nn.ELU(inplace=True),
                torch.nn.Linear(self.hidden2, self.num_classes)
            )

    def forward(self, x, task_id=None):
        features = self.FeatureExtractor(x)
        features = features.view(features.size(0), -1)
        if task_id is not None:
            outputs = self.classifiers[task_id](features)
        else:
            outputs = self.classifiers[0](features)
        return outputs

    def print_model_size(self):
        count_F = sum(p.numel() for p in self.FeatureExtractor.parameters() if p.requires_grad)
        count_C = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)

        print('Num parameters in Feature Extractor = %s' % self.pretty_print(count_F))
        print('Num parameters in Classifier       = %s' % self.pretty_print(count_C))
        print('Num parameters in F+C              = %s' % self.pretty_print(count_F + count_C))
        print('-------------------------->   Total architecture size: %s parameters (%sB)' %
              (self.pretty_print(count_F + count_C), self.pretty_print(4 * (count_F + count_C))))

    def pretty_print(self, num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])