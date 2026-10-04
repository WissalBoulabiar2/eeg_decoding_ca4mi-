import torch
from .backbones import EEGNet, DeepConvNet as DCN, ShallowConvNet as SCN

class Cls_HEAD(torch.nn.Module):
    """
    Unified classification head with shared and private encoders.
    Supports optional prototype feature integration.
    """

    def __init__(self, args):
        super(Cls_HEAD, self).__init__()
        _, nchans, ntimes = args.inputsize
        self.sub_cls = [[s, args.n_classes] for s in range(args.n_subjects)]
        self.latent_dim = args.latent_dim
        self.num_subjects = args.n_subjects
        self.hidden1 = args.hidden_dim[0]
        self.hidden2 = args.hidden_dim[1]
        self.samples = args.n_samples

        # Prototype configuration
        self.use_prototypes = getattr(args, 'use_prototypes', 'no') == 'yes'
        self.prototype_dim = getattr(args, 'prototype_dim', 64) if self.use_prototypes else 0

        # Shared and private feature extractors
        self.shared = Shared(args)
        self.private = Private(args)

        self.input_dim = self.latent_dim * 2

        # Create classification heads for each subject
        self.head = torch.nn.ModuleList()
        for i in range(self.num_subjects):
            self.head.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.input_dim, self.hidden1),
                    torch.nn.ELU(inplace=True),
                    torch.nn.Dropout(),
                    torch.nn.Linear(self.hidden1, self.hidden2),
                    torch.nn.ELU(inplace=True),
                    torch.nn.Linear(self.hidden2, self.sub_cls[i][1])
                )
            )

    def get_encoded(self, x_s, x_p, sub_id):
        """Get encoded features from shared and private encoders."""
        shared_encoded = self.shared(x_s)
        private_encoded = self.private(x_p, sub_id)
        return shared_encoded, private_encoded

    def forward(self, x_s, x_p, sub_module_label, sub_id):
        """Forward pass with optional prototype features."""
        # Encode features
        x_s = self.shared(x_s)
        x_p = self.private(x_p, sub_id)
        x = torch.cat([x_p, x_s], dim=1)
        return torch.stack([self.head[sub_module_label[i]].forward(x[i]) for i in range(x.size(0))])

    def print_model_size(self):
        count_P = sum(p.numel() for p in self.private.parameters() if p.requires_grad)
        count_S = sum(p.numel() for p in self.shared.parameters() if p.requires_grad)
        count_H = sum(p.numel() for p in self.head.parameters() if p.requires_grad)

        print('Num parameters in Shared       = %s ' % (self.pretty_print(count_S)))
        print('Num parameters in Private       = %s,  per subject = %s ' % (
            self.pretty_print(count_P), self.pretty_print(count_P / self.num_subjects)))
        print('Num parameters in Head       = %s,  per subject = %s ' % (
            self.pretty_print(count_H), self.pretty_print(count_H / self.num_subjects)))
        print('Total architecture size: %s parameters (%sB)' % (
            self.pretty_print(count_S + count_P + count_H),
            self.pretty_print(4 * (count_S + count_P + count_H))))

    def pretty_print(self, num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


class Private(torch.nn.Module):
    def __init__(self, args):
        super(Private, self).__init__()
        self.out = torch.nn.ModuleList([self._get_model(args) for _ in range(args.n_subjects)])
    def _get_model(self, args):
        if args.model == 'EEGNet':
            return EEGNet(args)
        elif args.model == 'SCN':
            return SCN(args)
        elif args.model == 'DCN':
            return DCN(args, latent_dim=args.latent_dim, kernel_1=64, kernel_2=16,
                       dropout=args.dropout, block_out_channels=[25, 25, 50, 100, 200])
        else:
            raise ValueError(f"Unsupported model type: {args.model}")

    def forward(self, x_p, sub_id):
        x_p = self.out[sub_id](x_p)
        return x_p.view(x_p.size(0), -1)


class Shared(torch.nn.Module):
    def __init__(self, args):
        super(Shared, self).__init__()
        self.model = self._get_model(args)

    def _get_model(self, args):
        if args.model == 'EEGNet':
            return EEGNet(args)
        elif args.model == 'SCN':
            return SCN(args)
        elif args.model == 'DCN':
            return DCN(args, latent_dim=args.latent_dim, kernel_1=64, kernel_2=16,
                       dropout=args.dropout, block_out_channels=[25, 25, 50, 100, 200])
        else:
            raise ValueError(f"Unsupported model type: {args.model}")

    def forward(self, x_s):
        return self.model(x_s)



