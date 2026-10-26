"""
ProtoMem network: one DeepConvNet encoder with subject-specific BatchNorm and a single multi-proxy cosine
classifier shared by all subjects.

- Subject-specific BatchNorm (domain-specific BN, Chang et al., CVPR 2019): every BatchNorm2d of the encoder is
  replaced by one BN per subject. The BN of the subject being learned is trained normally; the BN of a past subject
  is frozen (running statistics and affine parameters), so replayed trials of subject u are encoded exactly as
  they are at test time for subject u.
- Multi-proxy cosine classifier: K learnable proxies per class on the unit sphere,
  logit_c = logsumexp_k( -||z - p_ck||^2 / tau ), with z the L2-normalised embedding.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import DeepConvNet


class SubjectBatchNorm(nn.Module):
    def __init__(self, bn, n_subjects):
        super().__init__()
        self.bns = nn.ModuleList([copy.deepcopy(bn) for _ in range(n_subjects)])
        self.active = 0   # subject whose statistics are used for the current forward
        self.current = 0  # subject being learned (the only one whose BN is trained)

    def forward(self, x):
        bn = self.bns[self.active]
        if self.training and self.active == self.current:
            return bn(x)
        return F.batch_norm(x, bn.running_mean, bn.running_var, bn.weight, bn.bias, False, 0.0, bn.eps)


def _replace_bn(module, n_subjects):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, SubjectBatchNorm(child, n_subjects))
        else:
            _replace_bn(child, n_subjects)


class ProtoMemNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = DeepConvNet(args, latent_dim=args.latent_dim, kernel_1=64, kernel_2=16,
                                   dropout=args.dropout, block_out_channels=[25, 25, 50, 100, 200])
        _replace_bn(self.encoder, args.n_subjects)
        self.sbns = [m for m in self.encoder.modules() if isinstance(m, SubjectBatchNorm)]
        self.proxies = nn.Parameter(0.1 * torch.randn(args.n_classes, args.n_proxies, args.latent_dim))
        self.tau = float(args.tau)
        # subject_bn='no': control with one BN shared by all subjects (always trained, replayed trials included)
        self.shared_bn = args.get('subject_bn', 'yes') not in ('yes', True)
        # classifier='linear': control with a standard linear softmax head on the unnormalized embedding,
        # instead of the L2-normalized embedding and cosine proxies
        self.linear = args.get('classifier', 'proxy') == 'linear'
        if self.linear:
            self.fc = nn.Linear(args.latent_dim, args.n_classes)

    def start_subject(self, t, freeze_past=True):
        """Warm-start the BN of subject t from subject t-1. Past subjects always keep their running statistics;
        with freeze_past their affine parameters are frozen too, otherwise replay can still adapt them."""
        if self.shared_bn:
            return
        for m in self.sbns:
            if t > 0:
                m.bns[t].load_state_dict(m.bns[t - 1].state_dict())
            for u in range(t):
                for p in m.bns[u].parameters():
                    p.requires_grad_(not freeze_past)
            m.current = t

    def embed(self, x, sub):
        """x: [N, C, T] trials, sub: [N] subject ids. Each subject group goes through its own BN."""
        x = x[:, None].float()
        if self.shared_bn:
            sub = torch.zeros_like(sub)
        parts, order = [], []
        for u in sub.unique().tolist():
            idx = (sub == u).nonzero(as_tuple=True)[0]
            for m in self.sbns:
                m.active = u
            parts.append(self.encoder(x[idx]))
            order.append(idx)
        z, order = torch.cat(parts), torch.cat(order)
        out = z.new_zeros(len(x), z.size(1)).index_add(0, order, z)
        return out if self.linear else F.normalize(out, dim=1)

    def logits(self, z):
        if self.linear:
            return self.fc(z)
        p = F.normalize(self.proxies, dim=-1)                       # [C, K, D]
        d2 = ((z[:, None, None, :] - p[None]) ** 2).sum(-1)         # [N, C, K]
        return torch.logsumexp(-d2 / self.tau, dim=2)               # [N, C]

    def forward(self, x, sub):
        return self.logits(self.embed(x, sub))
