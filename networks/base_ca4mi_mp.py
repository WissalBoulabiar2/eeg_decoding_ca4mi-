"""
CA4MI-MP network: networks/base_ca4mi.Cls_HEAD with optional replay routing (replay_private='yes').

In the original forward, every sample goes through the private encoder of the current subject, while its head
is chosen by its own subject label. Replayed trials of subject u therefore train head_u on private_t features,
and those head updates are discarded at evaluation, which reloads (private_u, head_u) from checkpoint u.
With routing, replayed trials go through their own (frozen) private_u, so the shared encoder is trained to stay
compatible with exactly the (private_u, head_u) used to test subject u.
"""
import torch
from .base_ca4mi import Cls_HEAD as _Cls_HEAD


class Cls_HEAD(_Cls_HEAD):

    def __init__(self, args):
        super().__init__(args)
        self.route_replay = args.get('replay_private', 'yes') in ('yes', True)

    def forward(self, x_s, x_p, sub_module_label, sub_id):
        x_s = self.shared(x_s)
        p = self.private(x_p, sub_id)
        if self.route_replay:
            past = sub_module_label != sub_id
            if past.any():
                p = p.clone()
                for u in sub_module_label[past].unique().tolist():
                    m = sub_module_label == u
                    module = self.private.out[u]
                    was_training = module.training
                    module.eval()  # frozen: no gradient and no BatchNorm statistics update
                    with torch.no_grad():
                        p[m] = module(x_p[m]).view(int(m.sum()), -1)
                    module.train(was_training)
        x = torch.cat([p, x_s], dim=1)
        return torch.stack([self.head[sub_module_label[i]].forward(x[i]) for i in range(x.size(0))])

    def freeze_past_subjects(self, sub_id):
        """Freeze private encoders and heads of subjects < sub_id (they are the ones evaluated for those subjects)."""
        for u in range(sub_id):
            for p in list(self.private.out[u].parameters()) + list(self.head[u].parameters()):
                p.requires_grad_(False)
