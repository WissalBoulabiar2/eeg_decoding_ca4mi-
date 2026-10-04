"""
Multi-prototype memory for CA4MI-MP.

Each memory entry is a Gaussian mode (mu, diag(var)) of the shared latent space,
tagged with its class, the subject it comes from and the number of samples it
summarises (count). Anchors used by the prototype loss are obtained by clustering
the memory entries of each class into K modes instead of averaging them.
"""
import torch


def weighted_kmeans(x, k, weights=None, n_iter=50, seed=0):
    """Weighted k-means with k-means++ init. Returns (centers [k,D], assign [N])."""
    n = x.size(0)
    k = max(1, min(k, n))
    weights = torch.ones(n, device=x.device) if weights is None else weights.float()
    g = torch.Generator().manual_seed(seed)

    idx = [torch.multinomial((weights / weights.sum()).cpu(), 1, generator=g).item()]
    for _ in range(1, k):
        d2 = torch.cdist(x, x[idx]).min(dim=1).values.pow(2) * weights
        if d2.sum() <= 0:
            break
        idx.append(torch.multinomial((d2 / d2.sum()).cpu(), 1, generator=g).item())
    centers = x[idx].clone()

    assign = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(n_iter):
        assign = torch.cdist(x, centers).argmin(dim=1)
        new_centers = centers.clone()
        for j in range(centers.size(0)):
            m = assign == j
            if m.any():
                w = weights[m].unsqueeze(1)
                new_centers[j] = (w * x[m]).sum(0) / w.sum()
        if torch.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers, assign


def merge_moments(mu, var, count):
    """Moment-matching merge of several Gaussian modes into one (mu, var, count)."""
    w = count.float().unsqueeze(1)
    n = w.sum()
    m = (w * mu).sum(0) / n
    v = (w * (var + mu.pow(2))).sum(0) / n - m.pow(2)
    return m, v.clamp_min(0.), n.squeeze()


class PrototypeAnchors:
    """Anchors handed to the prototype loss: K modes per class."""

    def __init__(self, mu, var, labels):
        self.mu = mu          # [M, D]
        self.var = var        # [M, D]
        self.labels = labels  # [M]

    def __len__(self):
        return self.mu.size(0)

    def summary(self):
        return {int(c): int((self.labels == c).sum()) for c in self.labels.unique()}


class PrototypeMemory:
    def __init__(self, n_classes, max_prototypes, selection='diversity',
                 class_balanced=True, seed=0):
        self.n_classes = n_classes
        self.max_prototypes = max_prototypes
        self.selection = selection
        self.class_balanced = class_balanced
        self.seed = seed
        self.mu = None       # [N, D]
        self.var = None      # [N, D]
        self.count = None    # [N]
        self.labels = None   # [N]
        self.subjects = None  # [N]

    def __len__(self):
        return 0 if self.mu is None else self.mu.size(0)

    # ------------------------------------------------------------------ add
    def add_subject(self, features, labels, sub_id, modes_per_subject=1):
        """Summarise one subject's shared features into modes_per_subject modes per class."""
        mus, vars_, counts, lbls = [], [], [], []
        for c in range(self.n_classes):
            fc = features[labels == c]
            if fc.size(0) == 0:
                continue
            k = min(modes_per_subject, max(1, fc.size(0) // 2))
            _, assign = weighted_kmeans(fc, k, seed=self.seed + sub_id)
            for j in assign.unique():
                fj = fc[assign == j]
                mus.append(fj.mean(0))
                vars_.append(fj.var(0, unbiased=False) if fj.size(0) > 1 else torch.zeros_like(fj[0]))
                counts.append(float(fj.size(0)))
                lbls.append(c)

        device = features.device
        new = dict(mu=torch.stack(mus), var=torch.stack(vars_),
                   count=torch.tensor(counts, device=device),
                   labels=torch.tensor(lbls, device=device, dtype=torch.long),
                   subjects=torch.full((len(lbls),), sub_id, device=device, dtype=torch.long))
        for key, val in new.items():
            old = getattr(self, key)
            setattr(self, key, val if old is None else torch.cat([old, val]))
        self._consolidate()

    # ---------------------------------------------------------- consolidate
    def _class_budgets(self):
        base, rest = divmod(self.max_prototypes, self.n_classes)
        return {c: base + (1 if c < rest else 0) for c in range(self.n_classes)}

    def _consolidate(self):
        if len(self) <= self.max_prototypes:
            return
        if self.class_balanced:
            budgets = self._class_budgets()
            parts = [self._reduce(self.labels == c, budgets[c]) for c in range(self.n_classes)]
            parts = [p for p in parts if p is not None]
            self._set(*[torch.cat(t) for t in zip(*parts)])
        else:
            self._set(*self._reduce(torch.ones_like(self.labels, dtype=torch.bool), self.max_prototypes))

    def _set(self, mu, var, count, labels, subjects):
        self.mu, self.var, self.count, self.labels, self.subjects = mu, var, count, labels, subjects

    def _reduce(self, mask, budget):
        """Reduce the entries selected by mask to at most budget entries."""
        if not mask.any():
            return None
        mu, var, count = self.mu[mask], self.var[mask], self.count[mask]
        labels, subjects = self.labels[mask], self.subjects[mask]
        if mu.size(0) <= budget:
            return mu, var, count, labels, subjects

        if self.selection == 'random':
            # Equivalent to the original reservoir: uniform random subset, the rest is discarded
            g = torch.Generator().manual_seed(self.seed + len(self) * self.n_classes + int(labels[0]))
            keep = torch.randperm(mu.size(0), generator=g)[:budget].to(mu.device)
            return mu[keep], var[keep], count[keep], labels[keep], subjects[keep]

        if self.selection == 'diversity':
            return self._ward_merge(mu, var, count, labels, subjects, budget)

        raise ValueError(f"Unknown memory_selection: {self.selection}")

    @staticmethod
    def _ward_merge(mu, var, count, labels, subjects, budget):
        """
        Diversity-aware reduction: repeatedly merge the pair of same-class modes with the
        lowest Ward cost  n_i n_j / (n_i + n_j) * ||mu_i - mu_j||^2.
        Near-duplicates get merged (representativeness: the merged mode keeps their mass),
        isolated modes survive (diversity). Nothing is discarded at random.
        """
        mu, var, count = mu.clone(), var.clone(), count.clone().float()
        labels, subjects = labels.clone(), subjects.clone()
        while mu.size(0) > budget:
            d2 = torch.cdist(mu, mu).pow(2)
            n = count.unsqueeze(0) * count.unsqueeze(1) / (count.unsqueeze(0) + count.unsqueeze(1))
            cost = n * d2
            invalid = (labels.unsqueeze(0) != labels.unsqueeze(1)) | torch.eye(len(mu), dtype=torch.bool,
                                                                                 device=mu.device)
            cost[invalid] = float('inf')
            if torch.isinf(cost).all():
                break
            flat = torch.argmin(cost)
            i, j = divmod(flat.item(), mu.size(0))
            m, v, c = merge_moments(mu[[i, j]], var[[i, j]], count[[i, j]])
            if count[j] > count[i]:
                subjects[i] = subjects[j]  # merged mode is attributed to its dominant subject
            mu[i], var[i], count[i] = m, v, c
            keep = torch.arange(mu.size(0), device=mu.device) != j
            mu, var, count, labels, subjects = mu[keep], var[keep], count[keep], labels[keep], subjects[keep]
        return mu, var, count, labels, subjects

    # -------------------------------------------------------------- anchors
    def anchors(self, n_modes=1):
        """Cluster each class's memory into n_modes anchors (n_modes=1 -> single class prototype)."""
        if len(self) == 0:
            return None
        mus, vars_, lbls = [], [], []
        for c in range(self.n_classes):
            m = self.labels == c
            if not m.any():
                continue
            mu_c, var_c, cnt_c = self.mu[m], self.var[m], self.count[m]
            _, assign = weighted_kmeans(mu_c, n_modes, weights=cnt_c, seed=self.seed + c)
            for j in assign.unique():
                a = assign == j
                mj, vj, _ = merge_moments(mu_c[a], var_c[a], cnt_c[a])
                mus.append(mj)
                vars_.append(vj)
                lbls.append(c)
        return PrototypeAnchors(torch.stack(mus), torch.stack(vars_),
                                torch.tensor(lbls, device=self.mu.device, dtype=torch.long))

    def summary(self):
        if len(self) == 0:
            return 'empty'
        per_class = {c: int((self.labels == c).sum()) for c in range(self.n_classes)}
        per_subject = {int(s): int((self.subjects == s).sum()) for s in self.subjects.unique()}
        return f"{len(self)} entries | per class {per_class} | per subject {per_subject}"
