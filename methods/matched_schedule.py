"""
matched_regime='yes': gives a baseline exactly ProtoMem's optimisation regime (review 14, M2), so that a comparison
measures the method and not the reference hyperparameters. Per subject, as in methods/protomem.py:
- Adam restarted at lr 1e-3, weight decay 1e-4;
- at most 150 epochs; checkpoint selected on the current subject's validation loss, not before epoch 10;
- after 10 epochs without improvement the lr is divided by 3 and Adam is restarted; stop when lr < 1e-5;
- the selected checkpoint (deep copy) is restored at the end.
Only the schedule changes; losses, replay and architecture stay those of the baseline.
"""
import copy

LR, WD, EPOCHS, PATIENCE, FACTOR, LR_MIN, MIN_SELECT = 1e-3, 1e-4, 150, 10, 3.0, 1e-5, 10


def enabled(args):
    return str(getattr(args, 'matched_regime', 'no')) in ('yes', 'True', 'true')


class MatchedSchedule:
    def __init__(self, model, optimizers):
        self.model, self.opts = model, list(optimizers)
        self.lr, self.patience = LR, PATIENCE
        self._set_lr(LR, wd=WD)
        self.best_loss, self.best_epoch = float('inf'), 0
        self.best_state = copy.deepcopy(model.state_dict())

    def _set_lr(self, lr, wd=None):
        for opt in self.opts:
            opt.state.clear()                        # restart Adam, as ProtoMem builds a new optimizer
            for g in opt.param_groups:
                g['lr'] = lr
                if wd is not None:
                    g['weight_decay'] = wd

    def step(self, epoch, v_loss):
        """epoch: 1-based index of the epoch just trained. Returns True when training must stop."""
        if epoch >= MIN_SELECT:
            if v_loss < self.best_loss:
                self.best_loss, self.best_epoch, self.patience = v_loss, epoch, PATIENCE
                self.best_state = copy.deepcopy(self.model.state_dict())
            else:
                self.patience -= 1
                if self.patience <= 0:
                    self.lr /= FACTOR
                    if self.lr < LR_MIN:
                        return True
                    self._set_lr(self.lr)
                    self.patience = PATIENCE
        return epoch >= EPOCHS

    def restore(self):
        self.model.load_state_dict(self.best_state)
        print(f"  [matched regime] selected epoch {self.best_epoch} (valid loss {self.best_loss:.3f})")
