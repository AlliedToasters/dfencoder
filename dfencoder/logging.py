from collections import OrderedDict
import math

import numpy as np

class BasicLogger(object):
    """A minimal class for logging training progress."""

    def __init__(self, fts, baseline_loss=0.0):
        """Pass a list of fts as argument."""
        self.fts = fts
        self.train_fts = OrderedDict()
        self.val_fts = OrderedDict()
        for ft in self.fts:
            self.train_fts[ft] = [[], []]
            self.val_fts[ft] = []
        self.n_epochs = 0
        self.baseline_loss = baseline_loss

    def training_step(self, losses):
        for i, ft in enumerate(self.fts):
            self.train_fts[ft][0].append(losses[i])

    def end_epoch(self, val_losses=None):
        self.n_epochs += 1
        for i, ft in enumerate(self.fts):
            mean = np.array(self.train_fts[ft][0]).mean()
            self.train_fts[ft][1].append(mean)
            #reset train_fts log
            self.train_fts[ft][0] = []
            if val_losses is not None:
                self.val_fts[ft].append(val_losses[i])

class IpynbLogger(BasicLogger):
    """Plots Logging Data in jupyter notebook"""

    def __init__(self, *args, **kwargs):
        super(IpynbLogger, self).__init__(*args, **kwargs)
        import matplotlib.pyplot as plt
        from IPython.display import clear_output
        self.plt = plt
        self.clear_output = clear_output

    def end_epoch(self, val_losses=None):
        super(IpynbLogger, self).end_epoch(val_losses)
        if self.n_epochs > 1:
            self.plot_progress()

    def plot_progress(self):
        self.clear_output()
        x = list(range(1, self.n_epochs+1))
        train_loss = [self.train_fts[ft][1] for ft in self.fts]
        train_loss = np.array(train_loss).sum(axis=0)
        self.plt.plot(x, train_loss, label='train loss', color='orange')
        if len(self.val_fts[self.fts[0]]) > 0:
            self.plt.axhline(
                y=self.baseline_loss,
                linestyle='dotted',
                label='baseline val loss',
                color='blue'
            )
            val_loss = [self.val_fts[ft] for ft in self.fts]
            val_loss = np.array(val_loss).sum(axis=0)
            self.plt.plot(x, val_loss, label='val loss', color='blue')
        self.plt.ylim(0, math.ceil(2*self.baseline_loss))
        self.plt.xticks(x, x)
        self.plt.legend()
        self.plt.xlabel('epochs')
        self.plt.ylabel('loss')
        self.plt.show();