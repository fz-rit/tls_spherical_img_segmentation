import math

class EarlyStopping:
    def __init__(
        self, patience=5, min_delta=0.0, mode='loss',
        alpha=0.1  # smoothing factor for EWMA
    ):
        """
        Patience-based early stopping on a *smoothed* metric.

        Parameters:
        -----------
        patience : int
            How many epochs of *no improvement* in the smoothed metric to tolerate before stopping.
        min_delta : float
            Minimum change to qualify as improvement.
        mode : str
            'loss' => want to see a DECREASE, 'accu' => want to see an INCREASE.
        alpha : float
            EWMA smoothing factor, between 0 and 1. Smaller alpha (e.g., 0.1) makes it even less reactive to short-term changes. Larger alpha (e.g., 0.5) reacts more quickly but still smooths a bit.
        """
        assert mode in ['loss', 'accu'], "❗mode must be 'loss' or 'accu'❗"
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.alpha = alpha
        self.smooth_metric = None

        self.should_minimize = (mode == 'loss')
        self.best_score = None
        self.best_epoch = None

        self.counter = 0
        self.early_stop = False

    def __call__(self, current_value, epoch=None):
        # If NaN or inf, just treat as no improvement
        if not math.isfinite(current_value):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return

        # 1) Update smoothed metric:
        if self.smooth_metric is None:
            self.smooth_metric = current_value
        else:
            self.smooth_metric = (
                self.alpha * current_value
                + (1 - self.alpha) * self.smooth_metric
            )

        # 2) Initialize best_score if needed
        if self.best_score is None:
            self.best_score = self.smooth_metric
            self.best_epoch = epoch
            return

        # 3) Check improvement
        improved = (
            (self.smooth_metric < self.best_score - self.min_delta)
            if self.should_minimize
            else (self.smooth_metric > self.best_score + self.min_delta)
        )

        if improved:
            self.best_score = self.smooth_metric
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
