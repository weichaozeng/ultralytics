"""
Metrics for evaluation, such as running mean and peak signal-to-noise ratio (PSNR).
"""
import math
import torch


class MeanValue:
    """
    Tracks running mean for online updates.
    """
    def __init__(self):
        """Initialize/reset running mean."""
        self.mean = 0.0
        self.n = 0

    def __repr__(self):
        """String representation."""
        return str(self.mean)

    def __float__(self):
        """Conversion to float."""
        return float(self.mean)

    def __coerce__(self, other):
        """Coercion for numeric ops."""
        return float(self), other

    def __str__(self):
        """String value."""
        return str(float(self))

    def __format__(self, format_spec):
        """Format to custom string."""
        return f"{float(self) :{format_spec}}"

    def toJSON(self):
        """Convert to JSON serializable float."""
        return float(self)

    def reset(self):
        """Reset mean and count."""
        self.mean = 0.0
        self.n = 0

    def update(self, x):
        """Update with a new observation."""
        self.n += 1
        self.mean += (x - self.mean) / self.n


class PSNR:
    """
    Peak Signal to Noise Ratio (PSNR) calculator for images or tensors.
    """
    def __init__(self, max_value: float = 1.0):
        """Initialize accumulator, set value for perfect MSE=0."""
        self.max_value = max_value
        self.sum_squared_error = 0.0
        self.num_squared_error = 0

    def __call__(self, pred, true) -> torch.Tensor:
        """Compute PSNR between prediction and ground truth."""
        return self._compute_psnr(((pred - true) ** 2).mean())

    def compute(self):
        """Return PSNR for all accumulated values."""
        return self._compute_psnr(self.sum_squared_error / self.num_squared_error)

    def compute_and_reset(self):
        """Get PSNR and reset accumulators."""
        result = self.compute()
        self.reset()
        return result

    def reset(self):
        """Reset all counters and sums."""
        self.sum_squared_error = 0.0
        self.num_squared_error = 0

    def update(self, pred, true):
        """Accumulate another prediction/target pair."""
        squared_error = (pred - true) ** 2
        self.sum_squared_error += squared_error.mean()
        self.num_squared_error += squared_error.numel()

    def _compute_psnr(self, mse):
        """Compute PSNR from MSE (internal/utility)."""
        return 20.0 * math.log10(self.max_value) - 10.0 * torch.log10(mse)
