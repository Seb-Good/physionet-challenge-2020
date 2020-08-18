"""
lr_schedulers.py
---------------------------
This module provide classes and functions for managing learning rate schedules.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# 3rd party imports
import numpy as np


class AnnealingRestartScheduler(object):

    """
    Cyclical learning rate decay with warm restarts and cosine annealing.
    Reference: https://arxiv.org/pdf/1608.03983.pdf
    """

    def __init__(self, lr_min, lr_max, steps_per_epoch, lr_max_decay, epochs_per_cycle, cycle_length_factor):

        # Set parameters
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.steps_per_epoch = steps_per_epoch
        self.lr_max_decay = lr_max_decay
        self.epochs_per_cycle = epochs_per_cycle
        self.cycle_length_factor = cycle_length_factor

        # Set attributes
        self.lr = self.lr_max
        self.steps_since_restart = 0
        self.next_restart = self.epochs_per_cycle

    def on_batch_end_update(self):
        """Update at the end of each mini-batch."""
        # Update steps since restart
        self.steps_since_restart += 1

        # Update learning rate
        self.lr = self._compute_cosine_lr()

    def on_epoch_end_update(self, epoch):
        """Check for end of current cycle, apply restarts when necessary."""
        if epoch + 1 == self.next_restart:
            self.steps_since_restart = 0
            self.epochs_per_cycle = np.ceil(self.epochs_per_cycle * self.cycle_length_factor)
            self.next_restart += self.epochs_per_cycle
            self.lr_max *= self.lr_max_decay

    def _compute_cosine_lr(self):
        """Compute cosine learning rate decay."""
        # Compute the cycle completion factor
        fraction_complete = self._compute_fraction_complete()

        # Compute learning rate
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(fraction_complete * np.pi))

    def _compute_fraction_complete(self):
        """Compute the fraction of the cycle that is completed."""
        return self.steps_since_restart / (self.steps_per_epoch * self.epochs_per_cycle)


def exponential_step_decay(decay_epochs, decay_rate, initial_lr, epoch):
    """Compute exponential learning rate step decay."""
    return initial_lr * np.power(decay_rate, np.floor((epoch / decay_epochs)))


class AnnealingWarmRestartScheduler(object):
    def __init__(
        self,
        lr_min,
        lr_max,
        steps_per_epoch,
        lr_max_decay,
        epochs_per_cycle,
        cycle_length_factor,
        warmup_factor,
    ):

        # Set parameters
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.steps_per_epoch = steps_per_epoch
        self.lr_max_decay = lr_max_decay
        self.epochs_per_cycle = epochs_per_cycle
        self.cycle_length_factor = cycle_length_factor
        self.warmup_factor = warmup_factor

        # Set attributes
        self.lr = None
        self.cycle_epoch = None
        self.cycle_step = None
        self.warmup_step = None
        self.annealing_step = None
        self.lr_mode = None
        self.cycle_epochs = None
        self.cycle_steps = None
        self.warmup_epochs = None
        self.annealing_epochs = None
        self.warmup_steps = None
        self.annealing_steps = None

        # Get cycle schedule
        self._get_cycle_schedule()

    def on_batch_end_update(self):
        """Update learning rate at the end of each batch."""
        self.lr = self._compute_lr()

    def on_epoch_end_update(self):
        """Check for end of current cycle, apply restarts when necessary."""
        if self.cycle_epoch + 1 == self.cycle_epochs:
            self.epochs_per_cycle = int(np.ceil(self.epochs_per_cycle * self.cycle_length_factor))
            self.lr_max *= self.lr_max_decay
            self._get_cycle_schedule()
        else:
            self.cycle_epoch += 1

    def _compute_lr(self):
        """Compute learning rate."""
        if self.lr_mode[self.cycle_epoch] == 'warmup':
            return self._compute_warmup_lr()
        elif self.lr_mode[self.cycle_epoch] == 'annealing':
            return self._compute_annealing_lr()

    def _compute_warmup_lr(self):
        """Compute warmup learning rate."""
        # Update step
        self.warmup_step += 1

        # Compute learning rate
        lr = self.lr_min + (self.lr_max - self.lr_min) / self.warmup_steps * self.warmup_step

        return lr

    def _compute_annealing_lr(self):
        """Compute annealing learning rate."""
        # Update step
        self.annealing_step += 1

        # Compute the fraction of the annealing cycle that is completed
        fraction_complete = self._compute_fraction_complete_annealing()

        # Compute learning rate
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(fraction_complete * np.pi))

        return lr

    def _compute_fraction_complete(self):
        """Compute the fraction of the total cycle that is completed."""
        return self.cycle_step / (self.steps_per_epoch * self.epochs_per_cycle)

    def _compute_fraction_complete_warmup(self):
        """Compute the fraction of the warmup cycle that is completed."""
        return self.warmup_step / self.warmup_steps

    def _compute_fraction_complete_annealing(self):
        """Compute the fraction of the annealing cycle that is completed."""
        return self.annealing_step / self.annealing_steps

    def _get_cycle_schedule(self):
        """Generate warmup-annealing schedule for the current cycle."""
        self.cycle_epoch = 0
        self.cycle_step = 0
        self.warmup_step = 0
        self.annealing_step = 0

        # Get epochs
        self.cycle_epoch = 0
        self.cycle_epochs = self.epochs_per_cycle
        self.warmup_epochs = int(self.warmup_factor * self.cycle_epochs)
        self.annealing_epochs = self.epochs_per_cycle - self.warmup_epochs

        # Get steps
        self.cycle_step = 0
        self.cycle_steps = self.cycle_epochs * self.steps_per_epoch
        self.warmup_steps = self.warmup_epochs * self.steps_per_epoch
        self.annealing_steps = self.annealing_epochs * self.steps_per_epoch

        # Learning rate mode schedule
        self.lr_mode = [
            'annealing' if epoch not in range(int(self.warmup_factor * self.cycle_epochs)) else 'warmup'
            for epoch in range(self.cycle_epochs)
        ]

        # Learning rate
        if self.lr_mode[0] == 'warmup':
            self.lr = self.lr_min
        elif self.lr_mode[0] == 'annealing':
            self.lr = self.lr_max
