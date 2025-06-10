import types
import math
import torch
from torch import inf
from functools import wraps, partial
import warnings
import weakref
from collections import Counter
from bisect import bisect_right

from models import GCN




### Code from: https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau

Optimizer = torch.optim.Optimizer

__all__ = ['LambdaLR', 'MultiplicativeLR', 'StepLR', 'MultiStepLR', 'ConstantLR', 'LinearLR',
           'ExponentialLR', 'SequentialLR', 'CosineAnnealingLR', 'ChainedScheduler', 'ReduceLROnPlateau',
           'CyclicLR', 'CosineAnnealingWarmRestarts', 'OneCycleLR', 'PolynomialLR', 'LRScheduler']

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


def update_LR(opt, lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr

def print_LR(opt):
    for param_group in opt.param_groups:
        print(f"LR = {param_group['lr']}")

def _check_verbose_deprecated_warning(verbose):
    """Raises a warning when verbose is not the default value."""
    if verbose != "deprecated":
        warnings.warn("The verbose parameter is deprecated. Please use get_last_lr() "
                      "to access the learning rate.", UserWarning)
        return verbose
    return False

class LRScheduler:

    def __init__(self, optimizer, last_epoch=-1, verbose="deprecated"):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   f"in param_groups[{i}] when resuming an optimizer")
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.verbose = _check_verbose_deprecated_warning(verbose)

        self._initial_step()

    def _initial_step(self):
        """Initialize step counts and performs a step"""
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """Display the current learning rate.
        """
        if is_verbose:
            if epoch is None:
                print(f'Adjusting learning rate of group {group} to {lr:.4e}.')
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                             "%.5d") % epoch
                print(f'Epoch {epoch_str}: adjusting learning rate of group {group} to {lr:.4e}.')


    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


# Including _LRScheduler for backwards compatibility
# Subclass instead of assign because we want __name__ of _LRScheduler to be _LRScheduler (assigning would make it LRScheduler).
class _LRScheduler(LRScheduler):
    pass


class _enable_get_lr_call:

    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_lr_called_within_step = False


class Dynamic_LR(LRScheduler):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode = 'max', factor=0.1, patience=10,
                 plateau_var = "test_auc", 
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, max_lr=1e-4, eps=1e-8, verbose=False):
        
        """
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        """
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}")
            self.min_lrs = list(min_lr)
            self.max_lrs = list(max_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.plateau_var = plateau_var

        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, model, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics[self.plateau_var])
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            if(self.verbose):
                print("Model is improving!")
            self.best = current
            self.num_bad_epochs = 0
        else:
            if(self.verbose):
                print(f"Model is not improving :( best = {self.best}, current = {current}")
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        print("Adjusting Learning Rate")
        self._reset()
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            new_lr = min(new_lr, self.max_lrs[i])
            if abs(old_lr - new_lr) > self.eps:
                param_group['lr'] = new_lr

    def get_last_lr(self):
        return self._last_lr
    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)

class Action_On_Plateau():

    def __init__(self, mode = 'max', patience=10,
                 plateau_var = "test_auc", 
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 eps=1e-8, verbose=False):

        self.patience = patience
        self.plateau_var = plateau_var

        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, model, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics[self.plateau_var])
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            if(self.verbose):
                print("Model is improving!")
            self.best = current
            self.num_bad_epochs = 0
        else:
            if(self.verbose):
                print(f"Model is not improving :( best = {self.best}, current = {current}")
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self.action(model, metrics, epoch)
    
    def action(self, model, metrics, epoch=None):
        if(self.verbose):
            print("Doing my action")
    
    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

class Partial_Reset(Action_On_Plateau):

    def __init__(self, mode='max', patience=10, plateau_var="test_auc", 
                        threshold=0.0001, threshold_mode='rel', cooldown=0, 
                        eps=1e-8, verbose=False):
        
        super().__init__(mode, patience, plateau_var, threshold, 
                         threshold_mode, cooldown, eps, verbose)
    
    def action(self, model, metrics, epoch=None):
        print("Partial Reset!!")
        GCN.partial_reset(model)
        self._reset()
        self.cooldown_counter = self.cooldown
        self.num_bad_epochs = 0
    

class Full_Reset(Action_On_Plateau):

    def __init__(self, mode='max', patience=10, plateau_var="test_auc", 
                        threshold=0.0001, threshold_mode='rel', cooldown=0, 
                        eps=1e-8, verbose=False):
        
        super().__init__(mode, patience, plateau_var, threshold, 
                         threshold_mode, cooldown, eps, verbose)
    
    def action(self, model, metrics, epoch=None):
        print("Full Reset!!")
        GCN.full_reset(model)
        self._reset()
        self.cooldown_counter = self.cooldown
        self.num_bad_epochs = 0

class Dynamic_LR_AND_Partial_Reset():
    def __init__(self, optimizer, mode = 'max', factor=0.1, patience=10,
                plateau_var = "test_auc", reset_patience=None, reset_plateau_var=None,
                threshold=1e-4, threshold_mode='rel', cooldown=0,
                min_lr=0, max_lr=1e-4, eps=1e-8, verbose=False):

        if (reset_patience == None):
            reset_patience = patience
        if(reset_plateau_var == None):
            reset_plateau_var = plateau_var

        self.dynamic_lr = Dynamic_LR(optimizer, mode=mode, factor=factor, patience = patience, 
                        plateau_var=plateau_var, threshold=threshold, threshold_mode =threshold_mode, 
                        cooldown=cooldown, min_lr=min_lr, max_lr=max_lr, eps=eps, verbose=verbose)
        
        self.partial_reset = Partial_Reset(mode=mode, patience=reset_patience, plateau_var=reset_plateau_var,
                        threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown,
                        eps=eps)

    def step(self, model, metrics, epoch=None):
        self.dynamic_lr.step(model=model, metrics=metrics, epoch=epoch)
        self.partial_reset.step(model=model, metrics=metrics, epoch=epoch)

class Dynamic_LR_AND_Full_Reset():
    def __init__(self, optimizer, mode = 'max', factor=0.1, patience=10,
                plateau_var = "test_auc", reset_patience=None, reset_plateau_var=None,
                threshold=1e-4, threshold_mode='rel', cooldown=0,
                min_lr=0, max_lr=1e-4, eps=1e-8, verbose=False):

        if (reset_patience == None):
            reset_patience = patience
        if(reset_plateau_var == None):
            reset_plateau_var = plateau_var

        self.dynamic_lr = Dynamic_LR(optimizer, mode=mode, factor=factor, patience = patience, 
                        plateau_var=plateau_var, threshold=threshold, threshold_mode =threshold_mode, 
                        cooldown=cooldown, min_lr=min_lr, max_lr=max_lr, eps=eps, verbose=verbose)
        
        self.full_reset = Full_Reset(mode=mode, patience=reset_patience, plateau_var=reset_plateau_var,
                        threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown,
                        eps=eps)

    def step(self, model, metrics, epoch=None):
        self.dynamic_lr.step(model=model, metrics=metrics, epoch=epoch)
        self.full_reset.step(model=model, metrics=metrics, epoch=epoch)