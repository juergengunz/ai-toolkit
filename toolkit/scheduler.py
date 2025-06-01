import torch
from typing import Optional
# Use the get_scheduler from diffusers.optimization which is robust
from diffusers.optimization import get_scheduler, SchedulerType


class MinLRSchedulerWrapper:
    def __init__(self, scheduler, optimizer, min_lr):
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.min_lr = min_lr

    def step(self, epoch=None):
        self.scheduler.step(epoch)
        for param_group in self.optimizer.param_groups:
            if 'lr' in param_group:
                param_group['lr'] = max(param_group['lr'], self.min_lr)

    def __getattr__(self, name):
        # Forward other attributes to the original scheduler
        return getattr(self.scheduler, name)


def get_lr_scheduler(
        name: str, # Should be a string recognized by SchedulerType
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: Optional[int] = 0,
        num_training_steps: Optional[int] = None,
        **kwargs, # Allows for other scheduler-specific arguments from lr_scheduler_params
):
    """
    Returns a learning rate scheduler from the Hugging Face Transformers/Diffusers library.

    Args:
        name (str): The name of the scheduler (e.g., "linear", "cosine", 
                      "cosine_with_restarts", "polynomial", "constant", 
                      "constant_with_warmup").
        optimizer (torch.optim.Optimizer): The optimizer.
        num_warmup_steps (Optional[int]): The number of steps for the warmup phase.
        num_training_steps (Optional[int]): The total number of training steps.
        **kwargs: Additional arguments specific to the scheduler type.
    """
    if num_training_steps is None:
        # default to a large number. Should be set for most schedulers
        # raise ValueError("num_training_steps must be set for most schedulers")
        # just set a high number for now for ones that dont need it like constant
        num_training_steps = 1000

    print(f"DEBUG SCHEDULER: Attempting to create scheduler '{name}' with effective warmup_steps={num_warmup_steps}, training_steps={num_training_steps}, extra_params={kwargs}") # DEBUG ADDED

    min_lr = kwargs.pop('min_lr', None)

    try:
        # Map common names to SchedulerType if necessary, or expect direct SchedulerType string
        # For example, if user passes "cosine", SchedulerType("cosine") is "cosine".
        # If user passes "cosine_with_warmup", it should be SchedulerType.COSINE_WITH_RESTARTS or similar,
        # though many schedulers inherently handle warmup if num_warmup_steps > 0.
        
        # The `get_scheduler` function from diffusers/transformers handles mapping string names
        # to the correct scheduler functions and passes num_warmup_steps, num_training_steps,
        # and other kwargs appropriately.
        
        # Clean up any problematic kwargs that might have been added by older calling code
        kwargs.pop('total_iters', None)
        kwargs.pop('T_max', None)
        kwargs.pop('T_0', None)

        scheduler = get_scheduler(
            name=name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **kwargs # Pass through any other specific params from lr_scheduler_params
        )

        print(f"DEBUG SCHEDULER: Successfully created scheduler object: {type(scheduler)}") # DEBUG ADDED
        if min_lr is not None:
            print(f"DEBUG SCHEDULER: Wrapping scheduler with MinLRSchedulerWrapper, min_lr={min_lr}") # DEBUG ADDED
            return MinLRSchedulerWrapper(scheduler, optimizer, min_lr)
        return scheduler
    except Exception as e:
        print(f"Error creating scheduler '{name}' with Hugging Face/Diffusers: {e}")
        print(f"Make sure '{name}' is a valid scheduler name recognized by `diffusers.optimization.SchedulerType` "
              f"or `transformers.optimization.SchedulerType` and all required arguments are provided (like num_training_steps).")
        print("Valid names include: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup.")
        # Fallback or re-raise, depending on desired strictness
        # For now, re-raise to make issues apparent
        raise ValueError(f"Could not create scheduler: {name}") from e
