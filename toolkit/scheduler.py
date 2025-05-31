import torch
from typing import Optional
# Use the get_scheduler from diffusers.optimization which is robust
from diffusers.optimization import get_scheduler, SchedulerType
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_min_lr(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and
    the initial lr set in the optimizer.
    """
    def lr_lambda(current_step: int):
        # Get the initial learning rate from the optimizer
        initial_lr = optimizer.param_groups[0]["lr"]
        
        if current_step < num_warmup_steps:
            # During warmup: linear increase from 0 to initial_lr
            lr = initial_lr * float(current_step) / float(max(1, num_warmup_steps))
            print(f"DEBUG LR: step={current_step}, warmup phase, lr={lr}")
            return lr
        
        # After warmup: cosine decay from initial_lr to min_lr
        # Calculate progress from 0 to 1 over the remaining steps
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        
        # The cosine function goes from 1 to -1, so we scale it to go from 1 to 0
        cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)))
        
        # Scale the decay to go from initial_lr to min_lr
        # When progress = 0, cosine_decay = 1, so lr = initial_lr
        # When progress = 1, cosine_decay = 0, so lr = min_lr
        lr = min_lr + (initial_lr - min_lr) * cosine_decay
        
        # Ensure we never go below min_lr
        lr = max(lr, min_lr)
        print(f"DEBUG LR: step={current_step}, decay phase, progress={progress:.3f}, cosine_decay={cosine_decay:.3f}, lr={lr}")
        return lr

    return LambdaLR(optimizer, lr_lambda)


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
        num_training_steps = 1000

    print(f"DEBUG SCHEDULER: Attempting to create scheduler '{name}' with effective warmup_steps={num_warmup_steps}, training_steps={num_training_steps}, extra_params={kwargs}")

    try:
        # Handle our custom cosine scheduler with min_lr
        if name == "cosine_with_min_lr":
            min_lr = float(kwargs.pop("min_lr", 0.0001))  # Ensure min_lr is a float
            print(f"DEBUG SCHEDULER: Using custom cosine scheduler with min_lr={min_lr}")
            return get_cosine_schedule_with_min_lr(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                min_lr=min_lr,
            )

        # Clean up any problematic kwargs that might have been added by older calling code
        kwargs.pop('total_iters', None)
        kwargs.pop('T_max', None)
        kwargs.pop('T_0', None)

        scheduler = get_scheduler(
            name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **kwargs
        )

        print(f"DEBUG SCHEDULER: Successfully created scheduler object: {type(scheduler)}")
        return scheduler
    except Exception as e:
        print(f"Error creating scheduler '{name}' with Hugging Face/Diffusers: {e}")
        print(f"Make sure '{name}' is a valid scheduler name recognized by `diffusers.optimization.SchedulerType` "
              f"or `transformers.optimization.SchedulerType` and all required arguments are provided (like num_training_steps).")
        print("Valid names include: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup.")
        raise ValueError(f"Could not create scheduler: {name}") from e
