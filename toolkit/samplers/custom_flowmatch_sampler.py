import math
from typing import Union, Optional, List
from torch.distributions import LogNormal
from diffusers import FlowMatchEulerDiscreteScheduler
import torch
import numpy as np


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class CustomFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self, *args, **kwargs):
        # Pop custom arguments before passing to super, so super doesn't complain if they aren't its own config fields
        # and so we can ensure they are set on the instance correctly.
        self.custom_timesteps = kwargs.pop("custom_timesteps", None)
        self.timestep_type = kwargs.pop("timestep_type", "linear") # Default to linear if not provided

        super().__init__(*args, **kwargs) # Initializes self.config with remaining kwargs

        # We can still allow overrides from self.config if needed, but direct pop is safer for these specific ones.
        # If self.custom_timesteps is still None here, it means it wasn't in kwargs or was explicitly None.
        # Similarly for self.timestep_type.
        self.init_noise_sigma = self.config.get("init_noise_sigma", 1.0) # Get from self.config or default

        with torch.no_grad():
            # create weights for timesteps
            num_timesteps = 1000
            # Bell-Shaped Mean-Normalized Timestep Weighting
            # bsmntw? need a better name

            x = torch.arange(num_timesteps, dtype=torch.float32)
            y = torch.exp(-2 * ((x - num_timesteps / 2) / num_timesteps) ** 2)

            # Shift minimum to 0
            y_shifted = y - y.min()

            # Scale to make mean 1
            bsmntw_weighing = y_shifted * (num_timesteps / y_shifted.sum())

            # only do half bell
            hbsmntw_weighing = y_shifted * (num_timesteps / y_shifted.sum())

            # flatten second half to max
            hbsmntw_weighing[num_timesteps //
                             2:] = hbsmntw_weighing[num_timesteps // 2:].max()

            # Create linear timesteps from 1000 to 0
            timesteps = torch.linspace(1000, 0, num_timesteps, device='cpu')

            self.linear_timesteps = timesteps
            self.linear_timesteps_weights = bsmntw_weighing
            self.linear_timesteps_weights2 = hbsmntw_weighing
            pass

    def get_weights_for_timesteps(self, timesteps: torch.Tensor, v2=False) -> torch.Tensor:
        # Get the indices of the timesteps
        step_indices = [(self.timesteps == t).nonzero().item()
                        for t in timesteps]

        # Get the weights for the timesteps
        if v2:
            weights = self.linear_timesteps_weights2[step_indices].flatten()
        else:
            weights = self.linear_timesteps_weights[step_indices].flatten()

        return weights

    def get_sigmas(self, timesteps: torch.Tensor, n_dim, dtype, device) -> torch.Tensor:
        sigmas = self.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item()
                        for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)

        return sigma

    def add_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.Tensor,
    ) -> torch.Tensor:
        t_01 = (timesteps / 1000).to(original_samples.device)
        # forward ODE
        noisy_model_input = (1.0 - t_01) * original_samples + t_01 * noise
        # reverse ODE
        # noisy_model_input = (1 - t_01) * noise + t_01 * original_samples
        return noisy_model_input

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        return sample

    def set_train_timesteps(self, 
                           num_inference_steps: int, 
                           device: Union[str, torch.device] = None, 
                           timestep_type: str = None, 
                           latents: torch.Tensor = None,
                           custom_timesteps_arg: Optional[List[int]] = None):
        # Determine the effective timestep_type, prioritizing the argument
        effective_timestep_type = timestep_type if timestep_type is not None else self.timestep_type

        if effective_timestep_type == "custom":
            # Determine the source of custom timesteps: argument first, then instance variable
            current_custom_steps = custom_timesteps_arg if custom_timesteps_arg is not None else self.custom_timesteps
            
            if current_custom_steps is None:
                raise ValueError(
                    "CustomFlowMatchEulerDiscreteScheduler: timestep_type is 'custom' but no custom_timesteps provided "
                    "(neither as argument nor found in scheduler's instance config). "
                    "Ensure 'custom_timesteps' is configured correctly."
                )
            if not isinstance(current_custom_steps, (list, tuple)) or not current_custom_steps:
                raise ValueError(
                    f"CustomFlowMatchEulerDiscreteScheduler: timestep_type is 'custom' but custom_timesteps "
                    f"is not a non-empty list or tuple. Received: {current_custom_steps}"
                )
            
            # Use the custom timesteps
            self.timesteps = torch.tensor(current_custom_steps, device=device, dtype=torch.long)
            # num_inference_steps should ideally match len(current_custom_steps) when type is custom
            self.num_inference_steps = len(current_custom_steps) 
        else:
            # Use the original implementation for other timestep types
            # num_inference_steps here is the one passed as argument, which is standard for non-custom types
            super().set_train_timesteps(num_inference_steps, device, effective_timestep_type, latents)
