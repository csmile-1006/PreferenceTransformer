from functools import partial
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import distrax

from .jax_utils import extend_and_repeat, next_rng


def multiple_action_q_function(forward):
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(self, observations, actions, **kwargs):
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(-1, observations.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
        q_values = forward(self, observations, actions, **kwargs)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values
    return wrapped


class FullyConnectedNetwork(nn.Module):
    output_dim: int
    arch: str = '256-256'
    orthogonal_init: bool = False
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activation_final: Callable[[jnp.ndarray], jnp.ndarray] = None

    @nn.compact
    def __call__(self, input_tensor):
        x = input_tensor
        hidden_sizes = [int(h) for h in self.arch.split('-')]
        for h in hidden_sizes:
            if self.orthogonal_init:
                x = nn.Dense(
                    h,
                    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
                    bias_init=jax.nn.initializers.zeros
                )(x)
            else:
                x = nn.Dense(h)(x)
            x = self.activations(x)

        if self.orthogonal_init:
            output = nn.Dense(
                self.output_dim,
                kernel_init=jax.nn.initializers.orthogonal(1e-2),
                bias_init=jax.nn.initializers.zeros
            )(x)
        else:
            output = nn.Dense(
                self.output_dim,
                kernel_init=jax.nn.initializers.variance_scaling(
                    1e-2, 'fan_in', 'uniform'
                ),
                bias_init=jax.nn.initializers.zeros
            )(x)
        
        if self.activation_final is not None:
            output = self.activation_final(output)
        return output

class FullyConnectedQFunction(nn.Module):
    observation_dim: int
    action_dim: int
    arch: str = '256-256'
    orthogonal_init: bool = False
    activations: str = 'relu'
    activation_final: str = 'none'

    @nn.compact
    @multiple_action_q_function
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)

        activations = {
            'relu': nn.relu,
            'leaky_relu': nn.leaky_relu,
        }[self.activations]
        activation_final = {
            'none': None,
            'tanh': nn.tanh,
        }[self.activation_final]

        x = FullyConnectedNetwork(output_dim=1, arch=self.arch, orthogonal_init=self.orthogonal_init, activations=activations, activation_final=activation_final)(x)
        return jnp.squeeze(x, -1)
