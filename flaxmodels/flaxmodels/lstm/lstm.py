import functools
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any
import h5py

from .. import utils
from . import ops


class SimpleLSTM(nn.Module):
  """A simple unidirectional LSTM."""

  @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1, out_axes=1,
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x):
    return nn.OptimizedLSTMCell()(carry, x)

  @staticmethod
  def initialize_carry(batch_dims, hidden_size):
    # Use fixed random key since default state init fn is just zeros.
    return nn.OptimizedLSTMCell.initialize_carry(
        jax.random.PRNGKey(0), batch_dims, hidden_size)


class LSTMRewardModel(nn.Module):
    config: Any=None
    pretrained: str=None
    ckpt_dir: str=None
    observation_dim: int=29
    action_dim: int=8
    activation: str=None
    activation_final: str=None
    max_episode_steps: int=1000

    def setup(self):
        self.config_ = self.config
        self.config_.activation_function = self.activation
        self.config_.activation_final = self.activation_final
        self.vocab_size = self.config_.vocab_size
        self.max_pos = self.config_.n_positions
        self.embd_dim = self.config_.n_embd
        self.embd_dropout = self.config_.embd_pdrop
        self.num_layers = self.config_.n_layer
        self.inner_dim = self.config_.n_inner
        self.eps = self.config_.layer_norm_epsilon

    @nn.compact
    def __call__(
        self,
        states,
        actions,
        timesteps,
        attn_mask=None,
        training=False,
        reverse=False,
        target_idx=1
    ):
        batch_size = states.shape[0]

        x = jnp.concatenate([states, actions], axis=-1)
        for hd in [self.embd_dim, self.embd_dim // 2, self.embd_dim // 2]:
            x = nn.Dense(features=hd)(x)
            x = ops.apply_activation(x, activation=self.activation)
            x = nn.Dropout(rate=self.embd_dropout)(x, deterministic=not training)
       
        lstm = SimpleLSTM()
        initial_state = lstm.initialize_carry((batch_size, ), self.embd_dim // 2)
        _, lstm_outputs = lstm(initial_state, x)
        x = jnp.concatenate([x, lstm_outputs], axis=-1)
        for hd in [self.embd_dim // 2, self.embd_dim // 4, self.embd_dim // 4]:
            x = nn.Dense(features=hd)(x)
            x = ops.apply_activation(x, activation=self.activation)
            x = nn.Dropout(rate=self.embd_dropout)(x, deterministic=not training)
        output = nn.Dense(features=1)(x)

        return output, lstm_outputs
