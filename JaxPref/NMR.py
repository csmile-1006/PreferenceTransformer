from functools import partial

from ml_collections import ConfigDict

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from .jax_utils import next_rng, value_and_multi_grad, mse_loss, cross_ent_loss


class NMR(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.lstm_lr = 1e-3
        config.optimizer_type = 'adam'
        config.scheduler_type = 'none'
        config.vocab_size = 1
        config.n_layer = 3
        config.embd_dim = 256
        config.n_embd = config.embd_dim
        config.n_head = 1
        config.n_inner = config.embd_dim // 2
        config.n_positions = 1024
        config.resid_pdrop = 0.1
        config.attn_pdrop = 0.1

        config.use_kld = False
        config.lambda_kld = 0.1
        config.softmax_temperature = 5

        config.train_type = "sum"
        config.train_diff_bool = False

        config.explicit_sparse = False
        config.k = 5

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, lstm):
        self.config = config
        self.lstm = lstm
        self.observation_dim = lstm.observation_dim
        self.action_dim = lstm.action_dim

        self._train_states = {}

        optimizer_class = {
            'adam': optax.adam,
            'adamw': optax.adamw,
            'sgd': optax.sgd,
        }[self.config.optimizer_type]


        scheduler_class = {
           'none': None
        }[self.config.scheduler_type]

        if scheduler_class:
            tx = optimizer_class(scheduler_class)
        else:
            tx = optimizer_class(learning_rate=self.config.lstm_lr)

        lstm_params = self.lstm.init({"params": next_rng(), "dropout": next_rng()}, jnp.zeros((10, 10, self.observation_dim)), jnp.zeros((10, 10, self.action_dim)), jnp.ones((10, 10), dtype=jnp.int32))
        self._train_states['lstm'] = TrainState.create(
            params=lstm_params,
            tx=tx,
            apply_fn=None
        )

        model_keys = ['lstm']
        self._model_keys = tuple(model_keys)
        self._total_steps = 0
        
    def evaluation(self, batch):
        metrics = self._eval_pref_step(
            self._train_states, next_rng(), batch
        )
        return metrics

    def get_reward(self, batch):
        return self._get_reward_step(self._train_states, batch)

    @partial(jax.jit, static_argnames=('self'))
    def _get_reward_step(self, train_states, batch):
        obs = batch['observations']
        act = batch['actions']
        timestep = batch['timestep']
        # n_obs = batch['next_observations']

        train_params = {key: train_states[key].params for key in self.model_keys}
        lstm_pred, _ = self.lstm.apply(train_params['lstm'], obs, act, timestep)
        return lstm_pred, None
   
    @partial(jax.jit, static_argnames=('self'))
    def _eval_pref_step(self, train_states, rng, batch):

        def loss_fn(train_params, rng):
            obs_1 = batch['observations']
            act_1 = batch['actions']
            obs_2 = batch['observations_2']
            act_2 = batch['actions_2']
            timestep_1 = batch['timestep_1']
            timestep_2 = batch['timestep_2']
            labels = batch['labels']
          
            B, T, _ = batch['observations'].shape
            B, T, _ = batch['actions'].shape

            rng, _ = jax.random.split(rng)
            
            lstm_pred_1, _ = self.lstm.apply(train_params['lstm'], obs_1, act_1, timestep_1, training=True, attn_mask=None, rngs={"dropout": rng})
            lstm_pred_2, _ = self.lstm.apply(train_params['lstm'], obs_2, act_2, timestep_2, training=True, attn_mask=None, rngs={"dropout": rng})

            if self.config.train_type == "mean":
                sum_pred_1 = jnp.mean(lstm_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.mean(lstm_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "sum":
                sum_pred_1 = jnp.sum(lstm_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.sum(lstm_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "last":
                sum_pred_1 = lstm_pred_1.reshape(B, T)[:, -1].reshape(-1, 1)
                sum_pred_2 = lstm_pred_2.reshape(B, T)[:, -1].reshape(-1, 1)

            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)
            
            loss_collection = {}
            rng, split_rng = jax.random.split(rng)
            
            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            lstm_loss = cross_ent_loss(logits, label_target)
            loss_collection['lstm'] = lstm_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()


        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), _ = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        metrics = dict(
            eval_lstm_loss=aux_values['lstm_loss'],
        )

        return metrics
        
    def train(self, batch):
        self._total_steps += 1
        self._train_states, metrics = self._train_pref_step(
            self._train_states, next_rng(), batch
        )
        return metrics
    
    @partial(jax.jit, static_argnames=('self'))
    def _train_pref_step(self, train_states, rng, batch):

        def loss_fn(train_params, rng):
            obs_1 = batch['observations']
            act_1 = batch['actions']
            obs_2 = batch['observations_2']
            act_2 = batch['actions_2']
            timestep_1 = batch['timestep_1']
            timestep_2 = batch['timestep_2']
            labels = batch['labels']
          
            B, T, _ = batch['observations'].shape
            B, T, _ = batch['actions'].shape
            
            rng, _ = jax.random.split(rng)
            
            lstm_pred_1, _ = self.lstm.apply(train_params['lstm'], obs_1, act_1, timestep_1, training=True, attn_mask=None, rngs={"dropout": rng})
            lstm_pred_2, _ = self.lstm.apply(train_params['lstm'], obs_2, act_2, timestep_2, training=True, attn_mask=None, rngs={"dropout": rng})

            if self.config.train_type == "mean":
                sum_pred_1 = jnp.mean(lstm_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.mean(lstm_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            if self.config.train_type == "sum":
                sum_pred_1 = jnp.sum(lstm_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.sum(lstm_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "last":
                sum_pred_1 = lstm_pred_1.reshape(B, T)[:, -1].reshape(-1, 1)
                sum_pred_2 = lstm_pred_2.reshape(B, T)[:, -1].reshape(-1, 1)
            
            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)
            
            loss_collection = {}
            rng, split_rng = jax.random.split(rng)
            
            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            lstm_loss = cross_ent_loss(logits, label_target)

            loss_collection['lstm'] = lstm_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            lstm_loss=aux_values['lstm_loss'],
        )

        return new_train_states, metrics
    
    def train_regression(self, batch):
        self._total_steps += 1
        self._train_states, metrics = self._train_regression_step(
            self._train_states, next_rng(), batch
        )
        return metrics
    
    @partial(jax.jit, static_argnames=('self'))
    def _train_regression_step(self, train_states, rng, batch):

        def loss_fn(train_params, rng):
            observations = batch['observations']
            next_observations = batch['next_observations']
            actions = batch['actions']
            rewards = batch['rewards']
            
            in_obs = jnp.concatenate([observations, next_observations], axis=-1)

            loss_collection = {}

            rng, split_rng = jax.random.split(rng)
            
            """ reward function loss """
            rf_pred = self.rf.apply(train_params['rf'], observations, actions)
            reward_target = jax.lax.stop_gradient(rewards)
            rf_loss = mse_loss(rf_pred, reward_target)

            loss_collection['rf'] = rf_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            rf_loss=aux_values['rf_loss'],
            average_rf=aux_values['rf_pred'].mean(),
        )

        return new_train_states, metrics

    @property
    def model_keys(self):
        return self._model_keys

    @property
    def train_states(self):
        return self._train_states

    @property
    def train_params(self):
        return {key: self.train_states[key].params for key in self.model_keys}

    @property
    def total_steps(self):
        return self._total_steps