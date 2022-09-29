from functools import partial

from ml_collections import ConfigDict

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from .jax_utils import next_rng, value_and_multi_grad, mse_loss, cross_ent_loss 


class MR(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.rf_lr = 3e-4
        config.optimizer_type = 'adam'
        
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, rf):
        self.config = self.get_default_config(config)
        self.rf = rf
        self.observation_dim = rf.observation_dim
        self.action_dim = rf.action_dim

        self._train_states = {}

        optimizer_class = {
            'adam': optax.adam,
            'sgd': optax.sgd,
        }[self.config.optimizer_type]

        rf_params = self.rf.init(next_rng(), jnp.zeros((10, self.observation_dim)), jnp.zeros((10, self.action_dim)))
        self._train_states['rf'] = TrainState.create(
            params=rf_params,
            tx=optimizer_class(self.config.rf_lr),
            apply_fn=None,
        )

        model_keys = ['rf']
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
        # n_obs = batch['next_observations']
        # in_obs = jnp.concatenate([obs, n_obs], axis=-1)
        in_obs = obs
        train_params = {key: train_states[key].params for key in self.model_keys}
        rf_pred = self.rf.apply(train_params['rf'], in_obs, act)
        return rf_pred
    
    @partial(jax.jit, static_argnames=('self'))
    def _eval_pref_step(self, train_states, rng, batch):

        def loss_fn(train_params, rng):
            obs_1 = batch['observations']
            act_1 = batch['actions']
            obs_2 = batch['observations_2']
            act_2 = batch['actions_2']
            labels = batch['labels']
           
            B, T, obs_dim = batch['observations'].shape
            B, T, act_dim = batch['actions'].shape
            
            obs_1 = obs_1.reshape(-1, obs_dim)
            obs_2 = obs_2.reshape(-1, obs_dim)
            act_1 = act_1.reshape(-1, act_dim)
            act_2 = act_2.reshape(-1, act_dim)
           
            rf_pred_1 = self.rf.apply(train_params['rf'], obs_1, act_1)
            rf_pred_2 = self.rf.apply(train_params['rf'], obs_2, act_2)
            
            sum_pred_1 = jnp.mean(rf_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
            sum_pred_2 = jnp.mean(rf_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)
            
            loss_collection = {}

            rng, split_rng = jax.random.split(rng)
            
            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            rf_loss = cross_ent_loss(logits, label_target)

            loss_collection['rf'] = rf_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        metrics = dict(
            eval_rf_loss=aux_values['rf_loss'],
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
            labels = batch['labels']
            # n_obs_1 = batch['next_observations']
            # n_obs_2 = batch['next_observations_2']
            
            B, T, obs_dim = batch['observations'].shape
            B, T, act_dim = batch['actions'].shape
            
            obs_1 = obs_1.reshape(-1, obs_dim)
            obs_2 = obs_2.reshape(-1, obs_dim)
            act_1 = act_1.reshape(-1, act_dim)
            act_2 = act_2.reshape(-1, act_dim)
           
            rf_pred_1 = self.rf.apply(train_params['rf'], obs_1, act_1)
            rf_pred_2 = self.rf.apply(train_params['rf'], obs_2, act_2)
            
            sum_pred_1 = jnp.mean(rf_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
            sum_pred_2 = jnp.mean(rf_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)
            
            loss_collection = {}

            rng, split_rng = jax.random.split(rng)
            
            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            rf_loss = cross_ent_loss(logits, label_target)

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
        )

        return new_train_states, metrics

    def train_semi(self, labeled_batch, unlabeled_batch, lmd, tau):
        self._total_steps += 1
        self._train_states, metrics = self._train_semi_pref_step(
            self._train_states, labeled_batch, unlabeled_batch, lmd, tau, next_rng()
        )
        return metrics
    
    @partial(jax.jit, static_argnames=('self'))
    def _train_semi_pref_step(self, train_states, labeled_batch, unlabeled_batch, lmd, tau, rng):
        def compute_logits(batch):
            obs_1 = batch['observations']
            act_1 = batch['actions']
            obs_2 = batch['observations_2']
            act_2 = batch['actions_2']
            labels = batch['labels']
            # n_obs_1 = batch['next_observations']
            # n_obs_2 = batch['next_observations_2']
            
            B, T, obs_dim = batch['observations'].shape
            B, T, act_dim = batch['actions'].shape
            
            obs_1 = obs_1.reshape(-1, obs_dim)
            obs_2 = obs_2.reshape(-1, obs_dim)
            act_1 = act_1.reshape(-1, act_dim)
            act_2 = act_2.reshape(-1, act_dim)
           
            rf_pred_1 = self.rf.apply(train_params['rf'], obs_1, act_1)
            rf_pred_2 = self.rf.apply(train_params['rf'], obs_2, act_2)
            
            sum_pred_1 = jnp.mean(rf_pred_1.reshape(B,T), axis=1).reshape(-1,1)
            sum_pred_2 = jnp.mean(rf_pred_2.reshape(B,T), axis=1).reshape(-1,1)
            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)

            return logits, labels

        def loss_fn(train_params, lmd, tau, rng):
            logits, labels = compute_logits(labeled_batch)
            u_logits, _ = compute_logits(unlabeled_batch)
            
            loss_collection = {}

            rng, split_rng = jax.random.split(rng)
            
            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            rf_loss = cross_ent_loss(logits, label_target)

            u_confidence = jnp.max(jax.nn.softmax(u_logits, axis=-1), axis=-1)
            pseudo_labels = jnp.argmax(u_logits, axis=-1)
            pseudo_label_target = jax.lax.stop_gradient(pseudo_labels)
                    
            loss_ = optax.softmax_cross_entropy(logits=u_logits, 
                labels=jax.nn.one_hot(pseudo_label_target, num_classes=2))
            u_rf_loss = jnp.where(u_confidence > tau, loss_, 0).mean()
            u_rf_ratio = jnp.count_nonzero(u_confidence > tau) / len(u_confidence) * 100

            loss_collection['rf'] = rf_loss + lmd * u_rf_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, lmd, tau, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            rf_loss=aux_values['rf_loss'],
            u_rf_loss=aux_values['u_rf_loss'],
            u_rf_ratio=aux_values['u_rf_ratio']
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