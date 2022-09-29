from functools import partial

from ml_collections import ConfigDict

import jax
import jax.numpy as jnp

import optax
import numpy as np
from flax.training.train_state import TrainState

from .jax_utils import next_rng, value_and_multi_grad, mse_loss, cross_ent_loss, kld_loss


class PrefTransformer(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.trans_lr = 1e-4
        config.optimizer_type = 'adamw'
        config.scheduler_type = 'CosineDecay'
        config.vocab_size = 1
        config.n_layer = 3
        config.embd_dim = 256
        config.n_embd = config.embd_dim
        config.n_head = 1
        config.n_positions = 1024
        config.resid_pdrop = 0.1
        config.attn_pdrop = 0.1
        config.pref_attn_embd_dim = 256

        config.train_type = "mean"

        # Weighted Sum option
        config.use_weighted_sum = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, trans):
        self.config = config
        self.trans = trans
        self.observation_dim = trans.observation_dim
        self.action_dim = trans.action_dim

        self._train_states = {}

        optimizer_class = {
            'adam': optax.adam,
            'adamw': optax.adamw,
            'sgd': optax.sgd,
        }[self.config.optimizer_type]

        scheduler_class = {
            'CosineDecay': optax.warmup_cosine_decay_schedule(
                init_value=self.config.trans_lr,
                peak_value=self.config.trans_lr * 10,
                warmup_steps=self.config.warmup_steps,
                decay_steps=self.config.total_steps,
                end_value=self.config.trans_lr
            ),
            "OnlyWarmup": optax.join_schedules(
                [
                    optax.linear_schedule(
                        init_value=0.0,
                        end_value=self.config.trans_lr,
                        transition_steps=self.config.warmup_steps,
                    ),
                    optax.constant_schedule(
                        value=self.config.trans_lr
                    )
                ],
                [self.config.warmup_steps]
            ),
            'none': None
        }[self.config.scheduler_type]

        if scheduler_class:
            tx = optimizer_class(scheduler_class)
        else:
            tx = optimizer_class(learning_rate=self.config.trans_lr)

        trans_params = self.trans.init(
            {"params": next_rng(), "dropout": next_rng()},
            jnp.zeros((10, 25, self.observation_dim)),
            jnp.zeros((10, 25, self.action_dim)),
            jnp.ones((10, 25), dtype=jnp.int32)
        )
        self._train_states['trans'] = TrainState.create(
            params=trans_params,
            tx=tx,
            apply_fn=None
        )

        model_keys = ['trans']
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
        attn_mask = batch['attn_mask']

        train_params = {key: train_states[key].params for key in self.model_keys}
        trans_pred, attn_weights = self.trans.apply(train_params['trans'], obs, act, timestep, attn_mask=attn_mask, reverse=False)
        return trans_pred["value"], attn_weights[-1]
  
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
           
            trans_pred_1, _ = self.trans.apply(train_params['trans'], obs_1, act_1, timestep_1, training=False, attn_mask=None, rngs={"dropout": rng})
            trans_pred_2, _ = self.trans.apply(train_params['trans'], obs_2, act_2, timestep_2, training=False, attn_mask=None, rngs={"dropout": rng})
            
            if self.config.use_weighted_sum:
                trans_pred_1 = trans_pred_1["weighted_sum"]
                trans_pred_2 = trans_pred_2["weighted_sum"]
            else:
                trans_pred_1 = trans_pred_1["value"]
                trans_pred_2 = trans_pred_2["value"]

            if self.config.train_type == "mean":
                sum_pred_1 = jnp.mean(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.mean(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "sum":
                sum_pred_1 = jnp.sum(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.sum(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "last":
                sum_pred_1 = trans_pred_1.reshape(B, T)[:, -1].reshape(-1, 1)
                sum_pred_2 = trans_pred_2.reshape(B, T)[:, -1].reshape(-1, 1)
          
            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)
         
            loss_collection = {}

            rng, split_rng = jax.random.split(rng)
          
            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            trans_loss = cross_ent_loss(logits, label_target)
            cse_loss = trans_loss
            loss_collection['trans'] = trans_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), _ = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        metrics = dict(
            eval_cse_loss=aux_values['cse_loss'],
            eval_trans_loss=aux_values['trans_loss'],
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
           
            trans_pred_1, _ = self.trans.apply(train_params['trans'], obs_1, act_1, timestep_1, training=True, attn_mask=None, rngs={"dropout": rng})
            trans_pred_2, _ = self.trans.apply(train_params['trans'], obs_2, act_2, timestep_2, training=True, attn_mask=None, rngs={"dropout": rng})

            if self.config.use_weighted_sum:
                trans_pred_1 = trans_pred_1["weighted_sum"]
                trans_pred_2 = trans_pred_2["weighted_sum"]
            else:
                trans_pred_1 = trans_pred_1["value"]
                trans_pred_2 = trans_pred_2["value"]

            if self.config.train_type == "mean":
                sum_pred_1 = jnp.mean(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.mean(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "sum":
                sum_pred_1 = jnp.sum(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.sum(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "last":
                sum_pred_1 = trans_pred_1.reshape(B, T)[:, -1].reshape(-1, 1)
                sum_pred_2 = trans_pred_2.reshape(B, T)[:, -1].reshape(-1, 1)
           
            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)
           
            loss_collection = {}

            rng, split_rng = jax.random.split(rng)
           
            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            trans_loss = cross_ent_loss(logits, label_target)
            cse_loss = trans_loss

            loss_collection['trans'] = trans_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            cse_loss=aux_values['cse_loss'],
            trans_loss=aux_values['trans_loss'],
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
        def compute_logits(train_params, batch, rng):
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
           
            trans_pred_1, _ = self.trans.apply(train_params['trans'], obs_1, act_1, timestep_1, training=True, attn_mask=None, rngs={"dropout": rng})
            trans_pred_2, _ = self.trans.apply(train_params['trans'], obs_2, act_2, timestep_2, training=True, attn_mask=None, rngs={"dropout": rng})

            if self.config.use_weighted_sum:
                trans_pred_1 = trans_pred_1["weighted_sum"]
                trans_pred_2 = trans_pred_2["weighted_sum"]
            else:
                trans_pred_1 = trans_pred_1["value"]
                trans_pred_2 = trans_pred_2["value"]

            if self.config.train_type == "mean":
                sum_pred_1 = jnp.mean(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.mean(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "sum":
                sum_pred_1 = jnp.sum(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.sum(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "last":
                sum_pred_1 = trans_pred_1.reshape(B, T)[:, -1].reshape(-1, 1)
                sum_pred_2 = trans_pred_2.reshape(B, T)[:, -1].reshape(-1, 1)
           
            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)
            return logits, labels

        def loss_fn(train_params, lmd, tau, rng):
            rng, _ = jax.random.split(rng)
            logits, labels = compute_logits(train_params, labeled_batch, rng)
            u_logits, _ = compute_logits(train_params, unlabeled_batch, rng)
                        
            loss_collection = {}

            rng, split_rng = jax.random.split(rng)
            
            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            trans_loss = cross_ent_loss(logits, label_target)

            u_confidence = jnp.max(jax.nn.softmax(u_logits, axis=-1), axis=-1)
            pseudo_labels = jnp.argmax(u_logits, axis=-1)
            pseudo_label_target = jax.lax.stop_gradient(pseudo_labels)
                    
            loss_ = optax.softmax_cross_entropy(logits=u_logits, labels=jax.nn.one_hot(pseudo_label_target, num_classes=2))
            u_trans_loss = jnp.sum(jnp.where(u_confidence > tau, loss_, 0)) / (jnp.count_nonzero(u_confidence > tau) + 1e-4)
            u_trans_ratio = jnp.count_nonzero(u_confidence > tau) / len(u_confidence) * 100

            # labeling neutral cases.
            binarized_idx = jnp.where(unlabeled_batch["labels"][:, 0] != 0.5, 1., 0.)
            real_label = jnp.argmax(unlabeled_batch["labels"], axis=-1)
            u_trans_acc = jnp.sum(jnp.where(pseudo_label_target == real_label, 1., 0.) * binarized_idx) / jnp.sum(binarized_idx) * 100

            loss_collection['trans'] = last_loss = trans_loss + lmd * u_trans_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, lmd, tau, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            trans_loss=aux_values['trans_loss'],
            u_trans_loss=aux_values['u_trans_loss'],
            last_loss=aux_values['last_loss'],
            u_trans_ratio=aux_values['u_trans_ratio'],
            u_train_acc=aux_values['u_trans_acc']
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
