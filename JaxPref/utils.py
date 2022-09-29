import random
import pprint
import time
import uuid
import tempfile
import os
from copy import copy
from socket import gethostname
import cloudpickle as pickle

import numpy as np

import absl.flags
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import config_dict

import wandb

from .jax_utils import init_rng


class Timer(object):

    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time


class WandBLogger(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.online = False
        config.prefix = ''
        config.project = 'PrefRL'
        config.output_dir = './reward_model'
        config.random_delay = 0.0
        config.group = config_dict.placeholder(str)
        config.experiment_id = config_dict.placeholder(str)
        config.anonymous = config_dict.placeholder(str)
        config.notes = config_dict.placeholder(str)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, variant):
        self.config = self.get_default_config(config)

        if self.config.experiment_id is None:
            self.config.experiment_id = uuid.uuid4().hex

        if self.config.prefix != '':
            self.config.project = '{}--{}'.format(self.config.prefix, self.config.project)

        if self.config.output_dir == '':
            self.config.output_dir = tempfile.mkdtemp()
        else:
            # self.config.output_dir = os.path.join(self.config.output_dir, self.config.experiment_id)
            os.makedirs(self.config.output_dir, exist_ok=True)

        self._variant = copy(variant)

        if 'hostname' not in self._variant:
            self._variant['hostname'] = gethostname()

        if self.config.random_delay > 0:
            time.sleep(np.random.uniform(0, self.config.random_delay))

        self.run = wandb.init(
            reinit=True,
            config=self._variant,
            project=self.config.project,
            dir=self.config.output_dir,
            group=self.config.group,
            name=self.config.experiment_id,
            # anonymous=self.config.anonymous,
            notes=self.config.notes,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,
            ),
            mode='online' if self.config.online else 'offline',
        )

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)

    def save_pickle(self, obj, filename):
        with open(os.path.join(self.config.output_dir, filename), 'wb') as fout:
            pickle.dump(obj, fout)

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self.config.variant

    @property
    def output_dir(self):
        return self.config.output_dir


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, 'automatically defined flag')
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, 'automatically defined flag')
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, 'automatically defined flag')
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, 'automatically defined flag')
        else:
            raise ValueError('Incorrect value type')
    return kwargs


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    init_rng(seed)


def print_flags(flags, flags_def):
    logging.info(
        'Running training with hyperparameters: \n{}'.format(
            pprint.pformat(
                ['{}: {}'.format(key, val) for key, val in get_user_flags(flags, flags_def).items()]
            )
        )
    )


def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            output[key] = val

    return output


def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if prefix is not None:
            next_prefix = '{}.{}'.format(prefix, key)
        else:
            next_prefix = key
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=next_prefix))
        else:
            output[next_prefix] = val
    return output


def save_pickle(obj, filename, output_dir):
    with open(os.path.join(output_dir, filename), 'wb') as fout:
        pickle.dump(obj, fout)
            
def prefix_metrics(metrics, prefix):
    return {
        '{}/{}'.format(prefix, key): value for key, value in metrics.items()
    }
