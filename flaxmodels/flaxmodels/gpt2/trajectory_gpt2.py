import jax.numpy as jnp
import flax.linen as nn
from typing import Any
import h5py

from .. import utils
from . import ops


URLS = {'gpt2': 'https://www.dropbox.com/s/0wdgj0gazwt9nm7/gpt2.h5?dl=1',
        'gpt2-medium': 'https://www.dropbox.com/s/nam11kbd83wsm7d/gpt2-medium.h5?dl=1',
        'gpt2-large': 'https://www.dropbox.com/s/oy8623qwkkjm8gt/gpt2-large.h5?dl=1',
        'gpt2-xl': 'https://www.dropbox.com/s/6c6qt0bzz4v2afx/gpt2-xl.h5?dl=1'}

CONFIGS = {'gpt2': 'https://www.dropbox.com/s/s5xl32dgwc8322p/gpt2.json?dl=1',
           'gpt2-medium': 'https://www.dropbox.com/s/7mwkijxoh1earm5/gpt2-medium.json?dl=1',
           'gpt2-large': 'https://www.dropbox.com/s/nhslkxwxtpn7auz/gpt2-large.json?dl=1',
           'gpt2-xl': 'https://www.dropbox.com/s/1iv0nq1xigsfdvb/gpt2-xl.json?dl=1'}


class GPT2SelfAttention(nn.Module):
    """
    GPT2 Self Attention.

    Attributes:
        config (Any): Configuration object. If 'pretrained' is not None, this parameter will be ignored.
        param_dict (dict): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
    """
    config: dict = None
    
    def setup(self):
        self.max_pos = self.config.n_positions
        self.embd_dim = self.config.n_embd
        self.num_heads = self.config.n_head
        self.head_dim = self.embd_dim // self.num_heads
        self.attn_dropout = self.config.attn_pdrop
        self.resid_dropout = self.config.resid_pdrop
        self.scale_attn_weights = True
        
    @nn.compact
    def __call__(self, x, layer_past=None, attn_mask=None, head_mask=None, use_cache=False, training=False):
        """
        Run attention.

        Args:
            x (tensor): Input tensor.
            layer_past (Tuple): Tuple of past keys and values.
            attn_mask (tensor): Mask to avoid performing attention on padding token indices.
            head_mask (tensor): Mask to nullify selected heads of the self-attention modules.
            use_cache (bool): If True, keys and values are returned (past_key_values).
            training (bool): Training mode.

        Returns:
            (tensor, Tuple): Output tensor, tuple of keys and values.
        """
        x = nn.Dense(features=3*self.embd_dim)(x)

        query, key, value = jnp.split(x, 3, axis=2)
        
        query = ops.split_heads(query, self.num_heads, self.head_dim)
        value = ops.split_heads(value, self.num_heads, self.head_dim)
        key = ops.split_heads(key, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = jnp.concatenate((past_key, key), axis=-2)
            value = jnp.concatenate((past_value, value), axis=-2)

        present = (key, value) if use_cache else None

        query_len, key_len = query.shape[-2], key.shape[-2]
        casual_mask = jnp.tril(jnp.ones((1, 1, self.max_pos, self.max_pos)))[:, :, key_len - query_len :key_len, :key_len]
        # casual_mask = jnp.ones((1, 1, self.max_pos, self.max_pos))[:, :, key_len - query_len :key_len, :key_len]
        casual_mask = casual_mask.astype(bool)

        attn_dropout = nn.Dropout(rate=self.attn_dropout)
        out, _attn_weights = ops.attention(query, key, value, casual_mask, -1e4, attn_dropout, self.scale_attn_weights, training, attn_mask, head_mask)
        out = ops.merge_heads(out, self.num_heads, self.head_dim)

        out = nn.Dense(features=self.embd_dim)(out)

        out = nn.Dropout(rate=self.resid_dropout)(out, deterministic=not training)
        return out, present, _attn_weights


class GPT2MLP(nn.Module):
    """
    GPT2 MLP.

    Attributes:
        intermediate_dim (int): Dimension of the intermediate layer.
        config (Any): Configuration object. If 'pretrained' is not None, this parameter will be ignored.
        param_dict (dict): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
    """
    intermediate_dim: int
    config: dict = None
    
    def setup(self):
        self.embd_dim = self.config.n_embd
        self.resid_dropout = self.config.resid_pdrop
        self.activation = self.config.activation_function

    @nn.compact
    def __call__(self, x, training=False):
        """
        Run the MLP.

        Args:
            x (tensor): Input tensor.
            training (bool): Training mode.
        """
        x = nn.Dense(features=self.intermediate_dim)(x)
        x = ops.apply_activation(x, activation=self.activation)
        x = nn.Dense(features=self.embd_dim)(x)
        x = nn.Dropout(rate=self.resid_dropout)(x, deterministic=not training)
        return x


class GPT2Block(nn.Module):
    """
    GPT2 Block.

    Attributes:
        config (Any): Configuration object. If 'pretrained' is not None, this parameter will be ignored.
        param_dict (dict): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
    """
    config: dict = None
    
    def setup(self):
        self.embd_dim = self.config.n_embd
        self.eps = self.config.layer_norm_epsilon
        self.inner_dim = self.config.n_inner if self.config.n_inner is not None else 4 * self.embd_dim

    @nn.compact
    def __call__(self, x, layer_past=None, attn_mask=None, head_mask=None, use_cache=False, training=False):
        """
        Run the block.

        Args:
            x (tensor): Input tensor.
            layer_past (Tuple): Tuple of past keys and values.
            attn_mask (tensor): Mask to avoid performing attention on padding token indices.
            head_mask (tensor): Mask to nullify selected heads of the self-attention modules.
            use_cache (bool): If True, keys and values are returned (past_key_values).
            training (bool): Training mode.

        Returns:
            (tensor, Tuple): Output tensor, tuple of keys and values.
        """
        residual = x
        x = nn.LayerNorm(epsilon=self.eps)(x)
        kwargs = {'layer_past': layer_past, 'attn_mask': attn_mask, 'head_mask': head_mask,
                  'use_cache': use_cache, 'training': training}
        x, present, _attn_weights = GPT2SelfAttention(config=self.config)(x, **kwargs)
        x += residual
        residual = x
        x = nn.LayerNorm(epsilon=self.eps)(x)
        x = GPT2MLP(intermediate_dim=self.inner_dim, config=self.config)(x, training)
        x += residual
        return x, present, _attn_weights


class GPT2Model(nn.Module):
    """
    The GPT2 Model.

    Attributes:
        config (Any): Configuration object. If 'pretrained' is not None, this parameter will be ignored.
        pretrained (str): Which pretrained model to use, None for random initialization.
        ckpt_dir (str): Directory to which the pretrained weights are downloaded. If None, a temp directory will be used.
        param_dict (dict): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
    """
    config: dict = None
    pretrained: str = None
    ckpt_dir: str = None
    
    def setup(self):
        assert self.pretrained is None, "pretrain must be None for training."
        if self.pretrained is not None:
            assert self.pretrained in URLS.keys(), f'Pretrained model not available {self.pretrained}.'
            ckpt_file = utils.download(self.ckpt_dir, URLS[self.pretrained])
            self.param_dict_ = h5py.File(ckpt_file, 'r')['transformer']
            config_file = utils.download(self.ckpt_dir, CONFIGS[self.pretrained])
            self.config_ = ops.load_config(config_file)
        else:
            self.config_ = self.config
        self.vocab_size = self.config_.vocab_size
        self.max_pos = self.config_.n_positions
        self.embd_dim = self.config_.n_embd
        self.embd_dropout = self.config_.embd_pdrop
        self.num_layers = self.config_.n_layer
        self.eps = self.config_.layer_norm_epsilon

    @nn.compact
    def __call__(self,
                 input_ids=None,
                 past_key_values=None,
                 input_embds=None,
                 position_ids=None,
                 attn_mask=None,
                 head_mask=None,
                 use_cache=False,
                 training=False
                 ):
        """
        Run the model.

        Args:
            input_ids (tensor): Input token ids, shape [B, seq_len].
            past_key_values (Tuple): Precomputed hidden keys and values, tuple of tuples.
                                     If past_key_values is used, only input_ids that do not have their
                                     past calculated should be passed as input_ids.
            input_embds (tensor): Input embeddings, shape [B, seq_len, embd_dim].
            labels (tensor): Labels for language modeling, shape [B, seq_len]. Will be shifted inside the model. Ignore label = -100.
            position_ids (tensor): Indices of positions of each input sequence tokens in the position embeddings, shape [B, seq_len].
            attn_mask (tensor): Mask to avoid performing attention on padding token indices, shape [B, seq_len].
            head_mask (tensor): Mask to nullify selected heads of the self-attention modules, shape [num_heads] or [num_layers, num_heads].
            use_cache (bool): If True, keys and values are returned (past_key_values).
            training (bool): Training mode.

        Returns:
            (dict): Dictionary containing 'last_hidden_state', 'past_key_values'.            
        """
        if input_ids is not None and input_embds is not None:
            raise ValueError('You cannot specify both input_ids and input_embd at the same time.')
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = jnp.reshape(input_ids, newshape=(-1, input_shape[-1]))
            batch_size = input_ids.shape[0]
        elif input_embds is not None:
            input_shape = input_embds.shape[:-1]
            batch_size = input_embds.shape[0]
        else:
            raise ValueError('You have to specify either input_ids or input_embd.')

        if position_ids is not None:
            position_ids = jnp.reshape(position_ids, newshape=(-1, input_shape[-1]))

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.num_layers)
        else:
            past_length = past_key_values[0][0].shape[-2]
        
        if position_ids is None:
            position_ids = jnp.arange(start=past_length, stop=input_shape[-1] + past_length)
            position_ids = jnp.reshape(jnp.expand_dims(position_ids, axis=0), newshape=(-1, input_shape[-1])) 

        if input_embds is None:
            input_embds = nn.Embed(num_embeddings=self.vocab_size, features=self.embd_dim)(input_ids)

        if attn_mask is not None:
            attn_mask = ops.get_attention_mask(attn_mask, batch_size)

        if head_mask is not None:
            head_mask = ops.get_head_mask(head_mask, self.num_layers)
        else:
            head_mask = [None] * self.num_layers
        
        # position_embds = nn.Embed(num_embeddings=self.max_pos, features=self.embd_dim)(position_ids)

        # x = input_embds + position_embds
        x = input_embds
        
        x = nn.Dropout(rate=self.embd_dropout)(x, deterministic=not training)
        output_shape = input_shape + (x.shape[-1],)

        presents = () if use_cache else None
        attn_weights_list = []
        for i in range(self.num_layers):
            kwargs = {'layer_past': past_key_values[i], 'attn_mask': attn_mask, 'head_mask': head_mask[i],
                      'use_cache': use_cache, 'training': training}
            x, present, attn_weights = GPT2Block(config=self.config_)(x, **kwargs)

            if use_cache:
                presents = presents + (present,)
            attn_weights_list.append(attn_weights)

        x = nn.LayerNorm(epsilon=self.eps)(x)
        return {'last_hidden_state': x, 'past_key_values': presents, 'attn_weights_list': attn_weights_list}


class TransRewardModel(nn.Module):
    config: Any = None
    pretrained: str = None
    ckpt_dir: str = None
    observation_dim: int = 29
    action_dim: int = 8
    activation: str = None
    activation_final: str = None
    max_episode_steps: int = 1000

    def setup(self):
        self.config_ = self.config
        self.config_.activation_function = self.activation
        self.config_.activation_final = self.activation_final
        self.vocab_size = self.config_.vocab_size
        self.max_pos = self.config_.n_positions
        self.embd_dim = self.config_.n_embd
        self.pref_attn_embd_dim = self.config_.pref_attn_embd_dim
        self.embd_dropout = self.config_.embd_pdrop
        self.attn_dropout = self.config_.attn_pdrop
        self.resid_dropout = self.config_.resid_pdrop
        self.num_layers = self.config_.n_layer
        self.inner_dim = self.config_.n_embd // 2
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
        target_idx=1,
    ):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attn_mask is None:
            attn_mask = jnp.ones((batch_size, seq_length), dtype=jnp.float32)

        embd_state = nn.Dense(features=self.embd_dim)(states)
        embd_action = nn.Dense(features=self.embd_dim)(actions)
        embd_timestep = nn.Embed(num_embeddings=self.max_episode_steps + 1, features=self.embd_dim)(timesteps)

        embd_state = embd_state + embd_timestep
        embd_action = embd_action + embd_timestep

        if reverse:
            stacked_inputs = jnp.stack(
                [embd_state, embd_action],
                axis=1
            ).transpose(0, 2, 1, 3).reshape(batch_size, 2 * seq_length, self.embd_dim)
        else:
            stacked_inputs = jnp.stack(
                [embd_action, embd_state],
                axis=1
            ).transpose(0, 2, 1, 3).reshape(batch_size, 2 * seq_length, self.embd_dim)

        stacked_inputs = nn.LayerNorm(epsilon=self.eps)(stacked_inputs)

        stacked_attn_mask = jnp.stack(
            [attn_mask, attn_mask],
            axis=1
        ).transpose(0, 2, 1).reshape(batch_size, 2 * seq_length)

        transformer_outputs = GPT2Model(
            config=self.config
        )(
            input_embds=stacked_inputs,
            attn_mask=stacked_attn_mask,
            training=training,
        )
        
        x = transformer_outputs["last_hidden_state"]
        attn_weights_list = transformer_outputs["attn_weights_list"]
        x = x.reshape(batch_size, seq_length, 2, self.embd_dim).transpose(0, 2, 1, 3)
        hidden_output = x[:, target_idx]

        if self.config_.use_weighted_sum:
            '''
            add additional Attention Layer for Weighted Sum.
            x (= output, tensor): Predicted Reward, shape [B, seq_len, embd_dim]
            ''' 
            x = nn.Dense(features=2 * self.pref_attn_embd_dim + 1)(hidden_output)
            # only one head, because value has 1 dim for predicting rewards directly.
            num_heads = 1

            # query: [B, seq_len, embd_dim]
            # key: [B, seq_len, embd_dim]
            # value: [B, seq_len, 1]

            query, key, value = jnp.split(x, [self.pref_attn_embd_dim, self.pref_attn_embd_dim * 2], axis=2)
            query = ops.split_heads(query, num_heads, self.pref_attn_embd_dim)
            key = ops.split_heads(key, num_heads, self.pref_attn_embd_dim)
            value = ops.split_heads(value, num_heads, 1)

            # query: [B, 1, seq_len, embd_dim]
            # key: [B, 1, seq_len, embd_dim]
            # value: [B, 1, seq_len, 1]

            query_len, key_len = query.shape[-2], key.shape[-2]
            # casual_mask = jnp.tril(jnp.ones((1, 1, self.config_.n_positions, self.config_.n_positions)))[:, :, key_len - query_len :key_len, :key_len]
            # casual_mask = casual_mask.astype(bool)
            casual_mask = jnp.ones((1, 1, seq_length, seq_length))[:, :, key_len - query_len :key_len, :key_len]
            casual_mask = casual_mask.astype(bool)

            # attn_dropout = nn.Dropout(rate=self.attn_dropout) # split dropout rate
            attn_dropout = nn.Dropout(rate=0.0) # boilerplate code.
            new_attn_mask = ops.get_attention_mask(attn_mask, batch_size)
            
            out, last_attn_weights = ops.attention(query, key, value, casual_mask, -1e-4, attn_dropout, scale_attn_weights=True, training=training, attn_mask=new_attn_mask, head_mask=None)
            attn_weights_list.append(last_attn_weights)
            # out: [B, 1, seq_len, 1]
            output = ops.merge_heads(out, num_heads, 1)
            # output: [B, seq_len, 1]

            # output = nn.Dropout(rate=self.resid_dropout)(out, deterministic=not training)
            return {"weighted_sum": output, "value": value}, attn_weights_list

        else:
            x = nn.Dense(features=self.inner_dim)(hidden_output)
            x = ops.apply_activation(x, activation=self.activation)
            output = nn.Dense(features=1)(x)
            if self.activation_final != 'none':
                output = ops.apply_activation(output, activation=self.activation_final)

            return {"value": output}, attn_weights_list
