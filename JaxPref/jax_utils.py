import numpy as np
import jax
import jax.numpy as jnp
import optax

class JaxRNG(object):
    def __init__(self, seed):
        self.rng = jax.random.PRNGKey(seed)

    def __call__(self):
        self.rng, next_rng = jax.random.split(self.rng)
        return next_rng


def init_rng(seed):
    global jax_utils_rng
    jax_utils_rng = JaxRNG(seed)


def next_rng():
    global jax_utils_rng
    return jax_utils_rng()


def extend_and_repeat(tensor, axis, repeat):
    return jnp.repeat(jnp.expand_dims(tensor, axis), repeat, axis=axis)


def mse_loss(val, target):
    return jnp.mean(jnp.square(val - target))

def cross_ent_loss(logits, target):
    
    if len(target.shape) == 1:
        label = jax.nn.one_hot(target, num_classes=2)
    else:
        label = target
        
    loss = jnp.mean(optax.softmax_cross_entropy(
        logits=logits, 
        labels=label))
    return loss

def kld_loss(p, q):
    return jnp.mean(jnp.sum(jnp.where(p != 0, p * (jnp.log(p) - jnp.log(q)), 0), axis=-1))

def custom_softmax(array, axis=-1, temperature=1.0):
    array = array / temperature
    return jax.nn.softmax(array, axis=axis)


def pref_accuracy(logits, target):
    predicted_class = jnp.argmax(logits, axis=1)
    target_class = jnp.argmax(target, axis=1)
    return jnp.mean(predicted_class == target_class)

def value_and_multi_grad(fun, n_outputs, argnums=0, has_aux=False):
    def select_output(index):
        def wrapped(*args, **kwargs):
            if has_aux:
                x, *aux = fun(*args, **kwargs)
                return (x[index], *aux)
            else:
                x = fun(*args, **kwargs)
                return x[index]
        return wrapped

    grad_fns = tuple(
        jax.value_and_grad(select_output(i), argnums=argnums, has_aux=has_aux)
        for i in range(n_outputs)
    )
    def multi_grad_fn(*args, **kwargs):
        grads = []
        values = []
        for grad_fn in grad_fns:
            (value, *aux), grad = grad_fn(*args, **kwargs)
            values.append(value)
            grads.append(grad)
        return (tuple(values), *aux), tuple(grads)
    return multi_grad_fn


@jax.jit
def batch_to_jax(batch):
    return jax.tree_util.tree_map(jax.device_put, batch)
