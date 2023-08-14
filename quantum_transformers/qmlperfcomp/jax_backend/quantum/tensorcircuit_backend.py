
from typing import Callable

import jax.numpy as jnp
import flax.linen

class QuantumLayer(flax.linen.Module):
    circuit: Callable
    num_qubits: int
    num_layers: int = 1
    w_init: Callable = flax.linen.initializers.xavier_normal()

    @flax.linen.compact
    def __call__(self, x):
        shape = x.shape
        x = jnp.reshape(x, (-1, shape[-1]))
        weights = self.param('w', self.w_init, (self.num_layers, self.num_qubits))
        x = self.circuit(x, weights)
        x = jnp.concatenate(x, axis=-1)
        x = jnp.reshape(x, tuple(shape))
        return x
