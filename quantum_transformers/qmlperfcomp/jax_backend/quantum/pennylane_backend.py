from typing import Callable

import jax
import jax.numpy as jnp
import flax.linen as nn

import pennylane as qml


def get_circuit(num_qubits, qdevice="default.qubit.jax", diff_method="best", use_catalyst=False):
    dev = qml.device(qdevice, wires=num_qubits)

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def circuit(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(num_qubits))
        qml.templates.BasicEntanglerLayers(weights, wires=range(num_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]

    if not use_catalyst:
        jitted_circuit = jax.jit(circuit)
    else:
        import catalyst
        jitted_circuit = catalyst.qjit(circuit)

    return jitted_circuit


class QuantumLayer(nn.Module):
    circuit: Callable
    num_qubits: int
    num_layers: int = 1

    @nn.compact
    def __call__(self, x):
        shape = x.shape
        x = jnp.reshape(x, (-1, shape[-1]))
        weights = self.param('w', nn.initializers.xavier_normal(), (self.num_layers, self.num_qubits))
        x = jax.vmap(self.circuit, in_axes=(0, None))(x, weights)
        x = jnp.concatenate(x, axis=-1)
        x = jnp.reshape(x, tuple(shape))
        return x
