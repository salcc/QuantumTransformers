import flax.linen


def MLP(hidden_size: int, qml_backend="pennylane", qdevice="default.qubit", qdiff_method="best", use_catalyst=False):
    if qml_backend == "pennylane":
        from quantum_transformers.qmlperfcomp.jax_backend.quantum.pennylane_backend import QuantumLayer, get_circuit
    elif qml_backend == "tensorcircuit":
        from quantum_transformers.quantum_layer import QuantumLayer, get_circuit
    else:
        raise ValueError(f"Unknown qml_backend: {qml_backend}")

    circuit = get_circuit(num_qubits=hidden_size, qdevice=qdevice, diff_method=qdiff_method, use_catalyst=use_catalyst)

    class MLP(flax.linen.Module):
        hidden_size: int

        @flax.linen.compact
        def __call__(self, x, **_):
            x = flax.linen.Dense(self.hidden_size)(x)
            x = flax.linen.relu(x)
            x = QuantumLayer(circuit, num_qubits=self.hidden_size)(x)
            x = flax.linen.relu(x)
            x = flax.linen.Dense(1)(x)
            return x

    return MLP(hidden_size=hidden_size)
