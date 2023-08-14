import tensorcircuit as tc
import jax.numpy as jnp

K = tc.set_backend("jax")


def get_quantum_layer_circuit(inputs, weights):
    num_qubits = inputs.shape[-1]
    num_qlayers = weights.shape[-2]

    c = tc.Circuit(num_qubits)

    for j in range(num_qubits):
        c.rx(j, theta=inputs[j])

    for i in range(num_qlayers):
        for j in range(num_qubits):
            c.rx(j, theta=weights[i, j])
        if num_qubits == 2:
            c.cnot(0, 1)
        elif num_qubits > 2:
            for j in range(num_qubits):
                c.cnot(j, (j + 1) % num_qubits)

    return c


def get_circuit(torch_interface: bool = False, **_):
    def qpred(inputs, weights):
        c = get_quantum_layer_circuit(inputs, weights)
        return K.real(jnp.array([c.expectation_ps(z=[i]) for i in range(weights.shape[1])]))

    qpred_batch = K.vmap(qpred, vectorized_argnums=0)
    if torch_interface:
        qpred_batch = tc.interfaces.torch_interface(qpred_batch, jit=True)

    return qpred_batch
