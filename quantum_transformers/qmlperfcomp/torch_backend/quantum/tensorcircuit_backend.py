import torch

from quantum_transformers.qmlperfcomp.tc_common import get_circuit


class QuantumLayer(torch.nn.Module):
    def __init__(self, num_qubits, num_qlayers=1, **_):
        super().__init__()

        self.weights = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(num_qlayers, num_qubits)))
        self.linear = get_circuit(torch_interface=True)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        x = self.linear(x, self.weights)
        x = x.reshape(shape)
        return x
