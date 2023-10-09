import itertools
import sys

import cirq
import numpy as np
import matplotlib.pyplot as plt

from sim.circuit import gen_1d_chain_xeb_random_circuit
from sim.simulate import large_scale_xeb_sim

exponents = np.linspace(0, 7/4, 8)
SINGLE_QUBIT_GATES = tuple(
    cirq.PhasedXZGate(x_exponent=0.5, z_exponent=z, axis_phase_exponent=a)
    for a, z in itertools.product(exponents, repeat=2)
)

MAX_DEPTH = 200
N = 10
n_circuits = 10

qubits = cirq.LineQubit.range(N)
circuits = [gen_1d_chain_xeb_random_circuit
    (
        qubits,
        depth=MAX_DEPTH,
        single_qubit_gates=SINGLE_QUBIT_GATES,
        two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b)
    )
    for _ in range(n_circuits)
]

cycle_depths = np.arange(1, 51)
def run(e_pauli):
    noise_model = cirq.devices.noise_model.ConstantQubitNoiseModel(cirq.depolarize(e_pauli))
    
    xebs = np.mean([
        large_scale_xeb_sim(circuit, noise_model, cycle_depths=cycle_depths)
        for circuit in circuits
    ], axis=0)
    return e_pauli, xebs