import itertools

import cirq
import numpy as np
import sys
sys.path.append(".")

from sim.circuit import gen_1d_chain_xeb_random_circuit
from sim.simulate import large_scale_xeb_sim_qsim

exponents = np.linspace(0, 7/4, 8)
SINGLE_QUBIT_GATES = tuple(
    cirq.PhasedXZGate(x_exponent=0.5, z_exponent=z, axis_phase_exponent=a)
    for a, z in itertools.product(exponents, repeat=2)
)

MAX_DEPTH = 400
cycle_depths = list(range(1, 21)) + [25, 35, 40]
e_pauli = 3e-3
e_dep = e_pauli / (1 - 1/2**2)
noise_model = cirq.devices.noise_model.ConstantQubitNoiseModel(cirq.depolarize(e_pauli))
save_resume_filepath = "/home/cuquantum/xebsim/result/1d_chain/result.csv"

ns = [6, 8, 10, 12, 14, 16, 18, 20]
n_circuits = 10
shots = 50_000
results = []
for n in ns:
    qubits = cirq.LineQubit.range(n)
    circuits = [
        gen_1d_chain_xeb_random_circuit(
            qubits,
            depth=MAX_DEPTH,
            single_qubit_gates=SINGLE_QUBIT_GATES,
            two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b)
        )
        for _ in range(n_circuits)
    ]
    for circuit in circuits:
        results.append(
            large_scale_xeb_sim_qsim(circuit, noise_model, cycle_depths=cycle_depths, shots=shots, save_resume_filepath=save_resume_filepath)
        )