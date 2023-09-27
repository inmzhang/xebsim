import itertools

import cirq
import numpy as np
import sys
sys.path.append(".")

from sim.circuit import gen_2d_grid_xeb_random_circuits
from sim.simulate import large_scale_xeb_sim, large_scale_xeb_sim_qsim

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
n_circuits = 5
save_resume_filepath = "/home/cuquantum/xebsim/result/2d_grid/result.csv"


grids = [(3, 3), (3, 4), (4, 4)]
# grids = [(3, 3)]
results = []
shots = 50_000
for grid in grids:
    qubits = cirq.GridQubit.rect(grid[0], grid[1])
    circuits, _ = gen_2d_grid_xeb_random_circuits(
        qubits=qubits,
        depth=MAX_DEPTH,
        num_circuits=n_circuits,
        two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b)
    )
    for circuit in circuits:
        results.append(
            large_scale_xeb_sim_qsim(circuit, noise_model, cycle_depths=cycle_depths, task_name="2d_grid_linear_xeb", shots=shots, save_resume_filepath=save_resume_filepath)
        )