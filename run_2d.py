import cirq

import xebsim

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
    circuits, _ = xebsim.gen_2d_grid_rqc(
        qubits=qubits,
        depth=MAX_DEPTH,
        num_circuits=n_circuits,
        single_qubit_gates=xebsim.phasedxz_gateset(),
        two_qubit_gate=cirq.SQRT_ISWAP,
    )
    for circuit in circuits:
        results.append(
            xebsim.qsim_trajectory_simulation(circuit, noise_model, cycle_depths=cycle_depths, task_name="2d_grid_linear_xeb", shots=shots, save_resume_filepath=save_resume_filepath)
        )
