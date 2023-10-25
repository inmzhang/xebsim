import cirq

import xebsim

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
        xebsim.gen_1d_chain_brickwork_rqc(
            qubits,
            depth=MAX_DEPTH,
            single_qubit_gates=xebsim.phasedxz_gateset(),
            two_qubit_gate=cirq.SQRT_ISWAP,
        )
        for _ in range(n_circuits)
    ]
    for circuit in circuits:
        results.append(
            xebsim.qsim_trajectory_simulation(circuit, noise_model, cycle_depths=cycle_depths, shots=shots, save_resume_filepath=save_resume_filepath)
        )