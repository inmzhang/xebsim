import collections

import cirq
import numpy as np
import matplotlib.pyplot as plt

import xebsim

MAX_DEPTH = 200

link_freqs = [8, 12, 18]
cycles = [[18, 26, 32], [20, 32, 44], [30, 48, 66]]
N = 10
n_circuits = 5

for link_freq, cycle_depths in zip(link_freqs, cycles):
    qubits = cirq.LineQubit.range(N)
    circuits = [
        xebsim.gen_1d_chain_weak_link_rqc(
            qubits,
            depth=MAX_DEPTH,
            link_frequency=link_freq,
            single_qubit_gates=xebsim.phasedxz_gateset(),
            two_qubit_gate=cirq.SQRT_ISWAP,
        )
        for _ in range(n_circuits)
    ]
    res = collections.defaultdict(list)

    e_paulis = np.linspace(0.5e-3, 1e-2, 15)
    for e_pauli in e_paulis:
        dim = 2 ** N
        e_dep = e_pauli / (1 - 1/dim**2)
        noise_model = cirq.devices.noise_model.ConstantQubitNoiseModel(cirq.depolarize(e_pauli))
        
        xebs = np.mean([
            xebsim.density_matrix_simulation(circuit, noise_model, cycle_depths=cycle_depths)
            for circuit in circuits
        ], axis=0)
        for i, d in enumerate(cycle_depths):
            estimated_fidelity = (1 - e_dep) ** ((2 * d + 1) * N)
            ratio = estimated_fidelity / xebs[i]
            res[d].append(ratio)

    error_per_cycle = (e_paulis / (1 - 1/dim**2)) * 2 * N

    fig, ax = plt.subplots()
    for d in cycle_depths:
        ratios = res[d]
        ax.plot(error_per_cycle, ratios, label=f"{d}-th cycle", marker="o")
    ax.set_xlabel("Error per cycle")
    ax.set_ylabel("XEB ratio (pred./meas.)")
    ax.set_yscale("log")
    ax.set_yticks([1e-1, 1])
    ax.set_title(f"T={link_freq}")
    fig.savefig(f"T={link_freq}.png")