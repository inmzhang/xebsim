"""Use single qubit XEB to benchmark the effect of noise injection."""

from typing import Sequence, Tuple, Callable, List

import tqdm
import numpy as np
import cirq
import pandas as pd

from sim.utils import fit_exponential_decays, linear_xeb_between_probvectors


def single_qubit_xeb_sim(
        circuits: Sequence[cirq.Circuit],
        noise_model: cirq.NoiseModel,
        *,
        cycle_depths: Sequence[int],
        cycle_to_circuit_depth_func: Callable[[int], int] = lambda x: x + 1,
) -> Tuple[float, float]:
    pure_sim = cirq.Simulator()
    noisy_sim = cirq.DensityMatrixSimulator(noise=noise_model)
    qubit = list(circuits[0].all_qubits())[0]

    records = []
    for circuit_i, circuit in enumerate(circuits):
        pure_step_results = [np.abs(res.state_vector()) ** 2 for res in pure_sim.simulate_moment_steps(circuit)]
        noisy_step_results = [
            np.abs(np.diag(res.density_matrix()))
            for res in noisy_sim.simulate_moment_steps(circuit)
        ]

        for cycle_depth in sorted(cycle_depths):
            circuit_depth = cycle_to_circuit_depth_func(cycle_depth)
            assert circuit_depth <= len(circuit)
            pure_probs = pure_step_results[circuit_depth - 1]
            noisy_probs = noisy_step_results[circuit_depth - 1]

            # Save the results
            records += [{
                'circuit_i': circuit_i,
                'cycle_depth': cycle_depth,
                'circuit_depth': circuit_depth,
                'pure_probs': pure_probs,
                'sampled_probs': noisy_probs,
            }]
    for record in records:
        e_u = np.sum(record['pure_probs']**2)
        u_u = np.sum(record['pure_probs']) / 2
        m_u = np.sum(record['pure_probs'] * record['sampled_probs'])
        record.update(
            e_u=e_u,
            u_u=u_u,
            m_u=m_u,
        )

    df = pd.DataFrame(records)
    df['y'] = df['m_u'] - df['u_u']
    df['x'] = df['e_u'] - df['u_u']

    df['numerator'] = df['x'] * df['y']
    df['denominator'] = df['x'] ** 2

    def per_cycle_depth(df):
        fid_lsq = df['numerator'].sum() / df['denominator'].sum()
        return pd.Series({'fidelity': fid_lsq})

    fids = df.groupby(['cycle_depth']).apply(per_cycle_depth).reset_index()
    fids['qubit'] = [qubit] * len(fids)

    fit_df = fit_exponential_decays(fids)

    fit_row = fit_df.iloc[0]
    return 1 - fit_row['layer_fid'], fit_row['layer_fid_std']


def large_scale_xeb_sim(
        circuit: cirq.Circuit,
        noise_model: cirq.NoiseModel,
        *,
        cycle_depths: Sequence[int],
        cycle_to_circuit_depth_func: Callable[[int], int] = lambda x: 2 * x + 1,
) -> List[float]:
    n = len(circuit.all_qubits())

    circuit = circuit[:cycle_to_circuit_depth_func(max(cycle_depths))]
    pure_sim = cirq.Simulator()
    noisy_sim = cirq.DensityMatrixSimulator(noise=noise_model)
    dim = 2 ** len(circuit.all_qubits())

    pure_step_results = []
    for res in tqdm.tqdm(
            pure_sim.simulate_moment_steps(circuit),
            total=len(circuit),
            desc=f"{n} qubits pure state simulation"
    ):
        pure_step_results.append(np.abs(res.state_vector()) ** 2)

    noisy_step_results = []
    for res in tqdm.tqdm(
            noisy_sim.simulate_moment_steps(circuit),
            total=len(circuit),
            desc=f"{n} qubits noisy state simulation"
    ):
        noisy_step_results.append(np.abs(np.diag(res.density_matrix())))

    xeb_results = []
    for cycle_depth in sorted(cycle_depths):
        circuit_depth = cycle_to_circuit_depth_func(cycle_depth)
        assert circuit_depth <= len(circuit)
        pure_probs = pure_step_results[circuit_depth - 1]
        noisy_probs = noisy_step_results[circuit_depth - 1]
        xeb_results.append(linear_xeb_between_probvectors(pure_probs, noisy_probs, dim))
    return xeb_results
