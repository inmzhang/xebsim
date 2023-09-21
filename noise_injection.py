"""Use single qubit XEB to benchmark the effect of noise injection."""

import itertools
from typing import Sequence, Tuple, Iterable, Callable

import numpy as np
import cirq
import pandas as pd

from utils import fit_exponential_decays

# Random single qubit gates
exponents = np.linspace(0, 7/4, 8)
SINGLE_QUBIT_GATES = tuple(
    cirq.PhasedXZGate(x_exponent=0.5, z_exponent=z, axis_phase_exponent=a)
    for a, z in itertools.product(exponents, repeat=2)
)


class MixedNoiseModel(cirq.NoiseModel):
    def __init__(self, p: float, noise_amplitude_mrad: float):
        self.p = p
        # TODO: check the definition
        self.noise_amplitude = noise_amplitude_mrad / 1e3

    def noisy_moment(
            self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']
    ) -> 'cirq.OP_TREE':
        if any(isinstance(op.gate, cirq.DepolarizingChannel) for op in moment):
            return moment
        if all(len(op.qubits) == 1 for op in moment):
            size = len(system_qubits)
            random_a = np.random.normal(loc=-1, scale=1, size=size)
            random_x = np.random.normal(loc=0, scale=self.noise_amplitude, size=size)
            random_z = np.random.normal(loc=0, scale=self.noise_amplitude, size=size)
            coherent_noise_moment = cirq.Moment(
                cirq.PhasedXZGate(
                    x_exponent=random_x[i],
                    z_exponent=random_z[i],
                    axis_phase_exponent=random_a[i]
                )(system_qubits[i])
                for i in range(size)
            )
            incoherent_noise_moment = cirq.Moment(
                cirq.depolarize(p=self.p).on_each(system_qubits)
            )
            return [moment, coherent_noise_moment, incoherent_noise_moment]
        return moment


def step_simulate_with_noise_model(
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


def simulate_with_noise_model(
        circuits: Sequence[cirq.Circuit],
        noise_model: cirq.NoiseModel,
        *,
        cycle_depths: Sequence[int],
        shots: int = 10_000,
) -> Tuple[float, float]:
    pure_sim = cirq.Simulator()
    noisy_sims = []
    noisy_sim = cirq.DensityMatrixSimulator(noise=noise_model)
    noisy_sims.append(noisy_sim)
    qubit = list(circuits[0].all_qubits())[0]

    records = []
    for cycle_depth in cycle_depths:
        for circuit_i, circuit in enumerate(circuits):
            assert cycle_depth <= len(circuit)
            circuit_depth = cycle_depth + 1
            trunc_circuit = circuit[:circuit_depth]

            # Pure-state simulation
            psi = pure_sim.simulate(trunc_circuit).final_state_vector
            pure_probs = np.abs(psi)**2

            # Noisy execution
            meas_circuit = trunc_circuit + cirq.measure(qubit)
            sampled_inds = noisy_sim.sample(meas_circuit, repetitions=shots).values[:, 0]
            sampled_probs = np.bincount(sampled_inds, minlength=2) / len(sampled_inds)

            # Save the results
            records += [{
                'circuit_i': circuit_i,
                'cycle_depth': cycle_depth,
                'circuit_depth': circuit_depth,
                'pure_probs': pure_probs,
                'sampled_probs': sampled_probs,
            }]
        print('.', end='', flush=True)
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


def gen_single_qubit_xeb_random_circuits(
        qubit: cirq.LineQubit,
        *,
        depth: int,
        single_qubit_gates: Sequence['cirq.Gate'] = SINGLE_QUBIT_GATES,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
) -> cirq.Circuit:
    """Generate XEB circuits for a single qubit."""
    circuit = cirq.Circuit()
    random_state = cirq.value.parse_random_state(seed)
    for _ in range(depth + 1):
        circuit.append(random_state.choice(single_qubit_gates)(qubit))
    return circuit


def main():
    max_depth = 100
    n_circuits = 50
    q0 = cirq.LineQubit(0)
    circuits = [
        gen_single_qubit_xeb_random_circuits(q0, depth=max_depth)
        for _ in range(n_circuits)
    ]
    cycle_depths = np.arange(1, max_depth + 1, 9)



if __name__ == '__main__':
    main()
