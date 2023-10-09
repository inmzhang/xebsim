"""Use single qubit XEB to benchmark the effect of noise injection."""

from typing import Sequence, Tuple, Callable, List, Union
import time
import pathlib
import uuid

import tqdm
import numpy as np
import cirq
import pandas as pd

from sim.utils import fit_exponential_decays, linear_xeb_between_probvectors, linear_xeb_estimate, linear_xeb_std_err_estimate
from sim.result import SampleResult


CSV_HEADER = "task, size, circuit_id, noise_model, cycle_depth, shots, linear_xeb, linear_xeb_std_err, elapsed_time_sec, time_per_shot_sec"


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
        normalize: bool = False,
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
        normalizer = linear_xeb_between_probvectors(pure_probs, pure_probs, dim)
        xeb = linear_xeb_between_probvectors(pure_probs, noisy_probs, dim)
        if normalize:
            xeb = xeb / normalizer
        xeb_results.append(xeb)
    return xeb_results


def large_scale_xeb_sim_qsim(
        circuit: cirq.Circuit,
        noise_model: cirq.NoiseModel,
        *,
        cycle_depths: Sequence[int],
        cycle_to_circuit_depth_func: Callable[[int], int] = lambda x: 2 * x + 1,
        task_name: str = '',
        shots: int = 10_000,
        save_resume_filepath: Union[str, pathlib.Path]
) -> List[SampleResult]:
    import qsimcirq
    qubits = sorted(circuit.all_qubits())
    circuit_id = uuid.uuid4()
    circuit_filepath = f"/home/cuquantum/xebsim/result/circuits/{circuit_id}"
    cirq.to_json(circuit, circuit_filepath)
    
    save_resume_filepath = pathlib.Path(save_resume_filepath)
    if not save_resume_filepath.exists():
        save_resume_filepath.parent.mkdir(parents=True, exist_ok=True)
        save_resume_filepath.touch()
        with open(save_resume_filepath, 'w') as f:
            print(CSV_HEADER, file=f, flush=True)

    circuit = circuit[:cycle_to_circuit_depth_func(max(cycle_depths))]
    
    pure_gpu_options = qsimcirq.QSimOptions(disable_gpu=False, gpu_mode = 0, max_fused_gate_size=2)
    pure_sim = qsimcirq.QSimSimulator(qsim_options=pure_gpu_options)
    noisy_gpu_options = qsimcirq.QSimOptions(disable_gpu=False, gpu_mode = 1, max_fused_gate_size=2)
    noisy_sim = qsimcirq.QSimSimulator(qsim_options=noisy_gpu_options, noise=noise_model, circuit_memoization_size=1)
    
    dim = 2 ** len(qubits)
    
    results = []
    for cycle_depth in sorted(cycle_depths):
        circuit_depth = cycle_to_circuit_depth_func(cycle_depth)
        assert circuit_depth <= len(circuit)

        res = pure_sim.simulate(circuit[:circuit_depth])
        pure_probs = np.abs(res.state_vector()) ** 2

        circuit_with_measure = cirq.Circuit(
            *circuit[:circuit_depth],
            cirq.Moment(cirq.measure(qubits))
        )
        
        start = time.perf_counter()
        noisy_res = noisy_sim.sample(circuit_with_measure, repetitions=shots)
        sampled_inds = noisy_res.values[:, 0]
        end = time.perf_counter()
        elapsed_time_sec = end - start

        amplitudes = pure_probs[sampled_inds]

        linear_xeb = linear_xeb_estimate(amplitudes, dim)
        linear_xeb_std_err = linear_xeb_std_err_estimate(amplitudes, dim)

        res = SampleResult(
            task=task_name,
            size=len(qubits),
            circuit_id=circuit_id,
            noise_model=noise_model,
            cycle_depth=cycle_depth,
            shots=shots,
            linear_xeb=linear_xeb,
            linear_xeb_std_err=linear_xeb_std_err,
            elapsed_time_sec=elapsed_time_sec,
            time_per_shot_sec=elapsed_time_sec / shots,
        )
        results.append(res)
        with open(save_resume_filepath, 'a') as f:
            print(res.csv_line(), file=f, flush=True)

    return results
