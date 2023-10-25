from typing import Iterable, List, Sequence, Tuple
import itertools

import cirq
import numpy as np
from cirq.experiments import random_rotations_between_grid_interaction_layers_circuit
from xebsim import utils


SUPREMACY_FSIM = cirq.FSimGate(theta=np.pi/2, phi=np.pi/6)


def gen_1d_chain_brickwork_rqc(
        qubits: Sequence[cirq.LineQubit],
        *,
        depth: int,
        single_qubit_gates: Sequence['cirq.Gate'],
        two_qubit_gate: cirq.Gate,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
        neighbor_diff: bool = True,
        add_final_single_qubit_layer: bool = True,
) -> cirq.Circuit:
    """Generate XEB circuits for a 1D chain of qubits."""
    assert depth % 2 == 0, "Depth must be even."
    assert len(qubits) >= 6, "Need at least 6 qubits."

    random_gate_sequence = utils.gen_random_single_qubit_gate_sequence(
        single_qubit_gates,
        depth,
        n_qubits=len(qubits),
        seed=seed,
        neighbor_diff=neighbor_diff,
        add_final_single_qubit_layer=add_final_single_qubit_layer
    )
    circuit = cirq.Circuit()
    for i, sequence in enumerate(random_gate_sequence):
        circuit.append(cirq.Moment(g(q) for g, q in zip(sequence, qubits)))
        if add_final_single_qubit_layer and i == len(random_gate_sequence) - 1:
            continue
        for a, b in zip(qubits[i % 2::2], qubits[i % 2 + 1::2]):
            circuit.append(two_qubit_gate(a, b))
    return circuit


def gen_2d_grid_rqc(
        qubits: Iterable[cirq.GridQubit],
        *,
        depth: int,
        num_circuits: int,
        single_qubit_gates: Sequence['cirq.Gate'],
        two_qubit_gate: cirq.Gate,
        pattern: str = utils.COMMON_PATTERNS["supremacy"],
        add_final_single_qubit_layer: bool = True,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
) -> Tuple[List[cirq.Circuit], str]:
    """Generate supremacy-style random circuits for sampling."""
    layer_pattern = [utils.PATTERN_MAP[p] for p in pattern]
    random_state = cirq.value.parse_random_state(seed)
    circuits = [
        random_rotations_between_grid_interaction_layers_circuit(
            qubits,
            depth=depth,
            two_qubit_op_factory=lambda a, b, _: two_qubit_gate(a, b),
            pattern=layer_pattern,
            single_qubit_gates=single_qubit_gates,
            add_final_single_qubit_layer=add_final_single_qubit_layer,
            seed=random_state
        )
        for _ in range(num_circuits)
    ]
    return circuits, "".join(itertools.islice(itertools.cycle(list(pattern)), num_circuits))


def gen_1d_chain_weak_link_rqc(
        qubits: Sequence[cirq.LineQubit],
        *,
        depth: int,
        link_frequency: int,
        single_qubit_gates: Sequence['cirq.Gate'],
        two_qubit_gate: cirq.Gate,
        neighbor_diff: bool = True,
        add_final_single_qubit_layer: bool = True,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
) -> cirq.Circuit:
    """Generate XEB circuits for a 1D chain of qubits."""
    assert depth % 2 == 0, "Depth must be even."
    assert link_frequency % 2 == 0, "Link frequency must be even."
    assert len(qubits) >= 6, "Need at least 6 qubits."
    assert len(qubits) % 2 == 0, "The number of the qubits should be even."

    link1, link2 = qubits[len(qubits) // 2 - 1], qubits[len(qubits) // 2]

    random_gate_sequence = utils.gen_random_single_qubit_gate_sequence(
        single_qubit_gates,
        depth,
        n_qubits=len(qubits),
        seed=seed,
        neighbor_diff=neighbor_diff,
        add_final_single_qubit_layer=add_final_single_qubit_layer
    )
    circuit = cirq.Circuit()
    for i, sequence in enumerate(random_gate_sequence):
        circuit.append(cirq.Moment(g(q) for g, q in zip(sequence, qubits)))
        if add_final_single_qubit_layer and i == len(random_gate_sequence) - 1:
            continue
        if (i + 1) % link_frequency == 0:
            circuit.append(two_qubit_gate(link1, link2))
        for a, b in zip(qubits[i % 2::2], qubits[i % 2 + 1::2]):
            if a == link1:
                continue
            circuit.append(two_qubit_gate(a, b))
    return circuit
