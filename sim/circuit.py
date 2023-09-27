from typing import Iterable, List, Sequence, Callable, Tuple
import itertools

import cirq
import numpy as np
from cirq.experiments import random_rotations_between_grid_interaction_layers_circuit
from sim.utils import PATTERN_MAP

# Random single qubit gates
exponents = np.linspace(0, 7/4, 8)
SINGLE_QUBIT_GATES = tuple(
    cirq.PhasedXZGate(x_exponent=0.5, z_exponent=z, axis_phase_exponent=a)
    for a, z in itertools.product(exponents, repeat=2)
)
# SINGLE_QUBIT_GATES = tuple(
#     cirq.PhasedXZGate(x_exponent=0.5, z_exponent=0, axis_phase_exponent=a)
#     for a in exponents
# )


SUPREMACY = "ABCDCDAB"
VERIFY = "EFGH"
ABCD = "ABCD"

def _fsim_factory(a: cirq.GridQubit, b: cirq.GridQubit, _) -> cirq.OP_TREE:
    """Default two-qubit gate factory."""
    return cirq.FSimGate(theta=np.pi/2, phi=np.pi/6)(a, b)


def _sqrt_iswap_factory(a: cirq.GridQubit, b: cirq.GridQubit, _) -> cirq.OP_TREE:
    """Default two-qubit gate factory."""
    return cirq.SQRT_ISWAP(a, b)


TWO_QUBIT_GATE_FACTORIES_T = Callable[[cirq.GridQubit, cirq.GridQubit, np.random.RandomState], cirq.OP_TREE]


def gen_single_qubit_xeb_random_circuit(
        qubit: cirq.LineQubit,
        *,
        depth: int,
        single_qubit_gates: Sequence['cirq.Gate'] = SINGLE_QUBIT_GATES,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
) -> cirq.Circuit:
    """Generate XEB circuits for a single qubit."""
    circuit = cirq.Circuit()
    random_state = cirq.value.parse_random_state(seed)
    prev_gate = cirq.I
    for _ in range(depth + 1):
        new_gate = random_state.choice([g for g in single_qubit_gates if g != prev_gate])
        circuit.append(new_gate(qubit))
        prev_gate = new_gate
    return circuit


def gen_1d_chain_xeb_random_circuit(
        qubits: Sequence[cirq.LineQubit],
        *,
        depth: int,
        single_qubit_gates: Sequence['cirq.Gate'] = SINGLE_QUBIT_GATES,
        two_qubit_op_factory: TWO_QUBIT_GATE_FACTORIES_T = _sqrt_iswap_factory,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
) -> cirq.Circuit:
    """Generate XEB circuits for a 1D chain of qubits."""
    assert depth % 2 == 0, "Depth must be even."
    assert len(qubits) >= 6, "Need at least 6 qubits."
    circuit = cirq.Circuit()
    random_state = cirq.value.parse_random_state(seed)
    prev_gates = [cirq.I] * len(qubits)
    for d in range(depth // 2):
        new_gates = [
            random_state.choice([g for g in single_qubit_gates if g != prev_gates[i]])
            for i in range(len(qubits))
        ]
        circuit.append(cirq.Moment(new_gate(q) for q, new_gate in zip(qubits, new_gates)))
        prev_gates = new_gates
        for a, b in zip(qubits[d % 2::2], qubits[d % 2 + 1::2]):
            circuit.append(two_qubit_op_factory(a, b, random_state))
    new_gates = [
        random_state.choice([g for g in single_qubit_gates if g != prev_gates[i]])
        for i in range(len(qubits))
    ]
    circuit.append(cirq.Moment(new_gate(q) for q, new_gate in zip(qubits, new_gates)))
    return circuit


def gen_2d_grid_xeb_random_circuits(
        qubits: Iterable[cirq.GridQubit],
        *,
        depth: int,
        num_circuits: int,
        two_qubit_op_factory: TWO_QUBIT_GATE_FACTORIES_T = _fsim_factory,
        pattern: str = SUPREMACY,
        single_qubit_gates: Sequence['cirq.Gate'] = SINGLE_QUBIT_GATES,
        add_final_single_qubit_layer: bool = True,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
) -> Tuple[List[cirq.Circuit], str]:
    """Generate supremacy-style random circuits for sampling."""
    layer_pattern = [PATTERN_MAP[p] for p in pattern]
    random_state = cirq.value.parse_random_state(seed)
    circuits = [
        random_rotations_between_grid_interaction_layers_circuit(
            qubits,
            depth=depth,
            two_qubit_op_factory=two_qubit_op_factory,
            pattern=layer_pattern,
            single_qubit_gates=single_qubit_gates,
            add_final_single_qubit_layer=add_final_single_qubit_layer,
            seed=random_state
        )
        for _ in range(num_circuits)
    ]
    return circuits, "".join(itertools.islice(itertools.cycle(list(pattern)), num_circuits))

    
def gen_1d_chain_weak_link_random_circuit(
    qubits: Sequence[cirq.LineQubit],
    *,
    depth: int,
    link_frequency: int,
    single_qubit_gates: Sequence['cirq.Gate'] = SINGLE_QUBIT_GATES,
    two_qubit_op_factory: TWO_QUBIT_GATE_FACTORIES_T = _sqrt_iswap_factory,
    seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
) -> cirq.Circuit:
    """Generate XEB circuits for a 1D chain of qubits."""
    assert depth % 2 == 0, "Depth must be even."
    assert link_frequency % 2 == 0, "Link frequency must be even."
    assert len(qubits) >= 6, "Need at least 6 qubits."
    assert len(qubits) % 2 == 0, "The number of the qubits should be even."

    link1, link2 = qubits[len(qubits) // 2 - 1], qubits[len(qubits) // 2]
    
    circuit = cirq.Circuit()
    random_state = cirq.value.parse_random_state(seed)
    prev_gates = [cirq.I] * len(qubits)
    for d in range(depth // 2):
        new_gates = [
            random_state.choice([g for g in single_qubit_gates if g != prev_gates[i]])
            for i in range(len(qubits))
        ]
        circuit.append(cirq.Moment(new_gate(q) for q, new_gate in zip(qubits, new_gates)))
        prev_gates = new_gates
        if (d + 1) % link_frequency == 0:
            circuit.append(two_qubit_op_factory(link1, link2, random_state))
        for a, b in zip(qubits[d % 2::2], qubits[d % 2 + 1::2]):
            if a == link1:
                continue
            circuit.append(two_qubit_op_factory(a, b, random_state))
    new_gates = [
        random_state.choice([g for g in single_qubit_gates if g != prev_gates[i]])
        for i in range(len(qubits))
    ]
    circuit.append(cirq.Moment(new_gate(q) for q, new_gate in zip(qubits, new_gates)))
    return circuit