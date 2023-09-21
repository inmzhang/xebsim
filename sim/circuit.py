from typing import Sequence, Callable
import itertools

import cirq
import numpy as np

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
    assert depth % 2 == 1, "Depth must be odd."
    assert len(qubits) >= 6, "Need at least 6 qubits."
    circuit = cirq.Circuit()
    random_state = cirq.value.parse_random_state(seed)
    prev_gates = [cirq.I] * len(qubits)
    for d in range((depth - 1) // 2):
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
