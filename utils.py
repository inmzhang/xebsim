from typing import Iterable, Optional, Tuple, Callable, Sequence, List, Dict
import itertools

import cirq
from cirq.experiments import (
    GridInteractionLayer,
    random_rotations_between_grid_interaction_layers_circuit
)
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def _default_fsim_factory(a: cirq.GridQubit, b: cirq.GridQubit, _) -> cirq.OP_TREE:
    """Default two-qubit gate factory."""
    return cirq.FSimGate(theta=np.pi/2, phi=np.pi/6)(a, b)


TWO_QUBIT_GATE_FACTORIES_T = Callable[[cirq.GridQubit, cirq.GridQubit, np.random.RandomState], cirq.OP_TREE]


DEFAULT_SINGLE_QUBIT_GATES = (
    cirq.X**0.5,
    cirq.Y**0.5,
    cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5),
)


PATTERN_MAP: Dict[str, GridInteractionLayer] = {
    "A": GridInteractionLayer(col_offset=0, vertical=True, stagger=True),
    "B": GridInteractionLayer(col_offset=1, vertical=True, stagger=True),
    "C": GridInteractionLayer(col_offset=1, vertical=False, stagger=True),
    "D": GridInteractionLayer(col_offset=0, vertical=False, stagger=True),
    "E": GridInteractionLayer(col_offset=0, vertical=False, stagger=False),
    "F": GridInteractionLayer(col_offset=1, vertical=False, stagger=False),
    "G": GridInteractionLayer(col_offset=0, vertical=True, stagger=False),
    "H": GridInteractionLayer(col_offset=1, vertical=True, stagger=False),
}


SUPREMACY = "ABCDCDAB"
VERIFY = "EFGH"
ABCD = "ABCD"


def linear_xeb_between_statevector(
        statevector1: np.ndarray,
        statevector2: np.ndarray,
) -> float:
    """Calculate the XEB fidelity between two statevectors."""
    probs1 = np.abs(statevector1) ** 2
    probs2 = np.abs(statevector2) ** 2
    return len(statevector1) * (np.vdot(probs1, probs2)) - 1


def simulate_statevector(
        circuit: cirq.Circuit,
        cycles: Iterable[int],
        sampler: Optional[cirq.Sampler] = None,
) -> List[np.ndarray]:
    """Simulate the statevector of a circuit at different depths."""
    if 2 * max(cycles) + 1 > len(circuit):
        raise ValueError("Circuit is too short to simulate to the desired depth.")
    sampler = sampler or cirq.Simulator()
    return [sampler.simulate(circuit[:2 * cycle + 1]).final_state_vector for cycle in cycles]


def inject_noise(
        circuit: cirq.Circuit,
        noise_amplitude: float,
) -> cirq.Circuit:
    """Inject noise into a circuit."""
    depth = (len(circuit) - 1) // 2
    qubits = list(circuit.all_qubits())
    n_qubits = len(qubits)

    noisy_moments = []
    for d in range(depth):
        noisy_moments.append(circuit[2 * d])
        random_a = np.random.normal(loc=-1, scale=1, size=n_qubits)
        random_x = np.random.normal(loc=0, scale=noise_amplitude, size=n_qubits)
        random_z = np.random.normal(loc=0, scale=noise_amplitude, size=n_qubits)
        noise_moment = cirq.Moment(
            cirq.PhasedXZGate(x_exponent=random_x[i], z_exponent=random_z[i], axis_phase_exponent=random_a[i])(qubits[i])
            for i in range(n_qubits)
        )
        noisy_moments.append(noise_moment)
        noisy_moments.append(circuit[2 * d + 1])
    noisy_moments.append(circuit[-1])
    return cirq.Circuit(noisy_moments)


def gen_random_circuits_without_measurements(
        qubits: Iterable[cirq.GridQubit],
        *,
        depth: int,
        num_circuits: int,
        two_qubit_op_factory: TWO_QUBIT_GATE_FACTORIES_T = _default_fsim_factory,
        pattern: str = SUPREMACY,
        single_qubit_gates: Sequence['cirq.Gate'] = DEFAULT_SINGLE_QUBIT_GATES,
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


def plot_grid_interaction_layers(
        layers: str,
        qubits: Iterable[cirq.GridQubit],
        grid_width: Optional[int] = None,
        grid_height: Optional[int] = None,
):
    """Plot the interaction layer on a grid of qubits."""
    n_layers = len(layers)
    num_rows = int(np.sqrt(n_layers))
    num_cols = int(np.ceil(n_layers / num_rows))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    axes = axes.flat

    min_x = min(q.col for q in qubits)
    max_x = max(q.col for q in qubits)
    min_y = min(q.row for q in qubits)
    max_y = max(q.row for q in qubits)
    if grid_width is None:
        grid_width = max_x - min_x + 1
    if grid_height is None:
        grid_height = max_y - min_y + 1
    grid = nx.grid_2d_graph(grid_width, grid_height)
    shift_qubits = [(q.col - min_x, q.row - min_y) for q in qubits]
    for layer, ax in zip(layers, axes):
        nx.draw_networkx(
            grid,
            pos={n: n for n in grid.nodes},
            node_color=['blue' if q in shift_qubits else 'black' for q in grid.nodes],
            edge_color=[
                'red' if _edge_in_layer(e, PATTERN_MAP[layer]) else 'black'
                for e in grid.edges
            ],
            font_size=10,
            with_labels=False,
            node_size=100,
            width=[3 if _edge_in_layer(e, PATTERN_MAP[layer]) else 1 for e in grid.edges],
            ax=ax,
        )
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_title(str(layer))


def _edge_in_layer(edge: Tuple[Tuple[int, int], Tuple[int, int]], layer: GridInteractionLayer) -> bool:
    """Check if the coupler is active in the layer."""
    return (cirq.GridQubit(edge[0][1], edge[0][0]), cirq.GridQubit(edge[1][1], edge[1][0])) in layer
