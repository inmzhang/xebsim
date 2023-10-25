import itertools
from typing import List, Iterable, Dict, Optional, Tuple

import cirq
from cirq.experiments import GridInteractionLayer
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

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


COMMON_PATTERNS = {
    "supremacy": "ABCDCDAB",
    "verify": "EFGH",
    "abcd": "ABCD",
}


def phasedxz_gateset(axis_on_equator: bool = False) -> List[cirq.Gate]:
    exponents = np.linspace(0, 7/4, 8)
    if axis_on_equator:
        return [
            cirq.PhasedXZGate(x_exponent=0.5, z_exponent=0, axis_phase_exponent=a)
            for a in exponents
        ]
    return [
        cirq.PhasedXZGate(x_exponent=0.5, z_exponent=z, axis_phase_exponent=a)
        for a, z in itertools.product(exponents, repeat=2)
    ]


SUPREMACY_SINGLE_QUBIT_GATES = (
    cirq.X**0.5,
    cirq.Y**0.5,
    cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5),
)


def gen_random_single_qubit_gate_sequence(
        gateset: Iterable[cirq.Gate],
        n_cycles: int,
        n_qubits: int = 1,
        neighbor_diff: bool = True,
        add_final_single_qubit_layer: bool = True,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
) -> List[List[cirq.Gate]]:
    random_state = cirq.value.parse_random_state(seed)
    gateset = list(gateset)
    selected_gates = []
    prev = [cirq.I] * n_qubits
    n_cycles = n_cycles + 1 if add_final_single_qubit_layer else n_cycles
    for _ in range(n_cycles):
        new = []
        for i in range(n_qubits):
            valid_choices = [g for g in gateset if g != prev[0]] if neighbor_diff else gateset
            new.append(random_state.choice(valid_choices))
        selected_gates.append(new)
        prev = new
    return selected_gates


def finalize_rqc(circuit: cirq.Circuit) -> cirq.Circuit:
    """Finalize the building of rqc with measurements appended."""
    circuit.append(cirq.measure(*circuit.all_qubits(), key='m'))
    return circuit


def linear_xeb_between_probvectors(
        prob1: np.ndarray,
        prob2: np.ndarray,
        dim: int,
) -> float:
    return dim * np.vdot(prob1, prob2).item() - 1


def linear_xeb_estimate(amps: np.ndarray, dim: int) -> float:
    return dim * np.mean(amps) - 1


def linear_xeb_std_err_estimate(amps: np.ndarray, dim: int) -> float:
    return dim * np.sqrt(np.var(amps) / len(amps))


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
