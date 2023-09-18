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


def gen_random_circuits(
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
