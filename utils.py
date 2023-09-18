from typing import Iterable, Optional, Tuple

import cirq
from cirq.experiments import GridInteractionLayer
import networkx as nx
import matplotlib.pyplot as plt


def plot_grid_interaction_layer(
        layer: GridInteractionLayer,
        qubits: Iterable[cirq.GridQubit],
        grid_width: Optional[int] = None,
        grid_height: Optional[int] = None,
        figsize: Tuple[int, int] = (6, 6)
):
    """Plot the interaction layer on a grid of qubits."""
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
    plt.figure(figsize=figsize)
    nx.draw_networkx(
        grid,
        pos={n: n for n in grid.nodes},
        node_color=['blue' if q in shift_qubits else 'black' for q in grid.nodes],
        edge_color=[
            'red' if (cirq.GridQubit(e[0][1], e[0][0]), cirq.GridQubit(e[1][1], e[1][0])) in layer else 'black'
            for e in grid.edges
        ],
        node_size=200,
    )
    plt.gca().invert_yaxis()

