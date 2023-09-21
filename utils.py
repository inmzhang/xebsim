from typing import Iterable, Optional, Tuple, Callable, Sequence, List, Dict
import itertools

import cirq
import pandas as pd
from cirq.experiments import (
    GridInteractionLayer,
    random_rotations_between_grid_interaction_layers_circuit
)
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, optimize


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


def exponential_decay(cycle_depths: np.ndarray, a: float, layer_fid: float) -> np.ndarray:
    """An exponential decay for fitting.

    This computes `a * layer_fid**cycle_depths`

    Args:
        cycle_depths: The various depths at which fidelity was estimated. This is the independent
            variable in the exponential function.
        a: A scale parameter in the exponential function.
        layer_fid: The base of the exponent in the exponential function.
    """
    return a * layer_fid**cycle_depths


def _fit_exponential_decay(
        cycle_depths: np.ndarray,
        fidelities: np.ndarray
) -> Tuple[float, float, float, float]:
    """Fit an exponential model fidelity = a * layer_fid**x using nonlinear least squares.

    This uses `exponential_decay` as the function to fit with parameters `a` and `layer_fid`.

    Args:
        cycle_depths: The various depths at which fidelity was estimated. Each element is `x`
            in the fit expression.
        fidelities: The estimated fidelities for each cycle depth. Each element is `fidelity`
            in the fit expression.

    Returns:
        a: The first fit parameter that scales the exponential function, perhaps accounting for
            state prep and measurement (SPAM) error.
        layer_fid: The second fit parameters which serves as the base of the exponential.
        a_std: The standard deviation of the `a` parameter estimate.
        layer_fid_std: The standard deviation of the `layer_fid` parameter estimate.
    """
    cycle_depths = np.asarray(cycle_depths)
    fidelities = np.asarray(fidelities)

    # Get initial guess by linear least squares with logarithm of model.
    # This only works for positive fidelities. We use numpy fancy indexing
    # with `positives` (an ndarray of bools).
    positives = fidelities > 0
    if np.sum(positives) <= 1:
        # The sum of the boolean array is the number of `True` entries.
        # For one or fewer positive values, we cannot perform the linear fit.
        return 0, 0, np.inf, np.inf
    cycle_depths_pos = cycle_depths[positives]
    log_fidelities = np.log(fidelities[positives])

    slope, intercept, _, _, _ = stats.linregress(cycle_depths_pos, log_fidelities)
    layer_fid_0 = np.clip(np.exp(slope), 0, 1)
    a_0 = np.clip(np.exp(intercept), 0, 1)

    try:
        (a, layer_fid), pcov = optimize.curve_fit(
            exponential_decay,
            cycle_depths,
            fidelities,
            p0=(a_0, layer_fid_0),
            bounds=((0, 0), (1, 1)),
        )
    except ValueError:  # coverage: ignore
        # coverage: ignore
        return 0, 0, np.inf, np.inf

    a_std, layer_fid_std = np.sqrt(np.diag(pcov))
    return a, layer_fid, a_std, layer_fid_std


def fit_exponential_decays(fidelities_df: pd.DataFrame) -> pd.DataFrame:
    """Fit exponential decay curves to a fidelities DataFrame. """

    def _per_noise(f1):
        a, layer_fid, a_std, layer_fid_std = _fit_exponential_decay(
            f1['cycle_depth'], f1['fidelity']
        )
        record = {
            'a': a,
            'layer_fid': layer_fid,
            'cycle_depths': f1['cycle_depth'].values,
            'fidelities': f1['fidelity'].values,
            'a_std': a_std,
            'layer_fid_std': layer_fid_std,
        }
        return pd.Series(record)

    return fidelities_df.groupby('qubit').apply(_per_noise)


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
