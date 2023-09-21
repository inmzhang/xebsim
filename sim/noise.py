from typing import Sequence

import numpy as np
import cirq


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


class CoherentNoiseModel(cirq.NoiseModel):
    def __init__(self, noise_amplitude_mrad: float):
        # TODO: check the definition
        self.noise_amplitude = noise_amplitude_mrad / 1e3

    def noisy_moment(
            self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']
    ) -> 'cirq.OP_TREE':
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
            return [moment, coherent_noise_moment]
        return moment
