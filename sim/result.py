"""Store the sampling results and breakpoints."""
from typing import Optional, Union, List
import dataclasses
import pathlib

import cirq


@dataclasses.dataclass(frozen=True)
class SampleResult:
    task: str
    size: int
    circuit_id: str
    noise_model: cirq.NoiseModel
    cycle_depth: int
    shots: Optional[int]
    linear_xeb: float
    linear_xeb_std_err: Optional[float] = None
    elapsed_time_sec: Optional[float] = None
    time_per_shot_sec: Optional[float] = None

    def csv_line(self) -> str:
        return (f'{self.task},'
                f'{self.size},'
                f'{self.circuit_id},'
                f'{self.noise_model!r},'
                f'{self.cycle_depth},'
                f'{self.linear_xeb},'
                f'{self.shots},'
                f'{self.linear_xeb_std_err},'
                f'{self.elapsed_time_sec},'
                f'{self.time_per_shot_sec}')


def read_results_from_file(file: Union[str, pathlib.Path]) -> List[SampleResult]:
    import csv
    import cirq
    results = []
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [e.strip() for e in reader.fieldnames]
        for row in reader:
            results.append(SampleResult(
                task=row['task'],
                size=int(row['size']),
                circuit_id=row['circuit_id'],
                noise_model=eval(row['noise_model'], {'cirq': cirq}),
                cycle_depth=int(row['cycle_depth']),
                linear_xeb=float(row['linear_xeb']),
                shots=_load_nullable_value(row['shots'], int),
                linear_xeb_std_err=_load_nullable_value(row['linear_xeb_std_err'], float),
                elapsed_time_sec=_load_nullable_value(row['elapsed_time_sec'], float),
                time_per_shot_sec=_load_nullable_value(row['time_per_shot_sec'], float),
            ))
    return results
            
            
def _load_nullable_value(value, type_factory=None):
    if value == 'None':
        return None
    if type_factory:
        return type_factory(value)
    return value
    