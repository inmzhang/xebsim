# xebsim
Simulation of several XEB related experiments

NOTE: `qsim` is not used in the simulation because noisy quantum 
trajectory simulation is extremely slow for the statevector-type
simulator like `qsim`. Instead, we use density-matrix-type simulator
like the built-in one in `cirq`.