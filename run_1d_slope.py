import sys
import json
import multiprocessing

import numpy as np
sys.path.append("..")

from sim.parallel import run


def main():
    e_paulis = np.linspace(0.5e-3, 1e-2, 2)

    pool = multiprocessing.Pool(7)

    res = []
    def log_result(result):
        res.append(result)

    for e_pauli in e_paulis:
        pool.apply_async(run, (e_pauli,), callback=log_result)
    pool.close()
    pool.join()

    # xeb_ratios = {}
    # for e_pauli, xebs in res:
    #     ratio = xebs[1:] / xebs[:-1]
    #     xeb_ratios[e_pauli] = np.log(ratio)
        
    with open("result/1d_n10_slope_circuit10.json", 'w') as f:
        json.dump({e: xeb.tolist() for e, xeb in res}, f)

if __name__ == "__main__":
    main()