"""Comparison half of the cross-validation harness — see VALIDATION.md.
Run after `octave validate.m` has produced the mat_*.csv files."""
import numpy as np

def python_weights(T, tau):
    lamb = np.log(2) / tau
    w = np.exp(lamb * np.arange(T))
    return w / w.sum()

def python_ens(p):
    return float(np.exp(-np.sum(p * np.log(p))))

def python_ens_gen(p, gamma):
    return float(np.sum(p**gamma) ** (-1/(gamma-1)))

tests = []
for fname, T, tau in [("mat_p_T100_tau10.csv",100,10), ("mat_p_T1060_tau118.csv",1060,118)]:
    tests.append((f"weights T={T} tau={tau}", np.max(np.abs(np.loadtxt(fname,delimiter=",") - python_weights(T,tau)))))
e = np.loadtxt("mat_ens.csv", delimiter=",")
p1, p2 = python_weights(100,10), python_weights(1060,118)
for lab, m, py in zip(["ENS100","ENS1060","ENSg100","ENSg1060"], e,
                      [python_ens(p1),python_ens(p2),python_ens_gen(p1,2),python_ens_gen(p2,2)]):
    tests.append((lab, abs(m-py)))
er = np.loadtxt("mat_ens_ragged.csv", delimiter=",")
p3 = np.array([0.4,0.3,0.15,0.1,0.05])
tests += [("ENS ragged", abs(er[0]-python_ens(p3))), ("ENSg ragged", abs(er[1]-python_ens_gen(p3,2)))]
ok = all(d < 1e-12 for _, d in tests)
for lab, d in tests: print(f"{lab:<24}{d:.3e}  {'PASS' if d<1e-12 else 'FAIL'}")
print("ALL PASS" if ok else "FAILURES")
