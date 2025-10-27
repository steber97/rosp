import math
import numpy as np
import itertools

from typing import List

from pysat.solvers import Glucose4, Maplesat
from pysat.formula import CNF

from utils import psd, diagonally_dominant

def to_glucose(clauses) -> Maplesat:
        cnf = Maplesat()
        for clause in clauses:
            cnf.add_clause(clause)
        return cnf


def create_random_formula(n, m) -> CNF:
    clauses: List
    while True:
        clauses = []
        uniquevars = set()
        for c in range(m):
            vars = []
            while True:
                vars = [np.random.randint(1, n+1) for i in range(3)]
                if len(np.unique(vars)) == 3:
                    break
            for v in vars:
                uniquevars.add(v)
            vars = [int((-0.5 + np.random.randint(0, 2)) * 2) * x for x in vars]
            clauses.append(vars)
        if len(uniquevars) == n:
            break
    cnf = CNF()
    for c in clauses:
        cnf.append(c)
    return cnf

def R(cnf: CNF, n, m) -> np.array:
    f = max(2*n + 7, m)
    M = np.zeros((n + f + 1, n + f + 1))
    for i in range(0, n):
        M[i, i] = 2 * f + n + 3
    for i in range(n, n+f+1):
        M[i, i] = n + 2
    for i in range(n, n+f+1):
        for j in range(n, n+f+1):
            if i != j:
                M[i, j] = 1
    for i in range(n, n+f+1):
        clause_i = i - n if i - n < len(cnf.clauses) else len(cnf.clauses) - 1
        clause = cnf.clauses[clause_i]
        for var in clause:
            sign = (1 if var > 0 else -1)
            var_index = (var * sign) - 1
            M[i, var_index] = sign
    
    for i in range(n):
        for j in range(n, n+f+1):
            M[i, j] = M[j,i]
    
    return M

if __name__ == "__main__":
    n = 15
    feige_delta = 4.2
    m = math.ceil(n * feige_delta)
    # m = 3

    phi = create_random_formula(n, m)
    
    solver = Maplesat()
    for c in phi:
        solver.add_clause(c)
    sat = solver.solve()

    M_phi = R(phi, n, m)

    print("M_phi shape:", M_phi.shape)
    print("n = {}, m = {}".format(n, m))
    print("SAT?:", sat)
    print("PSD?:", psd(M_phi))
    print("DD?:", diagonally_dominant(M_phi))
    
    if sat:
        certificate = solver.get_model() 
        v = np.array([1 if x > 0 else -1 for x in certificate] + [1 for i in range(len(M_phi) - n)])
        print("ROSP?:", diagonally_dominant(M_phi - np.outer(v, v)))
    else:
        vectors = itertools.product([-1,1], repeat=n)
        for vect in vectors:
            v = np.array(list(vect) + [1 for i in range(len(M_phi) - n)])
            assert not diagonally_dominant(M_phi - np.outer(v, v))
        print("ROSP? False")
        
