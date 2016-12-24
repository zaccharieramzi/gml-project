import cvxopt as cvx
import numpy as np
import picos as pic


def solve_sdp(w, d=None, solver='cvxopt'):
    ''' Solves the SDP relaxation of the max-cut problem associated with the
    graph w.
        Args:
            - w (np.matrix): the adjacency matrix
        Outputs:
            - np.matrix : the solution of the SDP
    '''
    n = w.shape[0]
    if not(d):
        d = n
    prob = pic.Problem()
    v = prob.add_variable('v', (n, d))
    node_couples = [(i, j) for i in range(n) for j in range(n)]
    node_triples = [
        (i, j, k) for i in range(n) for j in range(n) for k in range(n)]
    prob.set_objective('max', pic.sum(
        [w[e]*pic.norm(v[:, e[0]]-v[:, e[1]], 2)**2 for e in node_couples],
        [('e', 2)],
        'node couples'
        )
    )
    # All indicators on unit sphere
    prob.add_list_of_constraints(
        [pic.norm(v[:, i])**2 == 1 for i in range(n)],
        'i',
        'nodes')

    def sqdist(u, v):
        return pic.norm(u - v)**2

    # First triangular inequality
    prob.add_list_of_constraints(
        [sqdist(v[:, t[0]], v[:, t[1]]) + sqdist(v[:, t[1]], v[:, t[2]]) >=
         sqdist(v[:, t[0]], v[:, t[2]]) for t in node_triples],
        't',
        'node triples')
    # Second triangular inequality
    prob.add_list_of_constraints(
        [sqdist(v[:, t[0]], v[:, t[1]]) + sqdist(v[:, t[1]], -v[:, t[2]]) >=
         sqdist(v[:, t[0]], -v[:, t[2]]) for t in node_triples],
        't',
        'node triples')
    # Third triangular inequality
    prob.add_list_of_constraints(
        [sqdist(v[:, t[0]], -v[:, t[1]]) + sqdist(v[:, t[1]], -v[:, t[2]]) >=
         sqdist(v[:, t[0]], v[:, t[2]]) for t in node_triples],
        't',
        'node triples')
    return prob.solve(solver=solver, verbose=0)


def assignment_solution(v):
    ''' Checks whether the solution returned by the SDP is integral, and if it
    is, returns the assignment defined by v.
        Args:
            - v (n.matrix): the solution of the SDP
        Outputs:
            - list of int: assignment for each node to a certain cluster if
                solution is integral, False otherwise.
    '''
    vectors = set()
    for i in range(v.shape[0]):
        vectors.add(v[:, i])
    if len(vectors) == 2:
        assignment = v.T == next(iter(vectors))
        assignment = assignment.astype(int)
        assignment = np.prod(assignment)
        return assignment
    else:
        return False
