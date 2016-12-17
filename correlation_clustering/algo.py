import cvxopt as cvx
import numpy as np
import picos as pic


def build_laplacian(w):
    ''' Out of the adjacency matrix, builds the laplacian of the graph
        Args:
            - w (np.matrix): the adjacency matrix
        Outputs:
            - np.matrix: the laplacian of the graph
    '''
    degrees = np.sum(W, axis=1)
    d = np.diag(degrees)  # the degree matrix
    return d - w


def solve_sdp(l):
    ''' Solves the SDP relaxation of the max-cut problem associated with the
    laplacian l.
        Args:
            - l (np.matrix) : the laplacian of the graph
        Outputs:
            - np.matrix : the solution of the SDP
    '''
    n = l.shape[0]
    prob = pic.Problem()
    x = prob.add_variable('x', (n, n), vtype='symmetric')
    prob.set_objective('min', x | l)
    prob.add_constraint(x >> 0)
    prob.add_list_of_constraints(
        [x[i, i] == 1 for i in range(n)],
        'i',
        '[n]')
    return prob.solve(solver='cvxopt', verbose=0)


def assignment_solution(x):
    ''' Checks whether the solution returned by the SDP is integral, and if it
    is, returns the assignment defined by x.
        Args:
            - x (n.matrix): the solution of the SDP
        Outputs:
            - list of int: assignment for each node to a certain cluster if
                solution is integral, False otherwise.
    '''
    v = np.linalg.cholesky(x)
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
