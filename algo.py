import cvxopt as cvx
import numpy as np
import picos as pic


def build_laplacian(W):
    ''' Out of the adjacency matrix, builds the laplacian of the graph
        Args:
            - W (ndarray): the adjacency matrix
        Outputs:
            - ndarray: the laplacian of the graph
    '''
    degrees = np.sum(W, axis=1)
    D = np.diag(degrees)  # the degree matrix
    return D - W


def solve_sdp(L, triangle_inequalities=False, solver='cvxopt'):
    ''' Solves the SDP relaxation (stengthened if specified) of the max-cut
    problem associated with the graph given by its laplacian L.
        Args:
            - L (ndarray): the graph laplacian
        Outputs:
            - ndarray : the solution of the SDP
    '''
    n = L.shape[0]
    prob = pic.Problem()
    X = prob.add_variable('X', (n, n), 'symmetric')
    if triangle_inequalities:
        node_triples = [
            (i, j, k) for i in range(n) for j in range(n) for k in range(n)]

    L_param = pic.new_param('L', L)
    # Objective
    prob.set_objective('max', L_param | X)
    # Ones on the diagonal
    prob.add_constraint(pic.tools.diag_vect(X) == 1)
    # X positive semidefinite
    prob.add_constraint(X >> 0)

    if triangle_inequalities:
        # First triangle inequality
        prob.add_list_of_constraints(
            [X[i, j] + X[j, k] + X[k, i] > -1 for (i, j, k) in node_triples],
            ['i', 'j', 'k'],
            'node triples')
        # Second triangle inequality
        prob.add_list_of_constraints(
            [X[i, j] - X[j, k] - X[k, i] > -1 for (i, j, k) in node_triples],
            ['i', 'j', 'k'],
            'node triples')
        # Third triangle inequality
        prob.add_list_of_constraints(
            [-X[i, j] + X[j, k] - X[k, i] > -1 for (i, j, k) in node_triples],
            ['i', 'j', 'k'],
            'node triples')
        # Fourth triangle inequality
        prob.add_list_of_constraints(
            [-X[i, j] - X[j, k] + X[k, i] > -1 for (i, j, k) in node_triples],
            ['i', 'j', 'k'],
            'node triples')
    return prob.solve(solver=solver, verbose=0)


def assignment_solution(X, decomp="cholesky"):
    ''' Checks whether the solution returned by the SDP is integral, and if it
    is, returns the assignment defined by X via a Cholesky factorization or a
    singular value decomposition.
        Args:
            - X (ndarray): the solution of the SDP
        Outputs:
            - list of bool: assignment for each node to a certain cluster if
                solution is integral, False otherwise.
    '''
    if decomp == "cholesky":
        V = np.linalg.cholesky(X)
    elif decomp == "svd":
        U, S, V = np.linalg.svd(X)
        S = np.sqrt(S)
        V = U.dot(np.diag(S))
    else:
        raise ValueError("Decomposition has to be cholesky or svd")
    vectors = set()
    for i in range(V.shape[0]):
        vectors.add(V[:, i])
    if len(vectors) == 2:
        assignment = V == next(iter(Vectors))
        assignment = assignment.astype(int)
        assignment = np.prod(assignment)
        return assignment[0]
    else:
        return False
