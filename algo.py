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
    print(prob.solve(solver=solver, verbose=0))
    return X.value


def assignment_solution(X):
    ''' Checks whether the solution returned by the SDP is integral, and if it
    is, returns the assignment defined by X.
        Args:
            - X (ndarray): the solution of the SDP
        Outputs:
            - list of bool: assignment for each node to a certain cluster if
                solution is integral, False otherwise.
    '''
    scalar_products = sorted(
        list(np.unique([int(round(x)) for x in np.unique(X)])))
    if scalar_products == [-1, 1]:
        return X[0, :] > 0
    else:
        return False
