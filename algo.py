''' The solve_sdp function was inspired by the picos documentation example.
[1]: http://www.cs.huji.ac.il/~yrabani/Papers/CalinescuKR-JCSS-revised.pdf
'''
from itertools import combinations
import math

import networkx as nx
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


def assignment_solution_sdp(X, threshold=0.00001):
    ''' Checks whether the solution returned by the SDP is integral, and if it
    is, returns the assignment defined by X.
        Args:
            - X (ndarray): the solution of the SDP
        Outputs:
            - list of bool: assignment for each node to a certain cluster if
                solution is integral, False otherwise.
    '''
    rounded_X = np.round(X)
    gap = np.absolute(rounded_X - X)
    n = X.shape[0]
    # we create a variable checking whether the rounding we do is correct
    rounding = all([gap[i, j] < threshold for i in range(n) for j in range(n)])
    scalar_products = sorted(
        list(np.unique([int(round(x)) for x in np.unique(X)])))
    if scalar_products == [-1, 1] and rounding:
        return X[0, :] > 0
    else:
        return False


def solve_multicut(W, T, solver='cvxopt'):
    '''Solves the LP relaxation (LP2 of [1]) of the minimum multiway cut
    problem associated to the graph given by its adjacency matrix W.
        Args:
            - W (ndarray): the adjacency matrix
            - T (list): the list of terminal nodes
        Output:
            - ndarray: the solution of the LP
    '''
    G = nx.from_numpy_matrix(W)
    n = W.shape[0]
    K = len(T)

    def s2i(row_col, array_shape=(n, n)):
        '''Equivalent of Matlab sub2ind. From rows and cols (n, n) indices to
        linear index.
        '''
        row, col = row_col
        ind = row*array_shape[1] + col
        return int(ind)

    node_triples = [
        (i, j, k) for i in range(n) for j in range(n) for k in range(n)]
    node_couples = [(i, j) for i in range(n) for j in range(n)]
    terminal_couples = [(i, j) for i in T for j in T if i != j]

    prob = pic.Problem()
    # Picos params and variables
    d = prob.add_variable('d', n**2)
    d_prime = {}
    for t in T:
        d_prime[t] = prob.add_variable('d_prime[{0}]'.format(t), n**2)
    # Objective
    prob.set_objective('min',
                       pic.sum([W[e]*d[s2i(e)]
                                for e in G.edges()],
                               [('e', 2)], 'edges'))
    # (V, d) semimetric (1)
    prob.add_list_of_constraints(
        [d[s2i((u, u))] == 0 for u in G.nodes()],
        'u',
        'nodes')
    # prob.add_list_of_constraints(
    #     [d[s2i(c)] == d[s2i((c[1], c[0]))] for c in node_couples],
    #     [('c', 2)],
    #     'node couples') Seems to be problematic may have to do with some kind
    # of redundance : http://stackoverflow.com/questions/16978763
    # prob.add_constraint(d >= 0) This constraint is redundant when we add the
    # complementary constraints due to the addition of d_prime
    # (2) terminals are far apart
    prob.add_list_of_constraints(
        [d[s2i(c)] == 1 for c in terminal_couples],
        [('c', 2)],
        'terminal couples')
    # (3) distance should be inferior to 1
    prob.add_constraint(d <= 1)
    # (4)
    prob.add_list_of_constraints(
        [pic.sum([d[s2i((u, t))] for t in T], 't', 'terminals') == K-1
         for u in G.nodes()],
        'u',
        'nodes')
    # (5') with d_prime
    prob.add_list_of_constraints(
        [d[s2i(c)] >= pic.sum([d_prime[t][s2i(c)] for t in T],
                              't',
                              'terminals') for c in node_couples],
        [('c', 2)],
        'node couples')
    # (6') constraints on d_prime
    prob.add_list_of_constraints(
        [d_prime[t] >= 0 for t in T],
        't',
        'terminals')
    prob.add_list_of_constraints(
        [d_prime[t][s2i((u, v))] >=
         d[s2i((u, t))] - d[s2i((v, t))]
         for (u, v) in node_couples for t in T],
        ['u', 'v', 't'],
        'node couples x terminals')
    print(prob.solve(solver=solver, verbose=0))
    return d.value


def assignment_solution_lp(d, T, threshold=0.00001):
    ''' Checks whether the solution returned by the LP is integral, and if it
    is, returns the assignment defined by d.
        Args:
            - d (ndarray): the solution of the SDP
            - T (list): the terminal nodes
        Outputs:
            - if integral : dict: - key: index of the cluster
                                  - value: list of nodes in the cluster
            - False otherwise
    '''
    rounded_d = np.round(d)
    gap = np.absolute(rounded_d - d)
    n = math.sqrt(len(d))
    rounding = all([gap[i] < threshold for idx in range(n**2)])
    distances = sorted(list(np.unique(rounded_d)))
    assignment = dict()
    if rounding and distances == [0, 1]:
        # Then we have an integral solution
        distance_per_node = np.reshape(rounded_d, (n, n))
        for t in T:
            assignment[t] = [u for u in range(n)
                             if distance_per_node[u, t] == 0]
    else:
        # The solution is not integral
        return False
