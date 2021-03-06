from algo import build_laplacian, solve_sdp, assignment_solution
from data_proc import build_a, build_graph, build_user_to_movies

# building the summary dictionary
summary_dictionary = build_user_to_movies('movielens.tsv')
# unpacking of the dictionary
users_to_movies = summary_dictionary['users_to_movies']
n_users = summary_dictionary['n_users']
k_users = 5
n_movies = summary_dictionary['n_movies']
k_movies = 5
# building the rating matrix
a = build_a(n_users, k_users, n_movies, k_movies, users_to_movies)
# building the adjacency matrix
W = build_graph(k_users, k_movies, a)
# the laplacian of the graph
L = build_laplacian(W)
X = solve_sdp(W)
print(X)
assignment = assignment_solution(X)
print(type(assignment))
