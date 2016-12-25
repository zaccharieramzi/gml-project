from algo import solve_sdp, assignment_solution
from data_proc import build_a, build_graph, build_user_to_movies

# building the summary dictionary
summary_dictionary = build_user_to_movies('movielens.tsv')
# unpacking of the dictionary
users_to_movies = summary_dictionary['users_to_movies']
n_users = summary_dictionary['n_users']
n_movies = summary_dictionary['n_movies']
# building the rating matrix
a = build_a(n_users, n_movies, users_to_movies)
# building the adjacency matrix
w = build_graph(n_users, n_movies, a)

v = solve_sdp(w, d=5)
a = assignment_solution(v)
print(type(a))
