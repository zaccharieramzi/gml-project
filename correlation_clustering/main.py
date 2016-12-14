from data_proc import *

n_users, n_movies, db_user_id_to_user_id, db_movies_id_to_movies_id, \
    users_to_movies = build_user_to_movies('movielens.tsv')
print users_to_movies[0]
