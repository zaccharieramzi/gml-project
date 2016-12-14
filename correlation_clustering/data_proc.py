import numpy as np


def build_graph(tsvfilename):
    f = open(tsvfilename)
    line1 = f.readline().split('\t')

    if is_number(line1[0]):
        lines = np.loadtxt(tsvfilename)
    else:
        lines = np.loadtxt(tsvfilename, skiprows=1)


def is_number(s):
    ''' test if the type of the object is float
        Args:
            - s (object)
        Output:
            - boolean
    '''
    try:
        float(s)
        return True
    except ValueError:
        return False


def build_user_to_movies(tsvfilename):
    ''' faire des commentaires pour Zaccharie // return dictionary
    '''
    # reading the tsv file, parsing it into a ndarray
    f = open(tsvfilename)
    line1 = f.readline().split('\t')

    # checking header
    if is_number(line1[0]):
        lines = np.loadtxt(tsvfilename)
    else:
        lines = np.loadtxt(tsvfilename, skiprows=1)

    # number of unique users in the tsv
    n_users = np.unique(lines[:, 0]).shape[0]

    # number of movies in the tsv
    n_movies = np.unique(lines[:, 1]).shape[0]

    # parcing db users id into {0,..,n_users}
    db_user_id_to_user_id = {}
    for index, db_user_id in enumerate(np.unique(lines[:, 0])):
        db_user_id_to_user_id[db_user_id] = index

    # parcing db movies id into {0,..,n_movies}
    db_movies_id_to_movies_id = {}
    for index, db_movies_id in enumerate(np.unique(lines[:, 1])):
        db_movies_id_to_movies_id[db_movies_id] = index

    # creating dictionary. Key: user_id Value: list of movies and ratings
    users_to_movies = {}
    for i in range(lines.shape[0]):
        user_id = db_user_id_to_user_id[lines[i, 0]]
        if user_id in users_to_movies:
            users_to_movies[user_id].append([
                db_movies_id_to_movies_id[lines[i, 1]], lines[i, 2]
                ])
        else:
            users_to_movies[user_id] = [
                [db_movies_id_to_movies_id[lines[i, 1]], lines[i, 2]]
                ]
    return [
        n_users,
        n_movies,
        db_user_id_to_user_id,
        db_movies_id_to_movies_id,
        users_to_movies]
