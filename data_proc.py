import numpy as np


def is_number(s):
    ''' Test if the type of the object is float
        Args:
            - s (object)
        Output:
            - boolean: True if s is a number
    '''
    try:
        float(s)
        return True
    except ValueError:
        return False


def build_user_to_movies(tsvfilename):

    ''' Read the tsv file and create the useful tools for the graph construction
        Args:
            - tsvfilename (str): tsv file with 3 columns: user movie rating
        Output:
            - dictionary of 5 items
                - key: n_users - value: number of users
                - key: n_movies - value: number of movies
                - key: db_user_id_to_user_id - value: dictonary of
                                correspondances between file user ids
                                                        and our user ids
                - key: db_movies_id_to_movies_id - value: dictonary of
                                    correspondances between file movie ids
                                                            and our movie ids
                - key: users_to_movies - value: dictionary with user_id as key
                                            and all shown films and rates
                                                as value (list of list object)
    '''
    # reading the tsv file, parsing it into a ndarray
    f = open(tsvfilename)
    line1 = f.readline().split('\t')

    # checking header
    if is_number(line1[0]):
        lines = np.loadtxt(tsvfilename)
    else:
        lines = np.loadtxt(tsvfilename, skiprows=1)

    # closing file
    f.close()

    # number of unique users in the tsv
    n_users = np.unique(lines[:, 0]).shape[0]

    # number of movies in the tsv
    n_movies = np.unique(lines[:, 1]).shape[0]

    # parsing db users id into {0,..,n_users}
    db_user_id_to_user_id = {}
    for index, db_user_id in enumerate(np.unique(lines[:, 0])):
        db_user_id_to_user_id[db_user_id] = index

    # parsing db movies id into {0,..,n_movies}
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
    output_dictionary = {}
    output_dictionary['n_users'] = n_users
    output_dictionary['n_movies'] = n_movies
    output_dictionary['db_user_id_to_user_id'] = db_user_id_to_user_id
    output_dictionary['db_movies_id_to_movies_id'] = db_movies_id_to_movies_id
    output_dictionary['users_to_movies'] = users_to_movies
    return output_dictionary


def build_a(n_users, k_users, n_movies, k_movies, users_to_movies):
    ''' Build  (ndarray) that summarize the rating of movies by users.
        Args:
            - k_users (int): number of users considered < n_users
            - n_users (int): total number of users
            - n_movies (int): number of movies
            - users_to_movies (dict): users rating movies
        Output:
            - a (ndarray): [i, j] gives the rating of user i to film j
        Remark: we changed the rate 2.5 to 2.6
    '''

    a = np.zeros(shape=(k_users, k_movies))
    # good_users have seen more than k_movies movies
    good_users = []
    for user in range(n_users):
        if len(users_to_movies[user]) > k_movies-1:
            good_users.append(user)
    good_users = np.array(good_users)
    selected_users = np.random.choice(good_users, k_users, False)
    new_user_id = {}
    # reattribute an id between 0 and k_users-1 to the selected users
    for index, old_user_id in enumerate(selected_users):
        new_user_id[old_user_id] = index
    # list the movies seen by users selected
    seen_movies_id = {}
    for user in selected_users:
        for i, (movie_id, rating) in enumerate(users_to_movies[user]):
            keys = seen_movies_id.keys()
            if movie_id in keys:
                seen_movies_id[movie_id] += 1
            else:
                seen_movies_id[movie_id] = 1
    # select k_movies movie_id within the list seen_movies_id
    list_seen_movies_id = []
    for movie_id in seen_movies_id.keys():
        if seen_movies_id[movie_id] > 1:
            list_seen_movies_id.append(movie_id)
    k_movies_selected = np.random.choice(list_seen_movies_id, k_movies, False)
    new_movie_id = {}
    # reattribute an id between 0 and k_movies-1 to the slected movies
    for index, old_movie_id in enumerate(k_movies_selected):
        new_movie_id[old_movie_id] = index
    for user in selected_users:
        for i, (movie_id, rating) in enumerate(users_to_movies[user]):
            # clearing up ambiguity concerning a 2.5 rate and non watched movie
            if movie_id in k_movies_selected:
                if float(rating) == 0:
                    a[new_user_id[user], new_movie_id[movie_id]] = 0.1
                else:
                    a[new_user_id[user],
                        new_movie_id[movie_id]] = float(rating)
    return a


def build_graph(k_users, k_movies, a):
    ''' Build w (ndarray) the adjacency matrix of the graph.
        Args:
            - k_users (int): number of users considered
            - n_movies (int): number of movies
            - a (ndarray): the matrix giving the rating of a movie by a user
        Output:
            - w (ndarray): adjacency matrix
    '''
    n = k_users + k_movies
    # size (n_users + n_movies, n_users + n_movies)
    w = np.zeros([n, n])
    # A in the top right corner, A^T en left down corner
    w[0:k_users, k_users:n] = a
    w[k_users:n, 0:k_users] = a.transpose()
    return w
