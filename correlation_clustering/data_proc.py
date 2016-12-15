import numpy as np


def build_graph(tsvfilename):
    f = open(tsvfilename)
    line1 = f.readline().split('\t')

    if is_number(line1[0]):
        lines = np.loadtxt(tsvfilename)
    else:
        lines = np.loadtxt(tsvfilename, skiprows=1)


def is_number(s):
    ''' Test if the type of the object is float
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

    ''' Read the tsv file and create the useful tools for the graph construction
        Args:
            - tsvfilename : tsv file with 3 columns: user movie rating
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
    output_dictionary = {}
    output_dictionary['n_users'] = n_users
    output_dictionary['n_movies'] = n_movies
    output_dictionary['db_user_id_to_user_id'] = db_user_id_to_user_id
    output_dictionary['db_movies_id_to_movies_id'] = db_movies_id_to_movies_id
    output_dictionary['users_to_movies'] = users_to_movies
    return [
        output_dictionary]


def build_A(n_users, n_movies, users_to_movies):
    ''' Build A (ndarray) that summarize the rating of movies by users.
        Args:
            - n_users : number of users (type: float)
            - n_movies : number of movies (type: float)
            - users_to_movies : users rating movies (type: dictionary)
        Output:
            - A (type: ndarray)
        Remark: for now, not watching a movie or giving a grade of 2.5 is equal
    '''

    A = np.zeros(shape=(n_users, n_movies))
    for user in range(n_users):
        for i, elt in enumerate(users_to_movies[user]):
            A[user, elt[0]] = elt[1]
    return A


def build_graph(tsvfilename):
    ''' Build W (ndarray) the adjacency matrix of the graph.
        Args:
            - tsvfilename (string): the source file for the graph construction
        Output:
            - W (ndarray): adjacency matrix
        Remark: for now, not watching a movie or giving a grade of 2.5 is equal
    '''
    # read file
    output_dictionary = build_user_to_movies(tsvfilename)

    # construct A using the specialized function
    A = build_A(output_dictionary['n_users'],
                output_dictionary['n_movies'],
                output_dictionary['users_to_movies']
                )
    # build W

    # size (n_users + n_movies, n_users + n_movies)
    W = np.zeros(shape=(output_dictionary['n_users'] +
                        output_dictionary['n_movies'],
                        output_dictionary['n_users'] +
                        output_dictionary['n_movies']
                        ))
    # A in the top right corner, A^T en left down corner
    W[0:output_dictionary['n_users'], output_dictionary['n_users']:end] = A
    W[output_dictionary['n_users']:end, 0:output_dictionary['n_users']] = \
        A.transpose()
    return W
