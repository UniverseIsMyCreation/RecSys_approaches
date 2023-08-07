from tools import *


def pirson_corr(item_based_matrix, Id1, Id2, sim_type):
    """
    Pirson correlation
    sim_type: user or item
    """
    # get known ratings
    if sim_type == 'user':
        Id1_ratings = item_based_matrix[Id1]
        Id2_ratings = item_based_matrix[Id2]
    elif sim_type == 'item':
        Id1_ratings = item_based_matrix[:, Id1]
        Id2_ratings = item_based_matrix[:, Id2]
    else:
        sim_type in ['user', 'item']

    # either first or second user is empty
    if (not (np.sum(Id1_ratings) > 0)) or (not (np.sum(Id2_ratings) > 0)):
        return 0

    # calculate mean across each user
    userId1_mean = np.mean(Id1_ratings[Id1_ratings > 0])
    userId2_mean = np.mean(Id2_ratings[Id2_ratings > 0])

    # calculate statistics
    userId1_std = np.sqrt(np.sum((Id1_ratings[Id1_ratings > 0] - userId1_mean) ** 2))
    userId2_std = np.sqrt(np.sum((Id2_ratings[Id2_ratings > 0] - userId2_mean) ** 2))
    
    mask = (Id1_ratings > 0) & (Id2_ratings > 0)
    # no similarity
    if not (mask.sum() > 0):
        return 0
    users_cov = np.sum((Id1_ratings[mask] - userId1_mean) * (Id2_ratings[mask] - userId2_mean))

    # either first or second std is zero
    if userId2_std == 0 or userId1_std == 0:
        return 0
    
    # return result
    return users_cov / (userId2_std * userId1_std)


def jakkar_metric(item_based_matrix, Id1, Id2, sim_type):
    """
    Jakkar_metric
    sim_type: user or item
    """
    # get known ratings
    if sim_type == 'user':
        Id1_ratings = item_based_matrix[Id1]
        Id2_ratings = item_based_matrix[Id2]
    elif sim_type == 'item':
        Id1_ratings = item_based_matrix[:, Id1]
        Id2_ratings = item_based_matrix[:, Id2]
    else:
        sim_type in ['user', 'item']

    # calculate statistics
    intersect = np.sum((Id1_ratings > 0) & (Id2_ratings > 0))
    union = np.sum((Id1_ratings > 0) | (Id2_ratings > 0))

    # return result
    if union == 0:
        return 0
    return intersect / union


def cosine_sim(item_based_matrix, Id1, Id2, sim_type):
    """
    Cosine_sim
    sim_type: user or item
    """
    from numpy.linalg import norm

    # get known ratings
    if sim_type == 'user':
        Id1_ratings = item_based_matrix[Id1]
        Id2_ratings = item_based_matrix[Id2]
    elif sim_type == 'item':
        Id1_ratings = item_based_matrix[:, Id1]
        Id2_ratings = item_based_matrix[:, Id2]
    else:
        sim_type in ['user', 'item']

    # calculate statistics
    mask = (Id1_ratings > 0) & (Id2_ratings > 0)
    if not (np.sum(mask) > 0):
        return 0
    Id1_vector = Id1_ratings[mask]
    Id2_vector = Id2_ratings[mask]

    # return result
    return Id1_vector * Id2_vector / (norm(Id1_vector) * norm(Id2_vector))


def grouplens(item_based_matrix, metric):
    """
    Fill item-based matrix according grouplens algo
    this is realization for users' sim
    """
    
    user_amount = item_based_matrix.shape[0]
    movie_amount = item_based_matrix.shape[1]

    # calculate users' mean
    users_mean = np.zeros(user_amount)
    for userId in range(user_amount):
        cur_user = item_based_matrix[userId]
        mask = cur_user > 0
        if not (np.sum(mask) > 0):
            users_mean[userId] = 0
        else:
            users_mean[userId] = np.mean(cur_user[mask])

    # calculate pairwise distances matrix 
    pairwise_distances_matrix = np.zeros((user_amount, user_amount))
    for i in range(user_amount):
        for j in range(user_amount):
            if i == j:
                pairwise_distances_matrix[i, j] = 0
            else:
                pairwise_distances_matrix[i, j] = metric(item_based_matrix, i, j, 'user')
    
    item_based_matrix_filled = np.zeros(*item_based_matrix.shape)
    # fill gaps in item-based matrix
    for i in range(user_amount):
        for j in range(movie_amount):
            if item_based_matrix[i, j] > 0:
                item_based_matrix_filled[i, j] = item_based_matrix[i, j]
            else:
                # go through all other users
                item_other_users = item_based_matrix[:, j]
                
                mask = item_other_users > 0
                if not (np.sum(mask) > 0):
                    item_based_matrix_filled[i, j] = 0
                    continue

                cur_pairwise_distances = pairwise_distances_matrix[i][mask]
                cur_users_mean = users_mean[mask]
                cur_items = item_other_users[mask]
                if not (np.sum(cur_pairwise_distances) > 0):
                    item_based_matrix_filled = users_mean[i]
                    continue
                
                result = np.sum((cur_items - cur_users_mean) * cur_pairwise_distances)/np.sum(cur_pairwise_distances) + users_mean[i]
                item_based_matrix_filled[i, j] = result

    # return result
    return item_based_matrix_filled

#item_based_matrix_filled = grouplens(item_based_matrix_train, pirson_corr)


def user2user_recommendation(item_based_matrix, userId, metric, movies, recommendation_type='positive', top_k=5, rate_threshold=3.0):
    """
    recommendation_type: can be 'positive', if we want to recommend something, or 'negative', if we want to not recommend something
    userId: user, who we want to recommend something
    metric: sim func
    top_k: how many items we want to recommend
    movies: dataset with movies' information
    item_based_matrix: user/item matrix with gaps
    rate_threshold: rate threshold to recommend something
    """
    user_amount = item_based_matrix.shape[0]
    movie_amount = item_based_matrix.shape[1]
    userId_ratings = item_based_matrix[userId]
    # not enough info to recommend something
    if not (np.sum(userId_ratings) > 0):
        return None
    
    # calculate sim score between users accorting to given metric
    user2user_sim = np.zeros(user_amount)
    for idx in range(user_amount):
        user2user_sim[idx] = metric(item_based_matrix, userId, idx, 'user')
    
    # sort it
    sorted_indexes = np.argsort(user2user_sim)
    sorted_user2user_sim = np.sort(user2user_sim)

    movie_ids = list()
    # if we want to recommend something
    if recommendation_type == 'positive':
        # make sim descending
        sorted_indexes = sorted_indexes[::-1]
        sorted_user2user_sim = sorted_user2user_sim[::-1]
    # if we want to not recommend something
    elif recommendation_type == 'negative':
        pass
    # error in input data
    else:
        assert recommendation_type in ['positive', 'negative']

    # add all movies' ids of similar or dissimilar users
    for idx in sorted_indexes:
        # skip the user itself 
        if idx == userId:
            continue
        sim_user_indexes = np.arange(movie_amount)
        sim_user_ratings = item_based_matrix[idx]
        
        sim_user_indexes = sim_user_indexes[sim_user_ratings > rate_threshold]
        sim_user_ratings = sim_user_ratings[sim_user_ratings > rate_threshold]
        for sim_user_index in sim_user_indexes:
            if len(movie_ids) >= top_k:
                break
            movie_ids.append(sim_user_index)
        movie_ids = np.array(movie_ids)
    
    # join the names and genres
    best_recommendations = movies.iloc[movie_ids][['title', 'genres']]

    # return result
    return best_recommendations

#userId = 1000
#user2user_recommendation(item_based_matrix_train, userId, pirson_corr, movies, top_k=4)


def item2item_recommendation(item_based_matrix, userId, metric, movies, top_k=5, rate_threshold=3.0):
    """
    userId: user, who we want to recommend something
    metric: sim func
    top_k: how many items we want to recommend
    movies: dataset with movies' information
    item_based_matrix: user/item matrix with gaps
    rate_threshold: rate threshold to recommend something
    """
    movie_amount = item_based_matrix.shape[1]
    userId_ratings = item_based_matrix[userId]
    # not enough info to recommend something
    if not (np.sum(userId_ratings) > 0):
        return None
    
    # get all user's liked items
    movie_ids = np.arange(movie_amount)[userId_ratings > rate_threshold]
    userId_ratings = userId_ratings[userId_ratings > rate_threshold]

    # sort by most liked items
    movie_ids = movie_ids[np.argsort(userId_ratings)][::-1]

    # add top k the most sim items
    movie_ids_2_recommend = list()
    for item_idx in movie_ids:
        # calculate sim score between items accorting to given metric
        item2item_sim = np.zeros(movie_amount)
        for idx in range(movie_amount):
            item2item_sim[idx] = metric(item_based_matrix, item_idx, idx, 'item')
        
        # sort it
        sorted_indexes = np.argsort(item2item_sim)[::-1]

        # add all movies' ids of similar or dissimilar users
        for idx in sorted_indexes:
            # skip the item itself 
            if idx == item_idx:
                continue
            # if it's enough
            if len(movie_ids_2_recommend) >= top_k:
                movie_ids_2_recommend = np.array(movie_ids_2_recommend)

                # join the names and genres
                best_recommendations = movies.iloc[movie_ids_2_recommend][['title', 'genres']]
                
                # return result
                return best_recommendations

            # add movie
            movie_ids_2_recommend.append(idx)

    movie_ids_2_recommend = np.array(movie_ids_2_recommend)

    # join the names and genres
    best_recommendations = movies.iloc[movie_ids_2_recommend][['title', 'genres']]

    # return result
    return best_recommendations

#userId = 2000
#item2item_recommendation(item_based_matrix_train, userId, pirson_corr, movies, top_k=4)