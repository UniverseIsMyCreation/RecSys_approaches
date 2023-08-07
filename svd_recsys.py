from tools import *


def svd_decompose_matrix(matrix, type_to_save, k=None):
    """
    type_to_save: r - best approximation of matrix A with matrix of rank r 
                  own_k - your own decision how many information you want to say (more than r or less),
                  this way you need to input parameter k
    """
    from numpy.linalg import svd
    U, sigma, Vt = svd(matrix, full_matrices=True)
    
    if type_to_save == 'r':
        non_zero_singular_values = sigma[sigma > 0]
        r = non_zero_singular_values.shape[0]
        U = U[:, :r]
        Vt = Vt[:r, :]
        sigma_diag = np.diag(non_zero_singular_values)

    elif type_to_save == 'own_k':
        assert not k is None
        assert k <= sigma.shape[0]

        sigma = sigma[:k]
        U = U[:, :k]
        Vt = Vt[:k, :]
        sigma_diag = np.diag(sigma)
    
    else:
        assert type_to_save in ['r', 'own_k']
    
    predictions = np.dot(np.dot(U, sigma_diag), Vt)
    return predictions

#predictions = svd_decompose_matrix(item_based_matrix_forward, 'r', 100)


def get_recommendation_for_user(predictions, userId, movies, item_based_matrix, top_k):
    """
    try to find best movie for user, which he hasn't seen yet
    predictions: matrix with preds
    userId: user who we want to recommend something new
    movies: description for movies
    item_based_matrix: origin matrix with ratings
    tok_k: k best recommendations
    """
    # to prepare index for pandas dataframe
    userId -= 1

    # get preds for user and his ratings
    preds_for_user = predictions[userId, :]
    ratings_for_user = item_based_matrix[userId, :]
    indexes = np.arange(max_movie_id)

    # get new recommendations for user
    mask = np.invert(ratings_for_user > 0)
    preds_for_user_index = indexes[mask]
    preds_for_user_new = preds_for_user[mask]

    # get top_k best recommendations
    recommendations_rating = preds_for_user_new[np.argsort(preds_for_user_new)][::-1][:top_k]
    recommendations_index = preds_for_user_index[np.argsort(preds_for_user_new)][::-1][:top_k]

    # join the names and genres
    best_recommendations = movies.iloc[recommendations_index][['title', 'genres']]
    best_recommendations['pred_rating'] = recommendations_rating

    return best_recommendations

#userId = 100
#top_k = 10
#best_recommendations = get_recommendation_for_user(predictions, userId, movies, item_based_matrix_forward, top_k)


# evaluate svd with different k
top_k_list = [i for i in range(100, 2500, 100)]
rmse_scores = list()
for top_k in top_k_list:
    predictions = svd_decompose_matrix(item_based_matrix_train, 'own_k', k=top_k)
    rmse = evaluate(predictions, item_based_matrix_test)
    rmse_scores.append(rmse)
# item_based_matrix_train - train user/item matrix
# item_based_matrix_test - test user/item matrix


# draw the results 
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.plot(top_k_list, rmse_scores)
ax.set_title('RMSE depending on k in svd decompression')
ax.grid(True)
ax.set_xlabel('k')
ax.set_ylabel('RMSE')

plt.show()