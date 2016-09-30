import theano
import theano.tensor as T
import numpy as np
import math
import matplotlib.pyplot as plt

from loaddata import get_data

def load_split_data(data_size, test_p):
    # Load data and split into train set, test set randomly.
    # data_size is either "100k", "1m", "10m" or "20m".
    # test_p is a float between 0 - 1 indicating the portion of data hold out as test set
    print("split data randomly")
    # Load ratings, data is already permuted in get_data
    ratings = get_data(data_size)
    nb_users = int(np.max(ratings[:, 0]))
    nb_movies = int(np.max(ratings[:, 1]))
    # split test/train set
    test_size = int(len(ratings) * test_p)
    test_ratings = ratings[:test_size]
    train_ratings = ratings[test_size:]
    # train_ratings is converted into a matrix
    train_M = np.zeros((nb_movies, nb_users), dtype = np.float32)
    for rating in train_ratings:
        train_M[int(rating[1]-1), int(rating[0]-1)] = rating[2]
    # save test and train data in case more training is needed on this split
    np.save("Data/" + data_size + "_" + str(int(test_p * 100))+ "percent_test.npy", test_ratings)
    np.save("Data/" + data_size + "_" + str(int(test_p * 100))+ "percent_trainM.npy", train_M)
    # test_ratings is numpy array of user id | item id | rating
    # train_M is numpy array with nb_movies rows and nb_users columns, missing entries are filled with zero
    return test_ratings, train_M, nb_users, nb_movies, len(train_ratings)

def cal_RMSE(prediction_M, test_ratings):
    RMSE = 0
    for rating in test_ratings:
        RMSE += (rating[2] - prediction_M[int(rating[1] - 1), int(rating[0] - 1)])**2
    RMSE = math.sqrt(RMSE / len(test_ratings))
    return RMSE

def train_auto(data_size = "100k", nb_epoch = 10, test_p = 0.1, nb_hunits = 10, lambda_reg = 0.001, learningrate = 0.01):
    test_ratings, train_M, nb_users, nb_movies, k = load_split_data(data_size, test_p)
    prediction_M = np.zeros((nb_movies, nb_users), dtype = np.float32)
    RMSE_list = [0] * nb_epoch

    # set up theano autoencoder structure and update function
    X = T.dvector("input")
    X_observed = T.dvector("observedIndex")
    update_matrix = T.matrix("updateIndex")
    V = theano.shared(np.random.randn(nb_hunits, nb_users), name='V')
    miu = theano.shared(np.zeros(nb_hunits), name='miu')
    W = theano.shared(np.random.randn(nb_users, nb_hunits), name='W')
    b = theano.shared(np.zeros(nb_users), name='b')
    z1 = T.nnet.sigmoid(V.dot(X) + miu)
    z2 = W.dot(z1) + b
    loss_reg = 1.0/nb_movies * lambda_reg/2 * (T.sum(T.sqr(V)) + T.sum(T.sqr(W)))
    loss = T.sum(T.sqr((X - z2) * X_observed)) + loss_reg
    gV, gmiu, gW, gb = T.grad(loss, [V, miu, W, b])

    train = theano.function(
          inputs=[X, X_observed, update_matrix],
          outputs=[z2],
          updates=((V, V - learningrate * gV * update_matrix),(miu, miu - learningrate * gmiu),
              (W, W - learningrate * gW * update_matrix.T), (b, b - learningrate * gb * X_observed)))

    for j in range(nb_epoch):
        print(str(j + 1) + " epoch")
        for i in np.random.permutation(nb_movies):
            Ri = train_M[i, :]
            Ri_observed = Ri.copy()
            Ri_observed[Ri > 0] = 1
            update_m = np.tile(Ri_observed, (nb_hunits, 1))
            Ri_predicted = train(Ri, Ri_observed, update_m)
            prediction_M[i, :] = np.array(Ri_predicted)
        RMSE_list[j] = cal_RMSE(prediction_M, test_ratings)

    print("done")
    plt.figure(1)
    plt.plot(list(range(nb_epoch)), RMSE_list, 'rs-')
    plt.show()

    return nb_epoch, RMSE_list



