from builtins import object
import numpy as np


class KNearestNeighbor(object):
    """a kNN classifier with L2 distance"""

    def __init__(self):
        """
        Initialize the KNN classifier.
        All data is passed via the train() method.
        """
        pass

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray, k=1) -> np.ndarray:
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """

        dists = self.compute_distances(X)
        return self.predict_locations(dists, k=k)

    def compute_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        # (x1 - x2)^2 = x1^2 + x2^2 - x1x2
        squared_sum = -2 * (X @ self.X_train.T)
        squared_sum += np.sum(np.power(X, 2), axis=1, keepdims=True)
        squared_sum += np.sum(np.power(self.X_train, 2), axis=1, keepdims=True).T

        dists = np.sqrt(squared_sum)

        return dists

    def predict_locations(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros([num_test, self.y_train.shape[1]])
        for i in np.arange(num_test):

            closest_y = []
            # A list of length k storing the labels of the k nearest neighbors to the ith test point.
            closest_y = self.y_train[np.argpartition(dists[i, :], k)[:k]]
            y_pred[i] = np.average(closest_y, axis=0)

        return y_pred
