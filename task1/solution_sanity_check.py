# import os
# import typing
# from sklearn.gaussian_process.kernels import *
# import numpy as np
# from sklearn.gaussian_process import GaussianProcessRegressor
# import matplotlib.pyplot as plt
# from matplotlib import cm



# Standard scientific Python imports
import numpy as np
import pickle
from time import time

import os
import typing

from matplotlib import cm
import matplotlib.pyplot as plt

import sklearn.gaussian_process as gp
import sklearn.gaussian_process.kernels as ker

from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import (RBFSampler, Nystroem)
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic, WhiteKernel
from sklearn.preprocessing import StandardScaler



# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0

SUBSAMPLE_DENOMINATOR = 1



## Constant for Cost function
THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.04

def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array

        return: float
    """
    cost = (true - predicted)**2

    # true above threshold (case 1)
    mask = true >= THRESHOLD
    mask_w1 = np.logical_and(predicted>=true,mask)
    mask_w2 = np.logical_and(np.logical_and(predicted<true,predicted > THRESHOLD),mask)
    mask_w3 = np.logical_and(predicted<=THRESHOLD,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2
    cost[mask_w3] = cost[mask_w3]*W3

    # true value below threshold (case 2)
    mask = true < THRESHOLD
    mask_w1 = np.logical_and(predicted>true,mask)
    mask_w2 = np.logical_and(predicted<=true,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2

    #reward = W4*np.logical_and(predicted <= THRESHOLD,true<=THRESHOLD)
    cost2 = W4*(np.logical_and(predicted >= THRESHOLD,true<=THRESHOLD).astype(int)
    - np.logical_and(predicted <= THRESHOLD,true<=THRESHOLD).astype(int))
    if cost2 is None:
        cost2 = 0

    return np.mean(cost) + np.mean(cost2)

"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:


    M = Model()
    M.fit_model(train_x,train_y)
    prediction = M.predict(test_x)

It uses predictions to compare to the ground truth using the cost_function above.
"""


class Model():

    def __init__(self):
        """
            TODO: enter your code here
        """
        pass

    def preprocess(self, train_x, train_y, num_dense_samples):
        """
            Data is heavily imbalaced, there is a lot of datapoints (17100) in the [-1, -0.5] x0 region
            and only few datapoints (150) in the [-0.5, 1] x0 region.

            We subsample the dense region to create a balanced dataset and at the same time deal with
            the large scale of the dataset.

            Args:
                train_x (numpy.array):  training data points
                train_y (numpy.array):  training labels
                num_dense_labels (int): number of dense data points that should be subsampled
            Returns:
                train_x_s (numpy.array):    uniformly sampled and balanced training data
                train_y_s (numpy.array):    corresponding training labels
        """
        # separate all the valuable datapoints
        train_x_sparse = train_x[train_x[:,0] > -0.5]
        train_y_sparse = train_y[train_x[:,0] > -0.5]

        train_x_dense = train_x[train_x[:,0] <= -0.5]
        train_y_dense = train_y[train_x[:,0] <= -0.5]

        dense_data = np.concatenate([train_x_dense, train_y_dense.reshape(-1, 1)], axis=1)
        np.random.shuffle(dense_data)

        # random shuffle datapoints in the dense region and select 150 to balance the data
        train_x_dense_sampled = dense_data[:num_dense_samples,:2]
        train_y_dense_sampled = dense_data[:num_dense_samples,2]

        train_x_s = np.concatenate([train_x_sparse, train_x_dense_sampled], axis=0)
        train_y_s = np.concatenate([train_y_sparse, train_y_dense_sampled])

        assert train_x_s.shape[0] == num_dense_samples + train_x_sparse.shape[0]
        assert train_y_s.shape[0] == num_dense_samples + + train_x_sparse.shape[0]

        print("Sampled train_x shape:", train_x_s.shape)
        print("Sampled train_y shape:", train_y_s.shape)

        self.scaler = StandardScaler().fit(train_x_s)

        return train_x_s, train_y_s

    def predict(self, test_x, test_x_area):
        """
            Predict labels for test data
        """

        y = self.model.predict(test_x)
        # we add a safety increase to 'safe' predictions since false negatives are penalized harshly
        predict_safe = (y < THRESHOLD).astype(int)
        y += 0.12 * predict_safe

        return y

    def fitting_model(self, train_x, train_y):
        """
            Fit a Gaussian process regressor with noisy Matern kernel to the given data
        """

        train_x, train_y = self.preprocess(train_x, train_y, 1500)

        k = ker.Matern(length_scale=0.01, nu=2.5) + \
            ker.WhiteKernel(noise_level=1e-05)

        gpr = gp.GaussianProcessRegressor(kernel=k, alpha=0.01, n_restarts_optimizer=20, random_state=42, normalize_y=True)
        noisyMat_gpr = pipeline.Pipeline([
            ("scaler", self.scaler),
            ("gpr", gpr)
        ])


        print("Fitting noisy Matern GPR")
        start = time()
        noisyMat_gpr.fit(train_x, train_y)
        print("Took {} seconds".format(time() - start))

        self.model = noisyMat_gpr


def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')

    M = Model()
    M.fit_model(train_x, train_y)
    prediction = M.predict(test_x)

    print(prediction)


if __name__ == "__main__":
    main()






# You don't have to change this function
def cost_function(ground_truth: np.ndarray, predictions: np.ndarray,
                  AREA_idxs: np.ndarray) -> float:
	"""
	Calculates the cost of a set of predictions.

	:param ground_truth: Ground truth pollution levels as a 1d NumPy float array
	:param predictions: Predicted pollution levels as a 1d NumPy float array
	:param AREA_idxs: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
	:return: Total cost of all predictions as a single float
	"""
	assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

	# Unweighted cost
	cost = (ground_truth - predictions)**2
	weights = np.ones_like(cost) * COST_W_NORMAL

	# Case i): underprediction
	mask = (predictions <
	        ground_truth) & [bool(AREA_idx) for AREA_idx in AREA_idxs]
	weights[mask] = COST_W_UNDERPREDICT

	# Weigh the cost and return the average
	return np.mean(cost * weights)


# You don't have to change this function
def is_in_circle(coor, circle_coor):
	"""
	Checks if a coordinate is inside a circle.
	:param coor: 2D coordinate
	:param circle_coor: 3D coordinate of the circle center and its radius
	:return: True if the coordinate is inside the circle, False otherwise
	"""
	return (coor[0] - circle_coor[0])**2 + (
	    coor[1] - circle_coor[1])**2 < circle_coor[2]**2


# You don't have to change this function
def determine_city_area_idx(visualization_xs_2D):
	"""
	Determines the city_area index for each coordinate in the visualization grid.
	:param visualization_xs_2D: 2D coordinates of the visualization grid
	:return: 1D array of city_area indexes
	"""
	# Circles coordinates
	circles = np.array([[0.5488135, 0.71518937, 0.17167342],
	                    [0.79915856, 0.46147936, 0.1567626],
	                    [0.26455561, 0.77423369, 0.10298338],
	                    [0.6976312, 0.06022547, 0.04015634],
	                    [0.31542835, 0.36371077, 0.17985623],
	                    [0.15896958, 0.11037514, 0.07244247],
	                    [0.82099323, 0.09710128, 0.08136552],
	                    [0.41426299, 0.0641475, 0.04442035],
	                    [0.09394051, 0.5759465, 0.08729856],
	                    [0.84640867, 0.69947928, 0.04568374],
	                    [0.23789282, 0.934214, 0.04039037],
	                    [0.82076712, 0.90884372, 0.07434012],
	                    [0.09961493, 0.94530153, 0.04755969],
	                    [0.88172021, 0.2724369, 0.04483477],
	                    [0.9425836, 0.6339977, 0.04979664]])

	visualization_xs_AREA = np.zeros((visualization_xs_2D.shape[0], ))

	for i, coor in enumerate(visualization_xs_2D):
		visualization_xs_AREA[i] = any(
		    [is_in_circle(coor, circ) for circ in circles])

	return visualization_xs_AREA


# You don't have to change this function
def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
	"""
	Visualizes the predictions of a fitted model.
	:param model: Fitted model to be visualized
	:param output_dir: Directory in which the visualizations will be stored
	"""
	print('Performing extended evaluation')

	# Visualize on a uniform grid over the entire coordinate system
	grid_lat, grid_lon = np.meshgrid(
	    np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) /
	    EVALUATION_GRID_POINTS,
	    np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) /
	    EVALUATION_GRID_POINTS,
	)
	visualization_xs_2D = np.stack((grid_lon.flatten(), grid_lat.flatten()),
	                               axis=1)
	visualization_xs_AREA = determine_city_area_idx(visualization_xs_2D)

	# Obtain predictions, means, and stddevs over the entire map
	predictions, gp_mean, gp_stddev = model.make_predictions(
	    visualization_xs_2D, visualization_xs_AREA)
	predictions = np.reshape(predictions,
	                         (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
	gp_mean = np.reshape(gp_mean,
	                     (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

	vmin, vmax = 0.0, 65.0

	# Plot the actual predictions
	fig, ax = plt.subplots()
	ax.set_title('Extended visualization of task 1')
	im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
	cbar = fig.colorbar(im, ax=ax)

	# Save figure to pdf
	figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
	fig.savefig(figure_path)
	print(f'Saved extended evaluation to {figure_path}')

	plt.show()


def extract_city_area_information(
    train_x: np.ndarray, test_x: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""
	Extracts the city_area information from the training and test features.
	:param train_x: Training features
	:param test_x: Test features
	:return: Tuple of (training features' 2D coordinates, training features' city_area information,
		test features' 2D coordinates, test features' city_area information)
	"""

	train_x_2D = train_x[:, :2]  #np.zeros((train_x.shape[0], 2), dtype=float)
	train_x_AREA = train_x[:, 2].astype("bool")  #np.zeros((train_x.shape[0],), dtype=bool)
	test_x_2D = test_x[:, :2]  #np.zeros((test_x.shape[0], 2), dtype=float)
	test_x_AREA = test_x[:, 2].astype("bool")  #np.zeros((test_x.shape[0],), dtype=bool)

	assert train_x_2D.shape[0] == train_x_AREA.shape[0] and test_x_2D.shape[
	    0] == test_x_AREA.shape[0]
	assert train_x_2D.shape[1] == 2 and test_x_2D.shape[1] == 2
	assert train_x_AREA.ndim == 1 and test_x_AREA.ndim == 1

	return train_x_2D, train_x_AREA, test_x_2D, test_x_AREA


# you don't have to change this function
def main():
	# Load the training dateset and test features
	#assert(False)
	train_x = np.loadtxt('train_x_subs.csv.npy', delimiter=',', skiprows=0)
	train_y = np.loadtxt('train_y_subs.csv.npy', delimiter=',', skiprows=0)
	test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

	# Extract the city_area information
	train_x_2D, train_x_AREA, test_x_2D, test_x_AREA = extract_city_area_information(
	    train_x, test_x)
	# Fit the model
	print('Fitting model')
	model = Model()
	model.fitting_model(train_y, train_x_2D)

	# Predict on the test features
	print('Predicting on test features')
	predictions = model.make_predictions(test_x_2D, test_x_AREA)
	print(predictions)

	if EXTENDED_EVALUATION:
		perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
	main()
