from solution import Model, cost_function, extract_city_area_information
import numpy as np
from joblib import dump, load  # sklearn-doc recommended this over pickle
# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv  # noqa
# now you can import normally from model_selection
from sklearn.model_selection import HalvingGridSearchCV as GS
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn import preprocessing
# Set numpy random seed
np.random.seed(0)

# The denominator by which we divide the number of samples
SUBSAMPLE_DENOMINATOR = 3

# Load the datasets
train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

# Extract the city_area information
train_x_2D, train_x_AREA, test_x_2D, test_x_AREA = extract_city_area_information(
    train_x, test_x)

#scaler = preprocessing.StandardScaler().fit(train_x_2D)
#train_x_2D_scaled = scaler.transform(train_x_2D)

# TODO make actual random sample
# Subsample training data by always using the highest data
#sorted_idxs = np.argsort(train_y)[-train_y.shape[0] //
#		                                  SUBSAMPLE_DENOMINATOR:]

N_SAMPLES = train_y.shape[0]
'''
idxs = np.random.choice(np.arange(N_SAMPLES), size=N_SAMPLES//SUBSAMPLE_DENOMINATOR, replace=False)
print(train_y[idxs].shape)

train_x_subs = train_x_2D[idxs]
train_y_subs = train_y[idxs]
'''

param_grid = {
    "kernel": [
        RBF(),
        Matern(),
        DotProduct(),
        #     ExpSineSquared(),
        RationalQuadratic(),
        Matern() * DotProduct() + WhiteKernel(),
        RationalQuadratic(length_scale=1.0, alpha=0.1)+ WhiteKernel(),
        RationalQuadratic(length_scale=1.0, alpha=0.1)+ WhiteKernel() + ConstantKernel(),
        RationalQuadratic(length_scale=1.0, alpha=0.1)+ WhiteKernel() + ConstantKernel() + DotProduct(),
        RationalQuadratic(length_scale=1.0, alpha=0.1)* Matern()+ WhiteKernel(),
        DotProduct()* Matern()+ WhiteKernel(),
        
    ]
}

clf = GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=4)

search = GS(clf,
            param_grid,
            refit=False,
            resource='n_samples',
            max_resources=N_SAMPLES // SUBSAMPLE_DENOMINATOR,
            random_state=0,
            n_jobs=-1).fit(train_x_2D, train_y)

params = search.get_params()
print(search.best_params_)
# Save the optimal paramters
dump(params, "parameters.joblib")
dump(search.best_params_, "best_params_.joblib")

# Safety thing: dump the search object
dump(search, "search.joblib")