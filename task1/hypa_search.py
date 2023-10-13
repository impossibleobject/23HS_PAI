from solution import Model, cost_function, extract_city_area_information
import numpy as np
from joblib import dump, load  # sklearn-doc recommended this over pickle
# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv  # noqa
# now you can import normally from model_selection
from sklearn.model_selection import HalvingGridSearchCV as GS
from sklearn.metrics import make_scorer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn import preprocessing
# Set numpy random seed
np.random.seed(0)


import subsampling as ss

# The denominator by which we divide the number of samples
SUBSAMPLE_DENOMINATOR = 1

# Load the datasets
train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

# Extract the city_area information
train_x_2D, train_x_AREA, test_x_2D, test_x_AREA = extract_city_area_information(
    train_x, test_x)


def loss_function(y, y_pred):
    x_idxs = np.argwhere(map(lambda x : x in y, train_y))
    return cost_function(y, y_pred, train_x_AREA[x_idxs])
#scaler = preprocessing.StandardScaler().fit(train_x_2D)
#train_x_2D_scaled = scaler.transform(train_x_2D)

# TODO make actual random samplew
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
        ConstantKernel() * DotProduct(sigma_0_bounds=(1e-10, 1e10)),
        RBF(length_scale_bounds=(1e-10, 1e10)) * DotProduct(sigma_0_bounds=(1e-10, 1e10)),
        ConstantKernel() * DotProduct(sigma_0_bounds=(1e-10, 1e10)) + WhiteKernel(noise_level_bounds=(1e-10, 1e10)),
        #     ExpSineSquared(),
        ConstantKernel() * RationalQuadratic(length_scale_bounds=(1e-10, 1e10)),
        ConstantKernel() * Matern(length_scale_bounds=(1e-10, 1e10)) * DotProduct() + WhiteKernel(noise_level_bounds=(1e-10, 1e10)),
        ConstantKernel() * RationalQuadratic(length_scale=1.0, alpha=0.1)+ WhiteKernel(noise_level_bounds=(1e-10, 1e10)),
        ConstantKernel() * RationalQuadratic(length_scale=1.0, alpha=0.1)+ WhiteKernel(noise_level_bounds=(1e-10, 1e10)) + ConstantKernel(),ConstantKernel() * RBF(length_scale_bounds=(1e-10, 1e10)),
        ConstantKernel() * Matern(length_scale_bounds=(1e-10, 1e10)),
        ConstantKernel() * RationalQuadratic(length_scale=1.0, alpha=0.1)+ WhiteKernel(noise_level_bounds=(1e-10, 1e10)) + ConstantKernel() + DotProduct(),
        ConstantKernel() * RationalQuadratic(length_scale=1.0, alpha=0.1)* Matern(length_scale_bounds=(1e-10, 1e10))+ WhiteKernel(noise_level_bounds=(1e-10, 1e10)),
        ConstantKernel() * DotProduct(sigma_0_bounds=(1e-10, 1e10))+ WhiteKernel(noise_level_bounds=(1e-10, 1e10)),
        
    ]
}

clf = GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=20, alpha=0.01)

search = GS(clf,
            param_grid,
            refit=False,
            resource='n_samples',
            max_resources=N_SAMPLES // SUBSAMPLE_DENOMINATOR,
            random_state=0,
            n_jobs=-1,
            scoring=make_scorer(loss_function, greater_is_better=False)
            ).fit(train_x_2D, train_y)

params = search.get_params()
print(search.best_params_)
print(search.best_score_)
# Save the optimal paramters
dump(params, "parameters.joblib")
dump(search.best_params_, "best_params_.joblib")

# Safety thing: dump the search object
dump(search, "search.joblib")