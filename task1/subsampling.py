import numpy as np

import matplotlib.pyplot as plt

from numpy.random import choice
from numpy.linalg import norm

np.random.seed(0)

def subsample(X, y, goal_samples, do_scale_w_y=False):
    idxs = np.array(list(range(X.shape[0])))
    #get y into prob range by sum normalization, set negative vals to 0
    y[y<0.] = 0.
    y_norm = y/y.sum()
    #print(y_norm)
    if do_scale_w_y:
        idxs_sub = choice(a=idxs, size=goal_samples, p=y_norm, replace=False)
    else:
        idxs_sub = choice(a=idxs, size=goal_samples, replace=False)
    return X[idxs_sub], y[idxs_sub]



#we don't use this anymore
def grid_subsample(feats, labels, n_squares=50, do_Area=False):
    """maps points to grid points averaging values if multiple points map to same one

    Args:
        feats (np.ndarray[float] nx2): 2d coordinates
        labels (np.ndarray[float] n): pollution values at coord
        do_Area (bool, optional): include area in subsampling. Defaults to False.
    """
    def build_matrix(feats, labels):
        #filter out negative labels
        feats = feats[labels>0]
        labels = labels[labels>0]

        n_samples = feats.shape[0]
        grid_sum = np.zeros((n_squares,n_squares))
        grid_num = np.zeros((n_squares,n_squares))
        if do_Area: grid_area = np.zeros((n_squares,n_squares))


        coords = np.linspace(0.0, 1.0, n_squares)

        def find_idx(discr_grid, cont_coord):
            discr_grid = np.asarray(discr_grid)
            idx = (np.abs(discr_grid - cont_coord)).argmin()
            return idx

        for i in range(n_samples):
            x,y = feats[i, :2]
            v = labels[i]

            x_idx = find_idx(coords, x)
            y_idx = find_idx(coords, y)

            grid_sum[x_idx, y_idx] += v
            grid_num[x_idx, y_idx] += 1
            if do_Area: grid_area[x_idx, y_idx] = feats[i, 2]
        
        #avoid division by 0
        grid_num[grid_num==0] = 1
        
        return np.divide(grid_sum,grid_num) #, grid_area


    def new_train(grid_avg):
        """maps training set of continuous coordinates to grid points

        Args:
            grid_avg (np.ndarray[float] n_squares x n_squares): grid with averaged values

        Returns:
            tuple(np.ndarray[float] (non_neg_samples)x2, np.ndarray[float] (non_neg_samples)): new feats, labels from grid without <0
        """
        it = np.nditer(grid_avg, flags=['f_index'])
        X = []
        y = []
        for i in range(50):
            for j in range(50):
                v = grid_avg[i,j]
                if v >0.:
                    y.append(v)
                    #X.append((i/50., j/50., grid_area[i,j]))
                    X.append((i/50., j/50.))

        X = np.array(X)
        y = np.array(y)
        return X, y


    #coord_matrix, grid_area = build_matrix(feats, labels)
    coord_matrix = build_matrix(feats, labels)
    return new_train(coord_matrix)

# train_X = np.loadtxt("train_x_orig.csv", skiprows=1, dtype="float", delimiter=",")
# train_y = np.loadtxt("train_y_orig.csv", skiprows=1, dtype="float", delimiter=",")

# idx = range(train_X.shape[0])
# print(train_X)



# coord_matrix, grid_area = build_matrix(train_X, train_y)
# X_subs, y_subs = new_train(coord_matrix, grid_area)
# print(X_subs)
# print(y_subs)


# np.savetxt("train_x.csv", X_subs, delimiter=',', header="lon,lat")
# np.savetxt("train_y.csv", y_subs, delimiter=',', header="pm25")

# do_plot = False
# if do_plot:
#     plt.scatter(X_subs[:,0], X_subs[:,1], y_subs)
#     plt.show()
    #plt.scatter(train_X[:,0], train_X[:,1])
    #plt.show()



def main():
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt("train_y.csv", skiprows=1, dtype="float", delimiter=",")
    #train_x = train_x[train_y>=0]
    #print(train_x[:, :2])
    #print(get_grid_coord(train_x[:, :2], 10))
    train_X_2d = train_x[:, :2]
    
    train_x_sub, train_y_sub = subsample(train_X_2d, train_y, 2000)
    #print(grid2)

    #test_if_empty(grid)





if __name__ == "__main__":
	main()












