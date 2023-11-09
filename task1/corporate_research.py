import numpy as np

import matplotlib.pyplot as plt





def subsample(feats, labels, do_Area=False):
    def build_matrix(feats, labels):
        n_samples = feats.shape[0]
        grid_sum = np.zeros((50,50))
        grid_num = np.zeros((50,50))
        if do_Area: grid_area = np.zeros((50,50))


        coords = np.linspace(0.0, 1.0, 50)

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

        grid_num[grid_num==0] = 1
        
        return np.divide(grid_sum,grid_num) #, grid_area


    def new_train(grid_avg, grid_area):
        it = np.nditer(grid_avg, flags=['f_index'])
        X = []
        y = []
        for i in range(50):
            for j in range(50):
                v = grid_avg[i,j]
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















