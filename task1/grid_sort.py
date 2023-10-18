import numpy as np
from numpy.random import choice

from itertools import chain

#set this a bit higher to account for spread between squares
DESIRED_SAMPLES = 2000
#P_CITY = 0.8
#P_NON_CITY = 0.1
np.random.seed(0)

#assigns each point idx coordinate for grid
def get_grid_coord(points, n_squares):
    """converts points with continuous coordinates into their respective grid squares

    Args:
        points (np.ndarray[float] nx2): 2d coordinates
        n_squares (int): number of squares grid is supposed to have

    Returns:
        np.ndarray[int] nx2: 2d grid coordinates
    """
    max_coord = points.max()
    grid_points = np.linspace(0,max_coord, n_squares+1)[1:]
    #print(len(grid_points))
    #print(grid_points)
    def get_grid_idx(point, grid_points):
        x, y = point
        #print(x,y)
        idx_x = np.searchsorted(grid_points, x)
        idx_y = np.searchsorted(grid_points, y)
        return idx_x, idx_y
    #print(points[0])
    #print(get_grid_idx(points[0], grid_points))
    return [get_grid_idx(p, grid_points) for p in points]


def subsample(idxs_in_square): #, test_x_AREA
    """subsamples based on if points are in city area

    Args:
        idxs_in_square (np.ndarray[list obj] n_squares x n_squares): list of idxs in points for each grid square
        test_x_AREA (np.ndarray[bool] n): is point at idx in city

    Returns:
        idxs_in_square (np.ndarray[list obj] n_squares x n_squares): subsampled input
    """
    n_squares = idxs_in_square.shape[0]
    idxs_square = DESIRED_SAMPLES//n_squares

    for i in range(n_squares):
       for j in range(n_squares):
            curr_idxs = idxs_in_square[i,j]
            
            if len(curr_idxs) > idxs_square:
                #get_prob = lambda i : P_CITY if test_x_AREA[i] else P_NON_CITY
                sub_idxs_area1 = choice(curr_idxs, idxs_square, replace=False) #, p=[get_prob[i] for i in curr_idxs]
                idxs_in_square[i,j] = list(sub_idxs_area1)

    
    return idxs_in_square

def grid_sort(points, n_squares, do_subsample=False):
    """gets idx list for each grid square

    Args:
        points (np.ndarray[float] nx2): 2d coordinates
        n_squares (int, optional): _description_. Defaults to 50.

    Returns:
        np.ndarray[list obj] n_squares x n_squares: list of idxs in points for each grid square
    """
    #avoid same list reference for each square
    idxs_in_square = np.zeros((n_squares, n_squares), dtype=object)
    for i in range(n_squares):
         for j in range(n_squares):
              idxs_in_square[i,j] = []

    #assign each point to corresponding grid square
    grid_coords = get_grid_coord(points, n_squares)
    for idx in range(len(points)):
        x_g, y_g = grid_coords[idx]
        #print(x_g, y_g)
        idxs_in_square[x_g,y_g].append(idx)
    if do_subsample:
        idxs_in_square = subsample(idxs_in_square)
    
    return idxs_in_square


def test_if_empty(idxs_in_square):
    for i in range(idxs_in_square.shape[0]):
        for j in range(idxs_in_square.shape[1]):
            if idxs_in_square[i,j] == []:
                 print(f"empty square {i, j}")





def main():
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    #train_y = np.loadtxt("train_y.csv", skiprows=1, dtype="float", delimiter=",")
    #train_x = train_x[train_y>=0]
    #print(train_x[:, :2])
    #print(get_grid_coord(train_x[:, :2], 10))
    train_X_2d = train_x[:, :2]
    grid1 = grid_sort(train_X_2d, n_squares=4, do_subsample=True)
    print(train_X_2d[list(chain.from_iterable([l for l in grid1]))[0]])
    #print(grid1)
    print("second one")
    #grid2 = grid_sort(train_X_2d, n_squares=4, do_subsample=True)
    #print(grid2)

    #test_if_empty(grid)





if __name__ == "__main__":
	main()