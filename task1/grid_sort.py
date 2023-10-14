import numpy as np


#assigns each point idx coordinate for grid
def get_grid_coord(points, n_squares):
    """converts points with continuous coordinates into their respective grid squares

    Args:
        points (np.ndarray[float] nx2): 2d coordinates
        n_squares (int): number of squares grid is supposed to have

    Returns:
        np.ndarray[int] nx2: 2d grid coordinates
    """
    grid_points = np.linspace(0,1, n_squares)
    #print(grid_points)
    def get_grid_idx(point, grid_points):
        x, y = point
        idx_x = np.searchsorted(grid_points, x)
        idx_y = np.searchsorted(grid_points, y)
        return idx_x, idx_y
    
    return [get_grid_idx(p, grid_points) for p in points]


#in: point list 
#
def grid_sort(points, n_squares=50):
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
        idxs_in_square[x_g,y_g].append(idx)
    
    return idxs_in_square


def test_if_empty(idxs_in_square):
    for i in range(idxs_in_square.shape[0]):
        for j in range(idxs_in_square.shape[1]):
            if idxs_in_square[i,j] == []:
                 print(f"empty square {i, j}")


def main():
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt("train_y.csv", skiprows=1, dtype="float", delimiter=",")
    train_x = train_x[train_y>=0]
    #print(train_x[:, :2])
    #print(get_grid_coord(train_x[:, :2], 10))
    grid = grid_sort(train_x[:, :2], n_squares=5)

    test_if_empty(grid)





if __name__ == "__main__":
	main()