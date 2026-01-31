import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError
from scipy.ndimage import binary_dilation

def create_occupancy_grid(pcd, labels, resolution=0.01, z_threshold=None):
    """ 
    Build a 2D occupancy grid by projecting clustered 3D point-cloud data onto the XY plane.

    For each cluster label (excluding noise label -1):
      - Keep only points with z <= z_threshold (to ignore higher points)
      - Compute the 2D convex hull of the remaining XY points
      - Fill the convex-hull polygon into the grid (mark those cells as occupied)

    Grid encoding:
      - 1: free cell
      - 0: occupied cell

    Args:
        pcd: Open3D PointCloud containing 3D points.
        labels: Array of shape (N,) with cluster labels for each point in pcd. Noise points are labeled as -1.
        resolution: Grid cell size in the same units as the point coordinates (e.g., meters per cell).
        z_threshold: Height cutoff. Only points with z <= z_threshold contribute to the projection. If None, a threshold at 2/3 of the z-range is used.

    Returns:
        grid: 2D NumPy array of shape (x_size, y_size) with values {0, 1}.
        x_min, y_min, x_max, y_max: Float bounds of the point cloud in XY, used for grid to world mapping.
    """
        
    points3D = np.asarray(pcd.points)
    if points3D.size == 0:
        raise ValueError("Point cloud is empty.")
    
    xy = points3D[:, :2]
    z = points3D[:, 2]

    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)

    x_size = int(np.floor((x_max - x_min) / resolution)) + 1
    y_size = int(np.floor((y_max - y_min) / resolution)) + 1
    
    grid = np.ones((x_size, y_size))

    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]
    
    if z_threshold is None:
        z_min, z_max = float(z.min()), float(z.max())
        z_threshold = z_min + (z_max - z_min) * (2.0 / 3.0) 

    for label in unique_labels:
        cluster_xy = xy[labels == label]
        cluster_z = z[labels == label]
        
        keep = cluster_z <= z_threshold
        cluster_xy = cluster_xy[keep]

        if cluster_xy.shape[0] < 3:
            continue

        try:
            hull = ConvexHull(cluster_xy)
        except QhullError:
            continue
        
        hull_pts = cluster_xy[hull.vertices]
        
        try:
            tri = Delaunay(hull_pts)
        except QhullError:
            continue
        
        hx_min, hy_min = hull_pts.min(axis=0)
        hx_max, hy_max = hull_pts.max(axis=0)

        i0 = max(0, int(np.floor((hx_min - x_min) / resolution)))
        i1 = min(x_size - 1, int(np.ceil((hx_max - x_min) / resolution)))
        j0 = max(0, int(np.floor((hy_min - y_min) / resolution)))
        j1 = min(y_size - 1, int(np.ceil((hy_max - y_min) / resolution)))

        if i1 < i0 or j1 < j0:
            continue

        xs = x_min + (np.arange(i0, i1 + 1) * resolution)
        ys = y_min + (np.arange(j0, j1 + 1) * resolution)
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        test_pts = np.column_stack([XX.ravel(), YY.ravel()])

        inside = tri.find_simplex(test_pts) >= 0
        inside = inside.reshape(XX.shape)

        sub = grid[i0:i1 + 1, j0:j1 + 1]
        sub[inside] = 0
        grid[i0:i1 + 1, j0:j1 + 1] = sub

    return grid, x_min, y_min, x_max, y_max

def expand_obstacles(grid, radius=0.1, resolution=0.01):
    """ 
    Expand occupied cells in a 2D occupancy grid by a given physical radius.

    The input grid is expected to use:
      - 0 for occupied cells
      - 1 for free cells

    This function performs a binary dilation of the occupied mask using a circular structuring element with radius ceil(radius / resolution) cells.
    Any free cell (value 1) that becomes covered by the dilation is marked as 2 (expanded obstacle).

    Args:
        grid: 2D NumPy array occupancy grid.
        radius: Expansion radius in the same units as the point cloud coordinates (e.g., meters).
        resolution: Grid cell size in the same units (e.g., meters per cell).

    Returns:
        expanded_grid: Copy of original grid where cells newly covered by the expansion are set to 2.
    """
    expand_cells = int(np.ceil(radius / resolution))    

    if expand_cells <= 0:
        return grid.copy()

    occupied = (grid == 0)

    yy, xx = np.ogrid[-expand_cells:expand_cells+1, -expand_cells:expand_cells+1]
    selem = (xx*xx + yy*yy) <= (expand_cells * expand_cells)

    dilated = binary_dilation(occupied, structure=selem)

    expanded_grid = grid.copy()
    expanded_grid[(dilated) & (grid == 1)] = 2
    
    return expanded_grid
