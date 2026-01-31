import open3d as o3d
import numpy as np

from scipy.spatial import ConvexHull, QhullError

def plane_ransac_mesh(mesh, num_iterations=2000, distance_threshold=0.1, rng = None):
    """ 
    Detect plane hypotheses on a TriangleMesh using RANSAC over mesh vertices.    
    
    Each RANSAC iteration: 
        - Samples 3 mesh vertices and fits a plane (normal, d) with equation normal·x + d = 0
        - Marks vertex inliers by point-to-plane distance < distance_threshold
        - Estimates plane coverage as the sum of areas of triangles whose 3 vertices are inliers 
    
    Plane hypotheses are returned sorted by covered area (descending).
    
    Args: 
        mesh: Open3D TriangleMesh.
        num_iterations: Number of RANSAC iterations.
        distance_threshold: Inlier threshold for point-to-plane distance.
        rng: Optional NumPy random generator. If None, a new generator is created.
        
    Returns: 
        List of tuples (area, normal, d, inliers):
            - area: estimated plane area coverage on the mesh
            - normal: unit normal vector
            - d: plane offset in equation normal·x + d = 0
            - inliers: vertex indices 
            
    """
    points = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    planes = []

    if rng is None:
        rng = np.random.default_rng()
        
    v0 = points[triangles[:, 0]]
    v1 = points[triangles[:, 1]]
    v2 = points[triangles[:, 2]]
    
    tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    
    n_points = len(points)
    if n_points < 3:
        return []
    
    for _ in range(num_iterations):
        indices = rng.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[indices]
        
        n = np.cross(p2 - p1, p3 - p1)
        n_norm = np.linalg.norm(n)  
        if n_norm < 1e-12:  
            continue
        n = n / n_norm
        d = -np.dot(n, p1)

        distances = np.abs(points @ n + d)
        inlier_mask = distances < distance_threshold
        inliers = np.flatnonzero(inlier_mask)

        tri_inliers = inlier_mask[triangles].all(axis=1)
        area = tri_areas[tri_inliers].sum()
        
        if area > 0:
            planes.append((area, n, d, inliers))
    
    planes.sort(key=lambda x: x[0], reverse=True)
    
    return planes

def plane_ransac_points(pcd, num_iterations=2000, distance_threshold=0.1, min_inliers=50, min_area=1e-6, rng = None):
    """ 
    Detect plane hypotheses on a point cloud using RANSAC.
    
    Each RANSAC iteration: 
        - Samples 3 points and fits a plane (normal, d) with equation normal·x + d = 0
        - Marks point inliers by point-to-plane distance < distance_threshold
        - Projects inlier points onto a 2D coordinate system on the plane and estimates the inlier footprint area using a 2D convex hull
        - Scores the plane by density = (number of inliers) / (estimated area)
    
    Plane hypotheses are returned sorted by the density (descending).
    
    Args: 
        pcd: Open3D PointCloud. 
        num_iterations: Number of RANSAC iterations.
        distance_threshold: Inlier threshold for point-to-plane distance.
        min_inliers: Minimum number of inlier points required to accept a plane. 
        min_area: Minimum convex hull area required to accept a plane hypothesis.
        rng: Optional NumPy random generator. If None, a new generator is created.
        
    Returns: 
        List of tuples (inlier_count, normal, d, inliers):
            - inlier_count: number of inlier points
            - normal: unit normal vector
            - d: plane offset in equation normal·x + d = 0
            - inliers: point indices 
            
    """

    points = np.asarray(pcd.points)
    n_points = len(points)
    if n_points < 3:
        return []

    if rng is None:
        rng = np.random.default_rng()

    planes = []

    for _ in range(num_iterations):
        indices = rng.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[indices]

        n = np.cross(p2 - p1, p3 - p1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        d = -np.dot(n, p1)

        distances = np.abs(points @ n + d)
        inlier_mask = distances < distance_threshold
        inliers = np.flatnonzero(inlier_mask)
        inlier_count = inliers.size
        
        if inlier_count < min_inliers:
            continue
        
        inlier_pts = points[inliers]
        
        a = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, n)) > 0.9:
            a = np.array([0.0, 1.0, 0.0])

        u = np.cross(n, a)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        
        x2d = np.column_stack((inlier_pts @ u, inlier_pts @ v))
        
        try:
            hull = ConvexHull(x2d)
            area = hull.volume
        except QhullError:
            area = 0.0

        if area < min_area:
            continue
      
        density = inlier_count / area 
        
        planes.append((density, inlier_count, n, d, inliers))
    
    planes.sort(key=lambda x: x[0], reverse=True)
    
    return [(inlier_count, n, d, inliers) for _, inlier_count, n, d, inliers in planes]

def remove_planes_mesh(mesh, inliers):
    """ 
    Remove a detected plane from a TriangleMesh by deleting triangles fully covered by inlier vertices.
    A triangle is removed if all three of its vertex indices are contained in inliers.

    Args:
        mesh: Open3D TriangleMesh.
        inliers: Iterable of vertex indices belonging to the plane to remove.

    Returns:
        A new Open3D TriangleMesh with the plane triangles removed.
    """

    mesh = o3d.geometry.TriangleMesh(mesh)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    n_verts = np.asarray(mesh.vertices).shape[0]

    inlier_mask = np.zeros(n_verts, dtype=bool)
    inlier_mask[np.asarray(inliers, dtype=np.int64)] = True

    keep = ~inlier_mask[triangles].all(axis=1)

    mesh.triangles = o3d.utility.Vector3iVector(triangles[keep])
    mesh.remove_unreferenced_vertices()
    return mesh

def remove_planes_points(pcd, inliers, color=(0.75, 0.75, 0.75)):
    """
    Remove a detected plane from a PointCloud by deleting points.
    Build a new point cloud containing only the points whose indices are not in inliers.
    
    Args:
        pcd: Open3D PointCloud.
        inliers: Iterable of point indices belonging to the plane to remove.
        color: RGB color in [0,1] used to paint all remaining points.
        
    Returns:
        A new Open3D PointCloud with the plane points removed.
    """
       
    pts = np.asarray(pcd.points)
    n = len(pts)

    mask = np.ones(n, dtype=bool)
    mask[np.asarray(inliers, dtype=np.int64)] = False

    filtered_points = pts[mask]

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    colors = np.tile(np.array(color, dtype=np.float64), (filtered_points.shape[0], 1))
    new_pcd.colors = o3d.utility.Vector3dVector(colors)

    return new_pcd