import numpy as np
import open3d as o3d

from typing import Optional, Sequence

def euclidean_distance(point1, point2):
    """ 
    Compute the Euclidean distance between two points. 
    
    Args: 
        point1, point2: Array points of the same shape.
    
    Returns:
        Euclidean distance as a float.
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))

def triangle_area(v1, v2, v3):
    """ 
    Compute the area of a 3D triangle. 
    
    Args: 
        v1, v2, v3: Triangle vertices as array of shape (3,).
    
    Returns: 
        Triangle area as a float number.
    """
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

def create_line_set_from_mesh(mesh: o3d.geometry.TriangleMesh, color: Optional[Sequence[float]] = None) -> o3d.geometry.LineSet:
    """ 
    Create a wireframe LineSet from a TriangleMesh by extracting unique triangle edges.
    
    Args: 
        mesh: Open3D TriangleMesh
        color: RGB color for all edges in [0,1]. Default is blue.
        
    Returns: 
        Open3D LineSet with the mesh vertices as points and unique edges as lines.
    """
    if color is None:
        color = (0.0, 0.0, 1.0)
    
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
        
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    
    if triangles.size == 0:
        line_set.lines = o3d.utility.Vector2iVector(np.empty((0, 2), dtype=np.int64))
        line_set.colors = o3d.utility.Vector3dVector(np.empty((0, 3), dtype=np.float64))
        return line_set
        
    edges = set()
    for tri in triangles:
        edges.add(tuple(sorted((tri[0], tri[1]))))
        edges.add(tuple(sorted((tri[1], tri[2]))))
        edges.add(tuple(sorted((tri[2], tri[0]))))
  
    lines = np.array(list(edges), dtype=np.int64)
    line_set.lines  = o3d.utility.Vector2iVector(lines)

    colors = np.tile(np.array(color, dtype=np.float64), (len(lines), 1))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set