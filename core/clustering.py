import numpy as np

def get_neighbors(data, point_idx, eps=0.17):
    """ 
    Find indices of points within eps-radius (Euclidean distance) of a given point.

    Args:
        data: Array of shape (N, D) containing N points in D dimensions.
        point_idx: Index of the query point in data.
        eps: Neighborhood radius.

    Returns:
        List of indices of all points whose distance to data[point_idx] is < eps.
        point_idx is included, since its distance is 0.
    """

    data = np.asarray(data)
    p = data[point_idx]
    dists = np.linalg.norm(data - p, axis=1)
    return np.flatnonzero(dists < eps).tolist()

def expand_cluster(data, labels, point_idx, neighbors, cluster_id, eps=0.17, min_samples=10):
    """ 
    Expand a DBSCAN cluster starting from a seed point.

    During expansion:
      - Assign the seed point to cluster_id
      - Iteratively visit its neighbors
      - If a neighbor is a core point (has at least min_samples neighbors), add its neighbors to the search queue
      - Relabel noise points (-1) as cluster members when encountered during expansion.

    Args:
        data: Array of shape (N, D) containing N points in D dimensions.
        labels: Array of shape (N,) with DBSCAN labels:
                    -  0 = unvisited 
                    - -1 = noise 
                    - >0 = assigned cluster id
        point_idx: Index of the initial seed point.
        neighbors: Initial list of neighbor indices for point_idx (within eps).
        cluster_id: id of the cluster being grown.
        eps: Neighborhood radius.
        min_samples: Minimum number of neighbors required for a point to be considered a core point.
    """
    
    labels[point_idx] = cluster_id
    seen = set(neighbors)
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]
        
        if labels[neighbor_idx] == -1:
            labels[neighbor_idx] = cluster_id
        
        if labels[neighbor_idx] == 0:
            labels[neighbor_idx] = cluster_id
            new_neighbors = get_neighbors(data, neighbor_idx, eps)
            if len(new_neighbors) >= min_samples:
                for new_neighbor in new_neighbors:
                    if new_neighbor not in seen:
                        seen.add(new_neighbor)
                        neighbors.append(new_neighbor)
        i += 1

def dbscan(data, eps=0.17, min_samples=10):
    """ 
    Cluster points using the DBSCAN algorithm.

    Label convention:
        -  0: unvisited
        - -1: noise
        - >0: cluster id

    For each unvisited point:
      - Find its neighbors within eps-radius
      - If it has fewer than min_samples neighbors, mark it as noise (-1)
      - Otherwise, start a new cluster and expand it using expand_cluster

    Args:
        data: Array of shape (N, D) containing N points in D dimensions.
        eps: Neighborhood radius.
        min_samples: Minimum number of neighbors required for a point to be considered a core point.

    Returns:
        labels: NumPy array of shape (N,) with cluster labels.
    """
    labels = np.zeros(len(data))
    cluster_id = 0
    
    for point_idx in range(len(data)):
        if labels[point_idx] != 0:
            continue
        
        neighbors = get_neighbors(data, point_idx, eps)
        
        if len(neighbors) < min_samples:
            labels[point_idx] = -1  
        else:
            cluster_id += 1
            expand_cluster(data, labels, point_idx, neighbors, cluster_id, eps, min_samples)
    
    return labels