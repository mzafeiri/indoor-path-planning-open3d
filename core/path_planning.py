import numpy as np
import heapq

def heuristic(a, b, D=1, D2=np.sqrt(2)):
    """ 
    Compute the A* heuristic cost between two grid cells using Octile distance.

    Octile distance is appropriate for 8-connected grids (moves allowed in 4 cardinal directions plus 4 diagonals). 
    It reduces to Manhattan distance when diagonal moves are not used.

    Args:
        a: Start cell as (x, y).
        b: Goal cell as (x, y).
        D: Cost of a cardinal (4-neighbor) move.
        D2: Cost of a diagonal move.

    Returns:
        Heuristic cost estimate from a to b.
    """
    
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

def a_star_search(grid, start, goal):
    """ 
    Find a shortest path on a 2D occupancy grid using A* search.

    The grid is treated as:
      - free cell: 1
      - blocked cell: otherwise

    Movement:
      - 8 directions (N, S, E, W + diagonals)
      - cost = 1 for cardinal moves, sqrt(2) for diagonal moves

    The heuristic used is the Octile distance.

    Args:
        grid: 2D NumPy array representing the occupancy grid.
        start: Start cell as (x, y) indices into grid.
        goal: Goal cell as (x, y) indices into grid.

    Returns:
        List of grid cells [(x0, y0), (x1, y1), ...] from start to goal or None if no path is found.
    """
    
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start))
    g_score = {start: 0.0}
    came_from = {}
    f_score = {start: heuristic(start, goal)}
    
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    direction_costs = [1, 1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)]

    def is_free(p):
        return grid[p[0], p[1]] == 1

    while open_set:
        current_f, current = heapq.heappop(open_set)
        if current_f != f_score.get(current, float("inf")):
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]  
        
        for direction, cost in zip(directions, direction_costs):
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            
            if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                continue
            if not is_free(neighbor):
                continue
            
            tentative_g_score = g_score[current] + cost 
                
            if tentative_g_score < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None 

def color_path_on_grid(grid, path):
    """ 
    Create a copy of a 2D grid and mark the cells along a path.

    This function does not modify the input grid. 
    It returns a new array where each (x, y) cell in path is set to value 3.

    Args:
        grid: 2D NumPy array.
        path: Iterable of (x, y) grid indices.

    Returns:
        colored_grid: Copy of original expanded grid with path cells set to 3.
    """
    
    colored_grid = np.copy(grid)
    for (x, y) in path:
        if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:  
            colored_grid[x, y] = 3 

    return colored_grid