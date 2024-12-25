"""
Transformation Library

This module provides a collection of utility functions for transforming ARC matrices. 
They can be used in both the create_input() and transform_input() methods of the
ARCTaskGenerator class. Only frequently used methods, which are difficult for AI 
systems to create from scratch without errors are included here.
"""

import numpy as np
from scipy.ndimage import label

def get_objects(matrix: np.ndarray,
                diagonal_connectivity: bool = False,
                color: int = None):
    """
    Finds all connected objects in the matrix.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix to find objects in
    diagonal_connectivity : bool, optional (default=False)
        If True, includes diagonal neighbors (8-way connectivity)
        If False, only uses orthogonal neighbors (4-way connectivity)
    color : int, optional (default=None)
        If specified, only consider cells == color (ignoring zero or other color cells)
        If None, consider all non-zero cells as part of connectivity
    
    Returns:
    --------
    list[set]
        A list of objects, each object being a set of (row, col) tuples
    """
    rows, cols = matrix.shape
    visited = set()

    # Define connectivity based on diagonal_connectivity parameter
    orthogonal = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    diagonal = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    directions = orthogonal + diagonal if diagonal_connectivity else orthogonal

    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols

    objects = []

    for r in range(rows):
        for c in range(cols):
            val = matrix[r, c]
            # Decide if this cell is "relevant" for connectivity
            if color is None:
                # We consider non-zero cells
                if val == 0:
                    continue
            else:
                # We only consider cells matching the specified color
                if val != color:
                    continue

            if (r, c) not in visited:
                # We found a new object; run BFS/DFS to collect it
                stack = [(r, c)]
                visited.add((r, c))
                comp = [(r, c)]

                while stack:
                    rr, cc = stack.pop()
                    # Explore neighbors
                    for dr, dc in directions:
                        nr, nc = rr + dr, cc + dc
                        if in_bounds(nr, nc) and (nr, nc) not in visited:
                            # Check if it has the same color criterion
                            if color is None:
                                # same => matrix[nr, nc] != 0
                                if matrix[nr, nc] != 0:
                                    stack.append((nr, nc))
                                    visited.add((nr, nc))
                                    comp.append((nr, nc))
                            else:
                                # must match exactly the color
                                if matrix[nr, nc] == color:
                                    stack.append((nr, nc))
                                    visited.add((nr, nc))
                                    comp.append((nr, nc))
                objects.append(set(comp))

    return objects

def overlap(coordsA, coordsB) -> bool:
    return not coordsA.isdisjoint(coordsB)

def adjacent(coordsA, coordsB) -> bool:
    # True if any pair from coordsA and coordsB has Manhattan-distance=1
    for (rA, cA) in coordsA:
        for (rB, cB) in coordsB:
            if abs(rA - rB) + abs(cA - cB) == 1:
                return True
    return False

def flood_fill(matrix: np.ndarray, start_pos: tuple, new_value: int):
    """
    Flood fills a 2D numpy array starting from start_pos with new_value. It fills the 
    starting position and all adjacent cells having the same value as the the cell at 
    the starting position.
    
    Parameters:
    array: 2D numpy array
    start_pos: tuple of (row, col) starting position
    new_value: value to fill with 
    """
    if not (0 <= start_pos[0] < matrix.shape[0] and 0 <= start_pos[1] < matrix.shape[1]):
        return matrix
    
    # Get the original value at the start position
    old_value = matrix[start_pos]
    
    # If the start position is already the new value, return
    if old_value == new_value:
        return matrix
    
    # Create a queue for flood fill
    queue = [start_pos]
    matrix[start_pos] = new_value
    
    while queue:
        row, col = queue.pop(0)
        
        # Check all 4 neighboring positions
        for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
            r, c = row + dr, col + dc
            
            # Check if position is within bounds and has the old value
            if (0 <= r < matrix.shape[0] and 
                0 <= c < matrix.shape[1] and 
                matrix[r,c] == old_value):
                
                matrix[r,c] = new_value
                queue.append((r,c))
    
    return matrix
