"""
Transformation Library

This module provides a collection of utility functions for transforming ARC matrices. 
They can be used in both the create_input() and transform_input() methods of the
ARCTaskGenerator class. Only frequently used methods, which are difficult for AI 
systems to create from scratch without errors are included here.
"""

import numpy as np

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
