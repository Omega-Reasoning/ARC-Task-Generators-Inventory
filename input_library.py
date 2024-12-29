"""
Input Matrix Generation Library

This module provides a collection of utility functions for creating ARC matrices. 
They can be used in the create_input() of the ARCTaskGenerator class, but not in
the transform_input() method. In other words, those functions are not relevant
for solving ARC tasks at runtime.
"""

import numpy as np
import random
from scipy.ndimage import label

from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar, Union

class Contiguity(Enum):
    NONE = 0    # No contiguity enforcement
    FOUR = 4    # 4-way connectivity
    EIGHT = 8   # 8-way connectivity

T = TypeVar('T')  # Type variable for the generator's return type

def retry(generator: Callable[[], T], 
         predicate: Callable[[T], bool],
         max_attempts: int = 100) -> T:
    """
    Repeatedly calls generator until the predicate returns True.
    
    Args:
        generator: Function that generates a value
        predicate: Function that takes the generated value and returns True if valid
        max_attempts: Maximum number of retry attempts
    
    Returns:
        Generated value that satisfies the predicate
    """    
    for attempt in range(max_attempts):
        result = generator()
        if predicate(result):
            return result
            
    raise ValueError(
        f"Failed to satisfy predicate after {max_attempts} attempts"
    )

def create_object(height: int, width: int,
                color_palette: Union[int, List[int]],
                contiguity: Optional[Contiguity] = Contiguity.EIGHT,
                background: int = 0) -> np.ndarray:
    """
    Creates a matrix containing a randomly generated object with specified properties.
    
    Args:
        height: Height of the matrix
        width: Width of the matrix
        color_palette: Single color or list of colors to use for the object
        contiguity: Type of connectivity enforcement (NONE, FOUR, or EIGHT)
        background: Color value for empty cells (must not be in color_palette)
        
    Returns:
        np.ndarray: Matrix containing the generated object
        
    Raises:
        ValueError: If background color is in the color palette
    """
    if isinstance(color_palette, int):
        color_palette = [color_palette]
    
    if background in color_palette:
        raise ValueError("Background color must not be in color palette")

    object_matrix = np.full((height, width), background, dtype=int)
    
    # Randomly fill cells
    for r in range(height):
        for c in range(width):
            if random.choice([True, False]):
                object_matrix[r, c] = random.choice(color_palette)

    # Ensure at least one colored cell if none were placed
    if np.all(object_matrix == background):
        r = random.randrange(height)
        c = random.randrange(width)
        object_matrix[r, c] = random.choice(color_palette)

    # Enforce contiguity if specified by keeping the largest contiguous component
    if contiguity != Contiguity.NONE:
        structure = (np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int) 
                    if contiguity == Contiguity.FOUR 
                    else np.ones((3, 3), dtype=int))
        
        labeled, n_obj = label(object_matrix != background, structure=structure)
        if n_obj > 1:
            sizes = [(labeled == i).sum() for i in range(1, n_obj + 1)]
            largest_idx = 1 + np.argmax(sizes)
            object_matrix[(labeled != largest_idx) & (object_matrix != background)] = background

    return object_matrix

def enforce_object_width(object_generator: Callable[[], np.ndarray]) -> np.ndarray:
    """
    Takes an object generator and ensures the resulting matrix has at least
    one non-zero element in each row.
    
    Args:
        object_generator: Function that generates a matrix
        
    Returns:
        np.ndarray: A matrix that satisfies the width constraint
        
    Raises:
        ValueError: If constraint cannot be satisfied after maximum attempts
    """
    return retry(
        object_generator,
        lambda x: np.all(np.any(x != 0, axis=1))
    )

def enforce_object_height(object_generator: Callable[[], np.ndarray]) -> np.ndarray:
    """
    Takes an object generator and ensures the resulting matrix has at least
    one non-zero element in each column.
    
    Args:
        object_generator: Function that generates a matrix
        
    Returns:
        np.ndarray: A matrix that satisfies the height constraint
        
    Raises:
        ValueError: If constraint cannot be satisfied after maximum attempts
    """
    return retry(
        object_generator,
        lambda x: np.all(np.any(x != 0, axis=0))
    )

def random_cell_coloring(grid: np.ndarray,
                        color_palette: Union[int, List[int]],
                        density: float = 0.5,
                        background: int = 0,
                        overwrite: bool = False) -> np.ndarray:
    """
    Randomly colors cells in a matrix using colors from the specified palette.
    
    Args:
        grid: 2D numpy array to be colored
        color_palette: Single color or list of colors to use
        density: Fraction of colorable cells to fill (between 0 and 1)
        background: Color value for empty cells
        overwrite: Whether to allow coloring non-background cells
        
    Returns:
        Grid with randomly colored cells (modifies input)
    """
    if isinstance(color_palette, int):
        color_palette = [color_palette]
    
    # Get indices of cells we can color
    colorable = np.where(grid == background) if not overwrite else np.where(np.ones_like(grid))
    n_cells = len(colorable[0])
    n_to_color = int(density * n_cells)
    
    # Randomly select cells to color
    if n_to_color > 0:
        indices = np.random.choice(n_cells, n_to_color, replace=False)
        rows = colorable[0][indices]
        cols = colorable[1][indices]
        grid[rows, cols] = np.random.choice(color_palette, n_to_color)
    
    return grid
