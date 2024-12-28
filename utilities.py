from typing import List, Optional, Set, Tuple
import numpy as np

# 8-bit color codes for terminal output
color_codes = {
    0: 0,    # black for empty
    1: 27,   # blue
    2: 196,  # red
    3: 46,   # green
    4: 226,  # yellow
    5: 244,  # grey
    6: 213,  # pink
    7: 208,  # orange
    8: 45,   # cyan
    9: 88    # maroon
}

def visualize_matrix(matrix: np.ndarray) -> str:
    """
    Creates a colored string visualization of a matrix.
    
    Args:
        matrix: Input matrix to visualize
    Returns:
        String representation with ANSI color codes
    """    
    rows = []
    for i in range(matrix.shape[0]):
        row = []
        for j in range(matrix.shape[1]):
            val = matrix[i,j]
            color_code = color_codes.get(val, 0)
            row.append(f"\033[38;5;{color_code}m{val}\033[0m")
        rows.append(" ".join(row))
    
    return "\n".join(rows)

def visualize_object(grid: np.ndarray, obj: Set[Tuple[int, int]]) -> str:
    """
    Creates a visualization of a single object within its matrix context.
    
    Args:
        matrix: Reference matrix containing the object
        obj: Set of (row, col) coordinates defining the object
    Returns:
        String representation with ANSI color codes
    """    
    grid = np.full(grid.shape, ".")
    for i, j in obj:
        grid[i,j] = str(grid[i,j])
    
    rows = []
    for i in range(grid.shape[0]):
        row = []
        for j in range(grid.shape[1]):
            val = grid[i,j]
            if val == ".":
                row.append(val)
            else:
                color_code = color_codes.get(int(val), 0)
                row.append(f"\033[38;5;{color_code}m{val}\033[0m")
        rows.append(" ".join(row))
    
    return "\n".join(rows)

def visualize_objects(grid: np.ndarray, objects: List[Set[Tuple[int, int]]]) -> str:
    """
    Creates a visualization of multiple objects.
    
    Args:
        matrix: Reference matrix containing the objects
        objects: List of objects to visualize
    Returns:
        String representation with each object shown separately
    """
    visualizations = []
    for idx, obj in enumerate(objects, 1):
        visualizations.extend([
            f"Object {idx}:",
            visualize_object(grid, obj),
            ""
        ])
    return "\n".join(visualizations)

def visualize_object(obj, grid_shape: Optional[Tuple[int, int]] = None) -> str:
    """Creates a visualization of a single object.
    
    Args:
        obj: GridObject to visualize
        grid_shape: Optional (rows, cols) tuple for full grid visualization.
                   If provided, object will be shown in context of full grid size.
                   If None, shows only the minimal bounding box of the object.
    Returns:
        String representation with ANSI color codes
    """    
    if not obj.cells:
        return ""
    
    if grid_shape is None:
        # Use minimal bounding box
        rows = [r for r, _, _ in obj.cells]
        cols = [c for _, c, _ in obj.cells]
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        offset_row, offset_col = min_row, min_col
    else:
        # Use provided grid shape
        height, width = grid_shape
        offset_row, offset_col = 0, 0
    
    grid = np.full((height, width), ".", dtype=str)
    
    for r, c, color in obj.cells:
        adj_r = r - offset_row
        adj_c = c - offset_col
        if 0 <= adj_r < height and 0 <= adj_c < width:  # Ensure within bounds
            grid[adj_r, adj_c] = str(color)
    
    rows = []
    for i in range(height):
        row = []
        for j in range(width):
            val = grid[i,j]
            if val == ".":
                row.append(val)
            else:
                color_code = color_codes.get(int(val), 0)
                row.append(f"\033[38;5;{color_code}m{val}\033[0m")
        rows.append(" ".join(row))
    
    return "\n".join(rows)
