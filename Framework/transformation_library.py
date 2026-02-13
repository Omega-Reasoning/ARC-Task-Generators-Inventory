"""
Transformation Library

This module provides a collection of utility functions for transforming ARC grids. 
They can be used in both the create_input() and transform_input() methods of the
ARCTaskGenerator class.
"""

from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Callable, Iterator, Optional, List, Set, Tuple, TypeVar, Union
import numpy as np
from scipy.ndimage import label
from scipy.spatial.distance import cdist

from utilities import visualize_grid, visualize_object

T = TypeVar('T')

def make_collection_method(method: Callable) -> Callable:
    """
    Decorator that converts a boolean method taking a single GridObject
    into a method that works with GridObjects and returns filtered GridObjects
    """
    @wraps(method)
    def wrapper(self, others: 'GridObjects') -> 'GridObjects':
        if isinstance(others, GridObject):
            # If called with single object, maintain original behavior
            return method(self, others)
        # Filter objects where the method returns True
        matching = [obj for obj in others.objects if method(self, obj)]
        return GridObjects(matching)
    return wrapper

class BorderBehavior(Enum):
    CLIP = "clip"       # Remove parts that go out of bounds
    WRAP = "wrap"       # Wrap around to other side of grid
    STOP = "stop"       # Cancel movement if any part would go out
    BOUNCE = "bounce"   # Reverse direction at grid boundary

class CollisionBehavior(Enum):
    IGNORE = "ignore"   # Move through other objects
    STOP = "stop"       # Cancel movement if hitting object
    BOUNCE = "bounce"   # Reverse direction when hitting object

@dataclass
class GridObject:
    """A collection of cells in a grid which is meaningful with respect to an ARC task."""
    cells: Set[Tuple[int, int, int]]  # Set of (row, col, color) coordinates
    
    @classmethod
    def from_grid(cls, grid: np.ndarray, coords: Set[Tuple[int, int]]) -> 'GridObject':
        """Create a GridObject from grid coordinates."""
        return cls({(r, c, grid[r, c]) for r, c in coords})

    @classmethod
    def from_array(cls, array: np.ndarray, offset: Tuple[int, int] = (0, 0)) -> 'GridObject':
        """Create a GridObject from an array at a specific position."""
        r_off, c_off = offset
        return cls({(r + r_off, c + c_off, val) 
                   for r, row in enumerate(array) 
                   for c, val in enumerate(row) if val != 0})

    def copy(self) -> 'GridObject':
        return GridObject(cells=self.cells.copy())

    def cut(self, grid, background=0):
        """
        Remove this object's cells from the grid by setting them to background.
        
        Args:
            grid: The target grid to modify
            background: The background color value to use (default 0)
        
        Returns:
            self for fluent chaining
        """
        for (r, c, _) in self.cells:  # We don't need the color for cutting
            grid[r, c] = background
        return self

    def paste(self, grid, overwrite=True, background=0):
        """
        Paste this object's cells onto the grid.
        
        Args:
            grid: The target grid to modify
            overwrite: If True, overwrites existing values; if False, only writes to background cells
            background: The background color value (default 0)
        
        Returns:
            self for fluent chaining
        """
        for (r, c, col) in self.cells:
            if overwrite or grid[r, c] == background:
                grid[r, c] = col
        return self

    def __len__(self) -> int:
        """Number of cells in the object."""
        return len(self.cells)
    
    def __iter__(self) -> Iterator[Tuple[int, int, int]]:
        """Make object iterable over its cells."""
        return iter(self.cells)
    
    def __eq__(self, other: 'GridObject') -> bool:
        """Check if two objects have the same cells and colors."""
        return isinstance(other, GridObject) and self.cells == other.cells

    @property
    def coords(self) -> Set[Tuple[int, int]]:
        """Get the (row, col) coordinates of cells without colors."""
        return {(r, c) for r, c, _ in self.cells}

    @property
    def colors(self) -> Set[int]:
        """Get unique colors in this object."""
        return {color for _, _, color in self.cells}
    
    @property
    def bounding_box(self) -> Tuple[slice, slice]:
        """Get row/col slices defining the bounding box."""
        if not self.cells:
            return (slice(0, 0), slice(0, 0))
        rows, cols, _ = zip(*self.cells)
        return (slice(min(rows), max(rows) + 1),
                slice(min(cols), max(cols) + 1))
    
    @property
    def height(self) -> int:
        """Height of object's bounding box."""
        box = self.bounding_box
        return box[0].stop - box[0].start
    
    @property
    def width(self) -> int:
        """Width of object's bounding box."""
        box = self.bounding_box
        return box[1].stop - box[1].start
    
    def to_array(self) -> np.ndarray:
        """Extract object as minimal grid array."""
        box = self.bounding_box
        array = np.zeros((self.height, self.width), dtype=int)
        for r, c, color in self.cells:
            array[r - box[0].start, c - box[1].start] = color
        return array

    @property
    def size(self) -> int:
        """Number of cells in object."""
        return len(self.cells)

    @property
    def is_monochromatic(self) -> bool:
        """Check if object has only one color."""
        return len(self.colors) == 1
    
    def has_color(self, color: int) -> bool:
        """Check if object contains specific color."""
        return color in self.colors

    def touches(self, other: 'GridObject', diag: bool = False) -> bool:
        """
        Check if this object is adjacent to another object.
        
        Args:
            other: The other GridObject to check adjacency with
            diagonal_connectivity: If True, includes diagonal neighbors (8-way connectivity)
                                If False, only cardinal neighbors (4-way connectivity)
        """
        coords1 = self.coords
        coords2 = other.coords
        
        neighbors = [(0,1), (0,-1), (1,0), (-1,0)]  # cardinal directions
        if diag:
            neighbors.extend([(1,1), (1,-1), (-1,1), (-1,-1)])  # diagonal directions
        
        return any((r+dr, c+dc) in coords2 
                for r, c in coords1
                for dr, dc in neighbors)

    @make_collection_method
    def overlaps_with(self, other: 'GridObject') -> bool:
        return not self.coords.isdisjoint(other.coords)

    @make_collection_method
    def fully_contains(self, other: 'GridObject') -> bool:
        return other.coords.issubset(self.coords)

    @make_collection_method
    def is_strictly_above(self, other: 'GridObject') -> bool:
        my_box = self.bounding_box
        other_box = other.bounding_box
        return my_box[0].stop <= other_box[0].start

    @make_collection_method
    def is_strictly_below(self, other: 'GridObject') -> bool:
        my_box = self.bounding_box
        other_box = other.bounding_box
        return my_box[0].start >= other_box[0].stop

    @make_collection_method
    def is_strictly_left_of(self, other: 'GridObject') -> bool:
        my_box = self.bounding_box
        other_box = other.bounding_box
        return my_box[1].stop <= other_box[1].start

    @make_collection_method
    def is_strictly_right_of(self, other: 'GridObject') -> bool:
        my_box = self.bounding_box
        other_box = other.bounding_box
        return my_box[1].start >= other_box[1].stop

    def manhattan_distance(self, other: 'GridObject') -> int:
        """Compute minimum Manhattan (L1) distance between this and another object."""
        return int(cdist(list(self.coords), list(other.coords), metric='cityblock').min())

    def chebyshev_distance(self, other: 'GridObject') -> int:
        """Compute minimum Chebyshev (L∞) distance between this and another object (allows diagonal moves)."""
        return int(cdist(list(self.coords), list(other.coords), metric='chebyshev').min())

    def color_all(self, color: int) -> 'GridObject':
        """Colors all cells with given color value. Returns self."""
        self.cells = {(r, c, color) for r, c, _ in self.cells}
        return self

    def translate(self, 
                dx: int, 
                dy: int,
                border_behavior: BorderBehavior = BorderBehavior.CLIP,
                grid_shape: Optional[Tuple[int, int]] = None) -> 'GridObject':
        """Translate object by (dx, dy) with specified behaviors.
        
        Args:
            dx: Translation amount in rows (positive = down)
            dy: Translation amount in columns (positive = right)
            border_behavior: How to handle matrix boundaries
            grid_shape: Optional (rows, cols) tuple defining grid boundaries
            
        Returns:
            self for method chaining
        """
        new_cells = set()
        would_hit_border = False
        
        for r, c, color in self.cells:
            new_r, new_c = r + dx, c + dy
            if grid_shape:
                rows, cols = grid_shape
                if not (0 <= new_r < rows and 0 <= new_c < cols):
                    would_hit_border = True
                    if border_behavior == BorderBehavior.WRAP:
                        new_r = new_r % rows
                        new_c = new_c % cols
                    elif border_behavior == BorderBehavior.CLIP:
                        continue
                    elif border_behavior in (BorderBehavior.STOP, BorderBehavior.BOUNCE):
                        return self  # Return unchanged object for STOP/BOUNCE
                new_cells.add((new_r, new_c, color))
            else:
                new_cells.add((new_r, new_c, color))
        
        if would_hit_border and border_behavior == BorderBehavior.BOUNCE:
            # For bounce, reverse the movement
            self.translate(-dx, -dy, border_behavior, grid_shape)
            return self
        
        self.cells = new_cells
        return self

    # TODO: needs to be extended to also cover other types of rotation around a reference point
    def rotate(self, rotations: int) -> 'GridObject':
        """Rotate object counterclockwise around its center.
        
        Args:
            rotations: Number of 90-degree counterclockwise rotations
        
        Returns:
            self for method chaining
        """
        rotated = np.rot90(self.to_array(), k=rotations)
        self.cells = self.from_array(rotated, 
                                offset=(self.bounding_box[0].start,
                                        self.bounding_box[1].start)).cells
        return self

    def extend(self,
            grid: np.ndarray,
            direction: Tuple[int, int],
            start_cells: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
            stop_predicate: Optional[Callable[[np.ndarray, int, int], bool]] = None,
            border_behavior: BorderBehavior = BorderBehavior.STOP,
            background: int = 0) -> 'GridObject':
        """
        Extends from specified cells or object cells in given direction until stop predicate is met.
        
        Args:
            grid: The grid to check for extensions
            direction: (dy, dx) tuple specifying direction of extension
            start_cells: Optional specific cell(s) to start extension from. Can be:
                - Single tuple (row, col)
                - List of tuples [(row1, col1), (row2, col2), ...]
                If None, uses all cells in the current GridObject
            stop_predicate: Optional function(grid, row, col) -> bool that returns True 
                when extension should stop. If None, stops at any non-background color.
            border_behavior: How to handle grid boundaries (WRAP/STOP/BOUNCE)
            background: Background color to extend through (used only if stop_predicate is None)
        
        Returns:
            New GridObject containing the extension cells
        """
        rows, cols = grid.shape
        dy, dx = direction
        new_cells = set()
        
        # Get initial positions
        if isinstance(start_cells, tuple):
            initial_positions = [(start_cells[0], start_cells[1], grid[start_cells])]
        elif start_cells is not None:
            initial_positions = [(r, c, grid[r, c]) for r, c in start_cells]
        else:
            initial_positions = self.cells
        
        if stop_predicate is None:
            stop_predicate = lambda g, r, c: g[r, c] != background
        
        def get_next_position(r: int, c: int, dm: int) -> Tuple[int, int]:
            """Calculate next position based on direction and multiplier"""
            return r + dy * dm, c + dx * dm
        
        def handle_border(r: int, c: int, base_r: int, base_c: int, dm: int) -> Tuple[int, int, int]:
            """Handle border collision and return new position and direction multiplier"""
            if not (0 <= r < rows and 0 <= c < cols):
                if border_behavior == BorderBehavior.STOP:
                    return r, c, 0  # multiplier 0 signals stop
                elif border_behavior == BorderBehavior.WRAP:
                    return r % rows, c % cols, dm
                else:  # BOUNCE
                    dm *= -1
                    r, c = get_next_position(base_r, base_c, dm)
                    if not (0 <= r < rows and 0 <= c < cols):
                        return r, c, 0
                    return r, c, dm
            return r, c, dm
        
        for base_r, base_c, color in initial_positions:
            curr_r, curr_c = get_next_position(base_r, base_c, 1)
            dir_mult = 1
            
            while dir_mult != 0:  # dir_mult becomes 0 to signal stopping
                curr_r, curr_c, dir_mult = handle_border(curr_r, curr_c, base_r, base_c, dir_mult)
                if dir_mult == 0 or stop_predicate(grid, curr_r, curr_c):
                    break
                    
                new_cells.add((curr_r, curr_c, color))
                curr_r, curr_c = get_next_position(curr_r, curr_c, dir_mult)
        
        return GridObject(new_cells)

class GridObjects:
    """Collection of GridObjects with filtering capabilities."""
    def __init__(self, objects: Optional[List[GridObject]] = None):
        self.objects = objects if objects is not None else []

    # Basic collection interface
    def __len__(self) -> int: return len(self.objects)
    def __iter__(self) -> Iterator[GridObject]: return iter(self.objects)
    def __getitem__(self, idx) -> GridObject: return self.objects[idx]
    
    # Filtering methods
    def filter(self, predicate) -> 'GridObjects':
        """Filter objects using a custom predicate."""
        return GridObjects([obj for obj in self.objects if predicate(obj)])
    
    def with_color(self, color: int) -> 'GridObjects':
        """Filter objects containing specific color."""
        return self.filter(lambda obj: color in obj.colors)
    
    def with_size(self, min_size: Optional[int] = None, 
                 max_size: Optional[int] = None) -> 'GridObjects':
        """Filter objects by size range."""
        return self.filter(lambda obj: 
            (min_size is None or len(obj) >= min_size) and 
            (max_size is None or len(obj) <= max_size))
    
    # Sorting methods
    def sort_by_size(self, reverse: bool = False) -> 'GridObjects':
        """Sort objects by size."""
        return GridObjects(sorted(self.objects, key=len, reverse=reverse))
    
    def sort_by_position(self, top_to_bottom: bool = True) -> 'GridObjects':
        """Sort objects by position."""
        key = lambda obj: min(r if top_to_bottom else c for r, c, _ in obj.cells)
        return GridObjects(sorted(self.objects, key=key))

    def __getattr__(self, method_name):
        """Delegates method calls to contained objects with chaining support."""
        def method(*args, **kwargs):
            return GridObjects([getattr(obj, method_name)(*args, **kwargs) 
                            for obj in self.objects])
        return method

def find_connected_objects(grid: np.ndarray,
                         diagonal_connectivity: bool = False,
                         background: int = 0,
                         monochromatic: bool = True) -> GridObjects:
    """Find all connected objects in the grid.
    
    Args:
        grid: 2D numpy array where non-background values represent objects
        diagonal_connectivity: If True, includes diagonal neighbors (8-way connectivity)
                             If False, only cardinal neighbors (4-way connectivity)
        background: Value representing empty space (default 0)
        monochromatic: If True, split objects by color. If False, allow
                      multiple colors in one object.
    
    Returns:
        GridObjects containing all connected non-background objects
    """
    structure = np.ones((3, 3)) if diagonal_connectivity else np.array([[0,1,0],[1,1,1],[0,1,0]])
    mask = grid != background
    labeled_array, num_features = label(mask, structure=structure)
    
    objects = []
    for label_id in range(1, num_features + 1):
        coords = np.where(labeled_array == label_id)
        cells = set((r, c, grid[r, c]) for r, c in zip(coords[0], coords[1]))
        
        if monochromatic:
            by_color = {}
            for r, c, color in cells:
                by_color.setdefault(color, set()).add((r, c, color))
            objects.extend(GridObject(color_cells) for color_cells in by_color.values())
        else:
            objects.append(GridObject(cells))
    
    return GridObjects(objects)

def parse_objects_by_color(grid: np.ndarray, background: int = 0) -> 'GridObjects':
    """Parse grid into separate GridObjects, one for each unique color (except background).
    
    Args:
        grid: 2D numpy array where non-background values represent objects
        background: Value representing empty space (default 0)
    
    Returns:
        GridObjects containing one object per unique color in the grid
    """
    # Find all unique colors except background
    colors = set(np.unique(grid)) - {background}
    
    objects = []
    for color in colors:
        coords = np.where(grid == color)
        cells = {(r, c, color) for r, c in zip(coords[0], coords[1])}
        objects.append(GridObject(cells))
    
    return GridObjects(objects)

def get_objects_from_raster(grid: np.ndarray,
                        subgrid_rows: int,
                        subgrid_cols: int,
                        has_delimiters: bool = True,
                        initial_rows: int = 0,
                        initial_cols: int = 0
                        ) -> List[List[GridObject]]:
    """
    Extract GridObjects from a raster grid where objects are separated by regular spacing
    and optional delimiters.
    
    Args:
        grid: Input grid to process
        subgrid_rows: Height of each complete subgrid
        subgrid_cols: Width of each complete subgrid
        has_delimiters: Whether delimiter rows/columns exist between objects
        initial_rows: Height of first row subgrid (0 if first row is delimiter)
        initial_cols: Width of first column subgrid (0 if first column is delimiter)
    
    Returns:
        2D array of GridObjects containing the extracted subgrids
    """
    def get_starts_and_sizes(total_size: int, subgrid_size: int, initial_size: int) -> tuple:
        starts, sizes = [], []
        current = 0 if initial_size > 0 else delimiter
        
        if initial_size > 0:
            starts.append(current)
            sizes.append(initial_size)
            current += initial_size + delimiter
            
        while current + subgrid_size <= total_size:
            starts.append(current)
            sizes.append(subgrid_size)
            current += subgrid_size + delimiter
            
        if current < total_size:
            starts.append(current)
            sizes.append(1)
            
        return starts, sizes

    delimiter = int(has_delimiters)
    row_starts, row_sizes = get_starts_and_sizes(grid.shape[0], subgrid_rows, initial_rows)
    col_starts, col_sizes = get_starts_and_sizes(grid.shape[1], subgrid_cols, initial_cols)
    
    return [[GridObject({(rs + r, cs + c, grid[rs + r, cs + c])
                        for r in range(rsz)
                        for c in range(csz)})
             for cs, csz in zip(col_starts, col_sizes)]
            for rs, rsz in zip(row_starts, row_sizes)]

def get_objects_from_raster_old(grid: np.ndarray,
                           subgrid_rows: int,
                           subgrid_cols: int,
                           has_delimiters: bool = True,
                           row_offset: int = 0,
                           col_offset: int = 0) -> List[List[GridObject]]:
    """
    Extract GridObjects from a raster grid where objects are separated by regular spacing
    and optional delimiters.
    
    Args:
        grid: Input grid to process
        subgrid_rows: Height of each subgrid
        subgrid_cols: Width of each subgrid
        has_delimiters: Whether delimiter rows/columns exist between objects
        row_offset: Row index where first object ends (default 0)
        col_offset: Column index where first object ends (default 0)
    
    Returns:
        2D array of GridObjects containing the extracted subgrids
    """
    nrows, ncols = grid.shape
    delimiter = int(has_delimiters)
    
    def get_starts_and_sizes(total_size: int, subgrid_size: int, offset: int) -> tuple:
        starts = []
        sizes = []
        current = 0
        
        # Skip initial offset if present
        if offset > 0:
            current = offset
            
        while current + subgrid_size <= total_size:
            starts.append(current)
            sizes.append(subgrid_size)
            current += subgrid_size + delimiter
            
        return starts, sizes

    row_starts, row_sizes = get_starts_and_sizes(nrows, subgrid_rows, row_offset)
    col_starts, col_sizes = get_starts_and_sizes(ncols, subgrid_cols, col_offset)
    
    return [[GridObject.from_array(
        array=grid[rs:rs + rsz, cs:cs + csz],
        offset=(rs, cs)
    ) for cs, csz in zip(col_starts, col_sizes)]
      for rs, rsz in zip(row_starts, row_sizes)]


