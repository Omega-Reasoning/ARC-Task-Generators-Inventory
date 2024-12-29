"""
Transformation Library

This module provides a collection of utility functions for transforming ARC grids. 
They can be used in both the create_input() and transform_input() methods of the
ARCTaskGenerator class.
"""

from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Callable, Iterator, Optional, List, Set, Tuple, TypeVar
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
        # Create set of (row, col, color) tuples
        cells = set((r, c, grid[r, c]) for r, c in zip(coords[0], coords[1]))
        
        if monochromatic:
            # Group cells by color and create separate objects
            by_color = {}
            for r, c, color in cells:
                by_color.setdefault(color, set()).add((r, c, color))
            objects.extend(GridObject(color_cells) for color_cells in by_color.values())
        else:
            # Create single object with all cells regardless of color
            objects.append(GridObject(cells))
    
    return GridObjects(objects)
