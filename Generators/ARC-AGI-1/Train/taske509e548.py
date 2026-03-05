from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObject, GridObjects
from Framework.input_library import retry, random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List, Set

class Taske509e548Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each grid contains a random number of objects, all in the color {color('object_color')}.",
            "There are three distinct shapes: L-shape: A right-angle structure made of two connected straight segments of varying lengths. U-shape: Formed by two parallel bars connected by a base. The dimensions of the base and bars may vary, and one bar may be longer than the other. Extended-U shape: A variation of the U-shape where one of the parallel bars extends beyond the base to the opposite side.",
            "The shapes can appear in different orientations.",
            "The thickness of all shapes is 1 cell.",
            "The number of occurrences of each shape is random but each shape appears at least once in each grid.",
            "No two shapes are adjacent—there is at least one cell of space between any two shapes."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "All shapes in the input grid are identified and classified into one of the following three types: L-shape: A right-angle structure made of two connected straight segments of varying lengths. U-shape: Formed by two parallel bars connected by a base. The dimensions of the base and bars may vary, and one bar may be longer than the other. Extended-U shape: A variation of the U-shape where one of the parallel bars extends beyond the base to the opposite side.",
            "The L shapes are colored with {color('color_1')}.",
            "The U shapes are colored with {color('color_2')}.",
            "The extended-U shapes are colored with {color('color_3')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        taskvars = {}
        taskvars['object_color'] = random.randint(1, 9)
        
        # Ensure all colors are different
        colors = list(range(1, 10))
        colors.remove(taskvars['object_color'])
        random.shuffle(colors)
        taskvars['color_1'] = colors[0]  # L-shape color
        taskvars['color_2'] = colors[1]  # U-shape color  
        taskvars['color_3'] = colors[2]  # Extended-U color

        # Generate train and test grids
        num_train = random.randint(3, 6)
        train_data = []
        test_data = []
        
        for i in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})
        
        # Create test grid
        input_grid = self.create_input(taskvars, {})
        output_grid = self.transform_input(input_grid, taskvars)
        test_data.append({'input': input_grid, 'output': output_grid})
        
        return taskvars, {'train': train_data, 'test': test_data}

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        def generate_valid_grid():
            # Random grid size
            height = random.randint(10, 20)
            width = random.randint(10, 20)
            grid = np.zeros((height, width), dtype=int)
            
            # Generate shapes ensuring each type appears at least once
            shapes_to_place = ['L', 'U', 'E']  # L, U, Extended-U
            
            # Add additional random shapes
            num_additional = random.randint(1, 4)
            for _ in range(num_additional):
                shapes_to_place.append(random.choice(['L', 'U', 'E']))
            
            random.shuffle(shapes_to_place)
            
            placed_shapes = []
            for shape_type in shapes_to_place:
                attempts = 0
                max_attempts = 100
                shape = None
                
                while attempts < max_attempts:
                    if shape_type == 'L':
                        shape = self._create_l_shape(taskvars['object_color'])
                    elif shape_type == 'U':
                        shape = self._create_u_shape(taskvars['object_color'])
                    else:  # Extended-U
                        shape = self._create_extended_u_shape(taskvars['object_color'])
                    
                    # Random rotation
                    rotations = random.randint(0, 3)
                    shape = np.rot90(shape, k=rotations)
                    
                    # Try to place shape
                    if self._try_place_shape(grid, shape, placed_shapes):
                        break
                    attempts += 1
                
                if attempts >= max_attempts:
                    return None  # Failed to place shape
            
            return grid
        
        return retry(generate_valid_grid, lambda x: x is not None)

    def _create_l_shape(self, color: int) -> np.ndarray:
        """Create an L-shape with random dimensions"""
        # Random arm lengths (at least 2 cells each)
        arm1_len = random.randint(2, 4)
        arm2_len = random.randint(2, 4)
        
        # Create L-shape (corner at bottom-left)
        shape = np.zeros((arm1_len, arm2_len), dtype=int)
        
        # Fill the L: bottom row and left column
        shape[-1, :] = color  # horizontal arm (bottom)
        shape[:, 0] = color   # vertical arm (left)
        
        return shape

    def _create_u_shape(self, color: int) -> np.ndarray:
        """Create a U-shape with random dimensions - bars can have different lengths"""
        # Random dimensions
        width = random.randint(3, 5)
        left_height = random.randint(2, 5)   # left bar height
        right_height = random.randint(2, 5)  # right bar height (can be different)
        max_height = max(left_height, right_height)
        
        shape = np.zeros((max_height, width), dtype=int)
        
        # Fill the U with potentially different bar heights
        # Base is at the bottom
        base_row = max_height - 1
        
        # Left bar - extends from base upward
        shape[base_row - left_height + 1:base_row + 1, 0] = color
        
        # Right bar - extends from base upward  
        shape[base_row - right_height + 1:base_row + 1, -1] = color
        
        # Base connecting the bars
        shape[base_row, :] = color
        
        return shape

    def _create_extended_u_shape(self, color: int) -> np.ndarray:
        """Create an extended U-shape where one bar extends beyond the base"""
        # Random dimensions
        width = random.randint(3, 5)
        base_height = random.randint(2, 3)  # height from base to shorter bar end
        extension = random.randint(1, 2)    # how much the other bar extends beyond base
        
        # Decide which bar extends and which bar is shorter
        left_extends = random.choice([True, False])
        
        if left_extends:
            # Left bar extends beyond base, right bar is shorter
            left_total_height = base_height + extension
            right_height = random.randint(2, base_height)  # right bar can vary too
            total_height = left_total_height
            base_row = total_height - base_height
        else:
            # Right bar extends beyond base, left bar is shorter
            right_total_height = base_height + extension
            left_height = random.randint(2, base_height)   # left bar can vary too
            total_height = right_total_height
            base_row = total_height - base_height
        
        shape = np.zeros((total_height, width), dtype=int)
        
        if left_extends:
            # Left bar (full height)
            shape[:, 0] = color
            # Right bar (shorter, connects to base)
            shape[base_row:base_row + right_height, -1] = color
            # Base
            shape[base_row, :] = color
        else:
            # Left bar (shorter, connects to base)
            shape[base_row:base_row + left_height, 0] = color
            # Right bar (full height)
            shape[:, -1] = color
            # Base
            shape[base_row, :] = color
        
        return shape

    def _try_place_shape(self, grid: np.ndarray, shape: np.ndarray, placed_shapes: List) -> bool:
        """Try to place a shape on the grid ensuring no adjacency"""
        shape_h, shape_w = shape.shape
        grid_h, grid_w = grid.shape
        
        # Try random positions
        for _ in range(100):
            start_r = random.randint(0, max(0, grid_h - shape_h))
            start_c = random.randint(0, max(0, grid_w - shape_w))
            
            # Check if placement is valid (no overlap and no adjacency)
            if self._can_place_shape(grid, shape, start_r, start_c):
                # Place the shape
                for r in range(shape_h):
                    for c in range(shape_w):
                        if shape[r, c] != 0:
                            grid[start_r + r, start_c + c] = shape[r, c]
                
                placed_shapes.append((start_r, start_c, shape))
                return True
        
        return False

    def _can_place_shape(self, grid: np.ndarray, shape: np.ndarray, start_r: int, start_c: int) -> bool:
        """Check if a shape can be placed without overlapping or being adjacent to existing shapes"""
        shape_h, shape_w = shape.shape
        grid_h, grid_w = grid.shape
        
        # Check bounds
        if start_r + shape_h > grid_h or start_c + shape_w > grid_w:
            return False
        
        # Check for overlap and adjacency (including diagonal)
        for r in range(max(0, start_r - 1), min(grid_h, start_r + shape_h + 1)):
            for c in range(max(0, start_c - 1), min(grid_w, start_c + shape_w + 1)):
                if grid[r, c] != 0:
                    # Check if this existing cell is adjacent to any shape cell
                    for sr in range(shape_h):
                        for sc in range(shape_w):
                            if shape[sr, sc] != 0:
                                shape_r, shape_c = start_r + sr, start_c + sc
                                if abs(r - shape_r) <= 1 and abs(c - shape_c) <= 1:
                                    return False
        
        return True

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output = grid.copy()

        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)

        for obj in objects:
            coords = list(obj.coords)
            coord_set = set(coords)

            # Count orthogonal neighbors for each cell
            neighbor_counts = {}
            for r, c in coords:
                count = 0
                for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                    if (r+dr, c+dc) in coord_set:
                        count += 1
                neighbor_counts[(r,c)] = count

            # Count endpoints (degree 1) and junctions (degree 3)
            deg1 = sum(1 for v in neighbor_counts.values() if v == 1)
            deg3 = sum(1 for v in neighbor_counts.values() if v == 3)

            # ---- Classification ----
            if deg1 == 2 and deg3 == 1:
                shape_type = 'L'
            else:
                # Use bounding box structure to distinguish U vs Extended-U
                rows = [r for r,c in coords]
                cols = [c for r,c in coords]

                min_r, max_r = min(rows), max(rows)
                min_c, max_c = min(cols), max(cols)

                height = max_r - min_r + 1
                width = max_c - min_c + 1

                # Count cells in extreme rows/cols
                top_count = sum(1 for r,c in coords if r == min_r)
                bottom_count = sum(1 for r,c in coords if r == max_r)
                left_count = sum(1 for r,c in coords if c == min_c)
                right_count = sum(1 for r,c in coords if c == max_c)

                # Simple heuristic:
                if top_count >= 2 and bottom_count >= 2:
                    shape_type = 'U'
                elif left_count >= 2 and right_count >= 2:
                    shape_type = 'U'
                else:
                    shape_type = 'E'

            # ---- Coloring ----
            if shape_type == 'L':
                new_color = taskvars['color_1']
            elif shape_type == 'U':
                new_color = taskvars['color_2']
            else:
                new_color = taskvars['color_3']

            for r, c, _ in obj.cells:
                output[r, c] = new_color

        return output

    

    
    def _can_trace_to_corner(self, coords_set: Set[Tuple[int, int]], 
                            start: Tuple[int, int], 
                            corner: Tuple[int, int]) -> bool:
        """Check if we can trace from start to corner along straight line"""
        r1, c1 = start
        r2, c2 = corner
        
        # Determine direction
        if r1 == r2:  # Same row - horizontal line
            step = 1 if c2 > c1 else -1
            for c in range(c1, c2 + step, step):
                if (r1, c) not in coords_set:
                    return False
        elif c1 == c2:  # Same column - vertical line
            step = 1 if r2 > r1 else -1
            for r in range(r1, r2 + step, step):
                if (r, c1) not in coords_set:
                    return False
        else:
            # Not in same row or column - not a straight line
            return False
        
        return True
    

    
    def _check_horizontal_base_u(self, coords: List[Tuple[int, int]]) -> str:
        """Check for U-shape with horizontal base"""
        rows = sorted(set(r for r, c in coords))
        
        # Try each row as potential base
        for base_row in rows:
            base_cells = [(r, c) for r, c in coords if r == base_row]
            if len(base_cells) < 3:  # Base must span at least 3 cells
                continue
                
            # Check if base is continuous
            base_cols = sorted([c for r, c in base_cells])
            if len(base_cols) != base_cols[-1] - base_cols[0] + 1:
                continue  # Not continuous
            
            # Find vertical bars at the ends of the base
            left_col = base_cols[0]
            right_col = base_cols[-1]
            
            left_bar = [(r, c) for r, c in coords if c == left_col]
            right_bar = [(r, c) for r, c in coords if c == right_col]
            
            if len(left_bar) < 2 or len(right_bar) < 2:
                continue
                
            # Check if bars are vertical and connected to base
            left_rows = sorted([r for r, c in left_bar])
            right_rows = sorted([r for r, c in right_bar])
            
            # Both bars must be continuous
            if (len(left_rows) != left_rows[-1] - left_rows[0] + 1 or
                len(right_rows) != right_rows[-1] - right_rows[0] + 1):
                continue
                
            # Base must be connected to both bars
            if base_row not in left_rows or base_row not in right_rows:
                continue
                
            # Check if this accounts for the shape (allowing for different bar lengths)
            expected_cells = len(left_bar) + len(right_bar) + len(base_cols) - 2  # -2 for overlap
            if expected_cells != len(coords):
                continue
                
            # Now check if it's regular U or Extended-U
            # For Extended-U: one bar should extend beyond the base on the opposite side
            # For regular U: bars may be different lengths but don't extend to opposite side
            
            # Check if either bar extends beyond the base to the opposite side
            left_extends_beyond_base = any(r != base_row for r in left_rows)
            right_extends_beyond_base = any(r != base_row for r in right_rows)
            
            if not (left_extends_beyond_base and right_extends_beyond_base):
                continue  # Both bars must extend beyond base for U/Extended-U
            
            # Check for extension to opposite side of base
            left_has_opposite_extension = any(
                (r < base_row and any(rr > base_row for rr in right_rows)) or
                (r > base_row and any(rr < base_row for rr in right_rows))
                for r in left_rows if r != base_row
            )
            
            right_has_opposite_extension = any(
                (r < base_row and any(rr > base_row for rr in left_rows)) or  
                (r > base_row and any(rr < base_row for rr in left_rows))
                for r in right_rows if r != base_row
            )
            
            # Extended-U: one bar extends to the opposite side of where the other bar extends
            if left_has_opposite_extension or right_has_opposite_extension:
                return 'E'
            else:
                return 'U'
                
        return 'unknown'
    
