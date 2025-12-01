from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List


class Task78176bbGenerator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['grid_size']} × {vars['grid_size']}.",
            "Each grid contains exactly one diagonal line parallel to the main diagonal (top-left to bottom-right).",
            "The diagonal line may be offset from the main diagonal by a few cells, either above or below it.",
            "One or two {color('cell_color')} triangular regions may appear, either above the diagonal, below it, or both.",
            "To form a triangle below the diagonal, choose n consecutive diagonal cells, then from the last chosen cell add n–1 {color('cell_color')} cells to the left, extend upwards toward the diagonal, and fill the enclosed area.",
            "To form a triangle above the diagonal, choose n consecutive diagonal cells, then from the first chosen cell add n–1 {color('cell_color')} cells to the right, extend downwards toward the diagonal, and fill the enclosed area.",
            "The triangular regions occupy the area between the diagonal and their constructed boundaries.",
            "The original diagonal line remains unchanged."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "For each {color('cell_color')} triangular region in the input, one new diagonal line is added in the output.",
            "If there is one triangular region, one new diagonal line is added; if there are two triangular regions, two new diagonal lines are added.",
            "Each added diagonal line has the same orientation as the original diagonal line.",
            "The position of each new diagonal line is determined by placing it directly above the triangular region if the region is above the diagonal, or directly below the triangular region if the region is below the diagonal.",
            "The line begins from the diagonal path immediately after the triangular region ends and continues in the same diagonal direction.",
            "After adding the new diagonal lines, all {color('cell_color')} triangular regions are removed, leaving only diagonal lines in the output."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    # --------- HELPER: triangle construction (fixed “above” case) ---------

    def _add_triangle_to_grid(
        self,
        grid: np.ndarray,
        grid_size: int,
        diag_offset: int,
        cell_color: int,
        side: str,
        n: int = None,
    ) -> bool:
        """
        Add a triangular region strictly attached to the diagonal.

        - Only uses rows that actually contain a diagonal cell.
        - Ensures a 1-cell margin from all borders for triangle cells.
        - side: 'above' (right side of diagonal) or 'below' (left side of diagonal).

        Returns:
            True if a triangle was placed, False otherwise.
        """
        # All rows that have a diagonal cell inside the grid
        diag_rows = [r for r in range(grid_size) if 0 <= r + diag_offset < grid_size]

        # Keep only rows that are not on the outer border (1-cell margin for rows)
        diag_rows = [r for r in diag_rows if 0 < r < grid_size - 1]
        if len(diag_rows) < 2:
            # Not enough rows to form a triangle of height >= 2
            return False

        max_height_by_rows = len(diag_rows)
        max_height_by_grid = grid_size - 2  # keep margin top/bottom
        max_possible_height = min(max_height_by_rows, max_height_by_grid)

        if max_possible_height < 2:
            return False

        attempts = 50
        while attempts > 0:
            attempts -= 1

            # Choose triangle height n
            if n is None:
                height = random.randint(2, max_possible_height)
            else:
                height = max(2, min(n, max_possible_height))

            max_start_idx = len(diag_rows) - height
            if max_start_idx < 0:
                # Not enough diagonal rows for this height, try again with another height
                n = None
                continue

            start_idx = random.randint(0, max_start_idx)
            rows = [diag_rows[start_idx + j] for j in range(height)]

            valid = True
            # Check that for all these rows the triangle would stay within 1-cell margin
            for j, r in enumerate(rows):
                c = r + diag_offset  # diagonal column for this row
                if not (0 <= c < grid_size):
                    valid = False
                    break
                # ensure diagonal itself is not on the extreme left/right border
                if not (0 < c < grid_size - 1):
                    valid = False
                    break

                if side == 'below':
                    start_c = c - (j + 1)
                    end_c = c - 1
                else:  # 'above'
                    start_c = c + 1
                    width = height - j              # width shrinks towards the diagonal
                    end_c = c + width               # inclusive

                if not (1 <= start_c <= end_c <= grid_size - 2):
                    valid = False
                    break

            if not valid:
                # Try different height / start index
                n = None
                continue

            # All checks passed -> actually fill the triangle
            for j, r in enumerate(rows):
                c = r + diag_offset
                if side == 'below':
                    for col in range(c - j - 1, c):
                        grid[r, col] = cell_color
                else:  # 'above'
                    width = height - j
                    # inclusive end, so +1 for Python range
                    for col in range(c + 1, c + width + 1):
                        grid[r, col] = cell_color

            return True  # success

        # If we exit the loop, no triangle was placed; this should be very rare.
        return False

    # -----------------------------------------------------------------------

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """Create an input grid with a main diagonal and 1–2 triangular regions."""
        grid_size = taskvars['grid_size']
        cell_color = taskvars['cell_color']
        diag_color = gridvars['diag_color']
        diag_offset = gridvars['diag_offset']

        # start with empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # draw main diagonal with offset
        for i in range(grid_size):
            col = i + diag_offset
            if 0 <= col < grid_size:
                grid[i, col] = diag_color

        # number of triangles: 1 or 2
        num_triangles = random.choice([1, 2])

        # when two triangles, place one above and one below to maximise solvability
        if num_triangles == 2:
            sides = ['above', 'below']
        else:
            sides = [random.choice(['above', 'below'])]

        for side in sides:
            self._add_triangle_to_grid(
                grid=grid,
                grid_size=grid_size,
                diag_offset=diag_offset,
                cell_color=cell_color,
                side=side,
                n=None,  # let the helper choose a valid height
            )

        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output = grid.copy()
        cell_color = taskvars['cell_color']
        grid_size = taskvars['grid_size']
        
        # Find the diagonal by looking for non-zero, non-cell_color cells
        # and determine its offset
        diagonal_cells = []
        for r in range(grid_size):
            for c in range(grid_size):
                if grid[r, c] != 0 and grid[r, c] != cell_color:
                    diagonal_cells.append((r, c))
        
        if not diagonal_cells:
            return output
        
        # Determine diagonal offset (col - row for each diagonal cell should be constant)
        diag_color = grid[diagonal_cells[0][0], diagonal_cells[0][1]]
        diag_offset = diagonal_cells[0][1] - diagonal_cells[0][0]
        
        # Find triangular regions
        triangular_regions = []
        for r in range(grid_size):
            for c in range(grid_size):
                if grid[r, c] == cell_color:
                    triangular_regions.append((r, c))
        
        if not triangular_regions:
            return output
        
        # Determine which triangles exist (above or below diagonal)
        upper_triangle_cells = []
        lower_triangle_cells = []
        
        for r, c in triangular_regions:
            expected_diag_col = r + diag_offset
            if c > expected_diag_col:  # Above diagonal (to the right)
                upper_triangle_cells.append((r, c))
            elif c < expected_diag_col:  # Below diagonal (to the left)
                lower_triangle_cells.append((r, c))
        
        # Remove all triangular regions
        for r, c in triangular_regions:
            output[r, c] = 0
        
        # Add new diagonal lines
        # For upper triangle: add diagonal parallel above it (shifted right)
        if upper_triangle_cells:
            # Find the rightmost column of the upper triangle
            max_col = max(c for r, c in upper_triangle_cells)
            min_row_at_max_col = min(r for r, c in upper_triangle_cells if c == max_col)
            # Calculate how much further right the new diagonal should be
            extra_offset = max_col - (min_row_at_max_col + diag_offset) + 1
            new_offset = diag_offset + extra_offset
            for i in range(grid_size):
                new_c = i + new_offset
                if 0 <= new_c < grid_size:
                    output[i, new_c] = diag_color
        
        # For lower triangle: add diagonal parallel below it (shifted left)
        if lower_triangle_cells:
            # Find the leftmost column of the lower triangle
            min_col = min(c for r, c in lower_triangle_cells)
            max_row_at_min_col = max(r for r, c in lower_triangle_cells if c == min_col)
            # Calculate how much further left the new diagonal should be
            extra_offset = (max_row_at_min_col + diag_offset) - min_col + 1
            new_offset = diag_offset - extra_offset
            for i in range(grid_size):
                new_c = i + new_offset
                if 0 <= new_c < grid_size:
                    output[i, new_c] = diag_color
        
        return output
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        grid_size = random.randint(8, 15)
        cell_color = random.randint(1, 9)
        
        taskvars = {
            'grid_size': grid_size,
            'cell_color': cell_color
        }
        
        # Create training examples with varying diagonal colors
        num_train = random.randint(3, 5)
        train_pairs = []
        
        # Track whether we have examples with 1 and 2 triangles
        has_one_triangle = False
        has_two_triangles = False
        
        for i in range(num_train):
            # Each training example gets a different diagonal color and offset
            diag_color = random.choice([c for c in range(1, 10) if c != cell_color])
            diag_offset = random.choice([-3, -2, -1, 0, 1, 2, 3])
            
            gridvars = {
                'diag_color': diag_color,
                'diag_offset': diag_offset
            }
            
            # Ensure we have at least one example with 1 triangle and one with 2 triangles
            if i == 0 and not has_two_triangles:
                # Force first example to have 2 triangles (one above, one below)
                input_grid = self._create_input_with_num_triangles(taskvars, gridvars, 2)
                has_two_triangles = True
            elif i == 1 and not has_one_triangle:
                # Force second example to have 1 triangle (randomly above or below)
                input_grid = self._create_input_with_num_triangles(taskvars, gridvars, 1)
                has_one_triangle = True
            else:
                # Random for remaining examples
                input_grid = self.create_input(taskvars, gridvars)
            
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_pairs.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example with its own diagonal color and offset
        test_diag_color = random.choice([c for c in range(1, 10) if c != cell_color])
        test_diag_offset = random.choice([-3, -2, -1, 0, 1, 2, 3])
        test_gridvars = {
            'diag_color': test_diag_color,
            'diag_offset': test_diag_offset
        }
        
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_pairs = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_pairs,
            'test': test_pairs
        }
    
    def _create_input_with_num_triangles(
        self,
        taskvars: Dict[str, Any], 
        gridvars: Dict[str, Any], 
        num_triangles: int
    ) -> np.ndarray:
        """Create input grid with specified number of triangles (1 or 2)."""
        grid_size = taskvars['grid_size']
        cell_color = taskvars['cell_color']
        diag_color = gridvars['diag_color']
        diag_offset = gridvars['diag_offset']

        # start with empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # draw main diagonal with offset
        for i in range(grid_size):
            col = i + diag_offset
            if 0 <= col < grid_size:
                grid[i, col] = diag_color

        # determine sides based on number of triangles
        if num_triangles == 2:
            sides = ['above', 'below']
        else:
            sides = [random.choice(['above', 'below'])]

        for side in sides:
            # Let the helper pick a valid height and placement
            self._add_triangle_to_grid(
                grid=grid,
                grid_size=grid_size,
                diag_offset=diag_offset,
                cell_color=cell_color,
                side=side,
                n=None
            )

        return grid


