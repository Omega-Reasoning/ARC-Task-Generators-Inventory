from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObject, GridObjects
from Framework.input_library import create_object, retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class taskh9woyuLPvSxM3pCCxQZKZCGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['grid_size']} × {vars['grid_size']}.",
            "Each input grid contains a {color('frame_colour')} square frame with a size of at least 3×3 or larger.",
            "The exterior of the frame is completely empty (0).",
            "The interior is either completely filled with the {color('interior_colour')} color or contains a checkerboard pattern, where empty cells and {color('interior_colour')} cells alternate across rows and columns.",
            "The size of the frame varies across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the {color('frame_colour')} frame and its interior.",
            "If the interior is completely filled with {color('interior_colour')} cells, then change all interior {color('interior_colour')} cells to {color('new_interior_colour')}.",
            "Otherwise, leave the interior unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate random colors (all different)
        frame_colour, interior_colour, new_interior_colour = random.sample(range(1, 10), 3)

        # Random grid size between 5 and 30
        grid_size = random.randint(5, 30)

        taskvars = {
            'grid_size': grid_size,
            'frame_colour': frame_colour,
            'interior_colour': interior_colour,
            'new_interior_colour': new_interior_colour
        }

        # 4–5 training examples
        num_train = random.randint(4, 5)
        train_examples: List[GridPair] = []
        test_examples: List[GridPair] = []

        used_frame_sizes: set = set()

        filled_count = 0
        checkerboard_count = 0

        for _ in range(num_train):
            if filled_count < 2:
                fill_type = 'filled'
            elif checkerboard_count < 2:
                fill_type = 'checkerboard'
            else:
                fill_type = random.choice(['filled', 'checkerboard'])

            if fill_type == 'filled':
                filled_count += 1
            else:
                checkerboard_count += 1

            gridvars = {'fill_type': fill_type, 'used_frame_sizes': used_frame_sizes}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})

        # Test example (doesn't force uniqueness but tries)
        test_fill_type = random.choice(['filled', 'checkerboard'])
        test_gridvars = {'fill_type': test_fill_type, 'used_frame_sizes': used_frame_sizes}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_examples.append({'input': test_input, 'output': test_output})

        return taskvars, {'train': train_examples, 'test': test_examples}

    def _sample_frame_size(self, grid_size: int, used_frame_sizes: set) -> int:
        """
        Pick a valid frame size:
          - minimum 3 (so interior can exist)
          - maximum min(grid_size - 2, 15) (leave 1-cell margin)
          - prefer sizes not used yet; fall back if necessary
        """
        min_frame_size = 3
        max_frame_size = min(grid_size - 2, 15)
        # Guard to ensure max >= min
        if max_frame_size < min_frame_size:
            # This can only happen if grid_size < 5, but grid_size ∈ [5, 30]; still guard:
            max_frame_size = min_frame_size

        candidates = [s for s in range(min_frame_size, max_frame_size + 1) if s not in used_frame_sizes]
        if not candidates:
            candidates = list(range(min_frame_size, max_frame_size + 1))
        return random.choice(candidates)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        frame_colour = taskvars['frame_colour']
        interior_colour = taskvars['interior_colour']
        fill_type = gridvars['fill_type']
        used_frame_sizes: set = gridvars.get('used_frame_sizes', set())

        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Safe frame size selection (prevents empty randrange)
        frame_size = self._sample_frame_size(grid_size, used_frame_sizes)
        used_frame_sizes.add(frame_size)

        # Position frame so it fits completely
        max_start_row = grid_size - frame_size
        max_start_col = grid_size - frame_size
        start_row = random.randint(0, max_start_row)
        start_col = random.randint(0, max_start_col)

        end_row = start_row + frame_size
        end_col = start_col + frame_size

        # Draw frame border
        grid[start_row, start_col:end_col] = frame_colour
        grid[end_row - 1, start_col:end_col] = frame_colour
        grid[start_row:end_row, start_col] = frame_colour
        grid[start_row:end_row, end_col - 1] = frame_colour

        # Interior area
        interior_start_row = start_row + 1
        interior_end_row = end_row - 1
        interior_start_col = start_col + 1
        interior_end_col = end_col - 1

        if interior_end_row > interior_start_row and interior_end_col > interior_start_col:
            if fill_type == 'filled':
                grid[interior_start_row:interior_end_row, interior_start_col:interior_end_col] = interior_colour
            else:  # checkerboard
                for r in range(interior_start_row, interior_end_row):
                    for c in range(interior_start_col, interior_end_col):
                        if (r + c) % 2 == 0:
                            grid[r, c] = interior_colour

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        frame_colour = taskvars['frame_colour']
        interior_colour = taskvars['interior_colour']
        new_interior_colour = taskvars['new_interior_colour']

        output = grid.copy()

        frame_cells = np.where(grid == frame_colour)
        if len(frame_cells[0]) == 0:
            return output

        min_row, max_row = frame_cells[0].min(), frame_cells[0].max()
        min_col, max_col = frame_cells[1].min(), frame_cells[1].max()

        interior_start_row = min_row + 1
        interior_end_row = max_row
        interior_start_col = min_col + 1
        interior_end_col = max_col

        if (interior_start_row < interior_end_row and
            interior_start_col < interior_end_col):

            interior_region = grid[interior_start_row:interior_end_row,
                                   interior_start_col:interior_end_col]

            # Completely filled iff no zeros and all cells == interior_colour
            flat = interior_region.flatten()
            if flat.size > 0 and np.all(flat == interior_colour):
                output[interior_start_row:interior_end_row,
                       interior_start_col:interior_end_col] = new_interior_colour

        return output
