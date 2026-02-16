from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import create_object
from Framework.transformation_library import find_connected_objects

class TaskYDtuXDKYpd7y7oVjdFqpoeGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "Each input grid contains a {color('object_color')} rectangular object surrounded by empty (0) cells.",
            "The {color('object_color')} object is positioned within the interior of the grid, excluding the outermost borders."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are created by copying the input grid and adding a one-cell-wide {color('frame_color')} frame around the {color('object_color')} rectangular object."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict, used_positions: set) -> np.ndarray:
        rows, cols = taskvars['rows'], taskvars['cols']
        object_color = taskvars['object_color']

        grid = np.zeros((rows, cols), dtype=int)

        # Random size for the object, ensuring it is at least 2x2
        obj_height = random.randint(2, rows - 4)
        obj_width = random.randint(2, cols - 4)

        # Ensure the object's position varies
        max_attempts = 100  # Avoid infinite loops in edge cases
        for _ in range(max_attempts):
            start_row = random.randint(1, rows - obj_height - 1)
            start_col = random.randint(1, cols - obj_width - 1)
            if (start_row, start_col) not in used_positions:
                used_positions.add((start_row, start_col))
                break
        else:
            # If no new position is found (edge case), fallback to any valid position
            start_row, start_col = list(used_positions)[0]

        # Place object in grid
        grid[start_row:start_row + obj_height, start_col:start_col + obj_width] = object_color

        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        frame_color = taskvars['frame_color']
        output_grid = grid.copy()
        
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        if not objects:
            return output_grid
        
        obj = objects[0]  # We assume only one object is placed
        row_slice, col_slice = obj.bounding_box
        
        row_start, row_end = row_slice.start, row_slice.stop
        col_start, col_end = col_slice.start, col_slice.stop
        
        # Add frame around object
        output_grid[row_start - 1: row_end + 1, col_start - 1] = frame_color
        output_grid[row_start - 1: row_end + 1, col_end] = frame_color
        output_grid[row_start - 1, col_start - 1: col_end + 1] = frame_color
        output_grid[row_end, col_start - 1: col_end + 1] = frame_color
        
        return output_grid
    
    def create_grids(self) -> tuple:
        taskvars = {
            'rows': random.randint(8, 30),
            'cols': random.randint(8, 30),
            'object_color': random.randint(1, 9),
            'frame_color': random.randint(1, 9)
        }

        while taskvars['frame_color'] == taskvars['object_color']:
            taskvars['frame_color'] = random.randint(1, 9)

        used_positions = set()  # Track used object positions

        train_data = [
            {'input': (inp := self.create_input(taskvars, {}, used_positions)), 'output': self.transform_input(inp, taskvars)}
            for _ in range(random.randint(3, 4))
        ]

        test_data = [
            {'input': (inp := self.create_input(taskvars, {}, used_positions)), 'output': self.transform_input(inp, taskvars)}
            for _ in range(2)
        ]

        return taskvars, {'train': train_data, 'test': test_data}

