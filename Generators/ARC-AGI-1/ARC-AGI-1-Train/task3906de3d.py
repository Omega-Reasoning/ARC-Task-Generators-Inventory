from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple
from input_library import create_object, retry
from transformation_library import find_connected_objects, GridObject, GridObjects

class Task3906de3dGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain a single {color('object_color1')} rectangular block, with some vertical strips removed from the interior columns, leaving the first and last columns unchanged.",
            "The {color('object_color1')} rectangular block has four rows and a variable number of columns, always touching the top edge of the grid.",
            "Sometimes, a single {color('object_color1')} cell is added directly below the columns that do not have removed vertical strips.",
            "The bottom of the grid contains {color('object_color2')} vertical strips, appearing only in the columns where vertical strips were removed from the rectangular block."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grid and moving all {color('object_color2')} vertical strips upward so that each {color('object_color2')} strip is vertically connected to a {color('object_color1')} cell."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        taskvars = {
            'rows': random.randint(10, 20),
            'cols': random.randint(8, 15),
            'object_color1': random.randint(1, 9),
            'object_color2': random.randint(1, 9)
        }
        
        while taskvars['object_color1'] == taskvars['object_color2']:
            taskvars['object_color2'] = random.randint(1, 9)
        
        num_train_examples = random.randint(3, 5)
        train_examples = []
        added_cell_below = False

        for i in range(num_train_examples):
            block_width = random.randint(4, taskvars['cols'] - 1)
            add_cell_below = (not added_cell_below and (i == num_train_examples - 2 or i == num_train_examples - 1))
            if add_cell_below:
                added_cell_below = True
            
            gridvars = {
                'block_width': block_width, 
                'add_cell_below': add_cell_below
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        test_gridvars = {
            'block_width': random.randint(4, taskvars['cols'] - 1),
            'add_cell_below': random.choice([True, False])
        }
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        color1 = taskvars['object_color1']
        color2 = taskvars['object_color2']
        add_cell_below = gridvars.get('add_cell_below', False)

        block_width = gridvars['block_width']
        grid = np.zeros((rows, cols), dtype=int)

        # Ensure the first column remains empty
        start_col = random.randint(1, cols - block_width)

        # Create the initial block
        grid[0:4, start_col:start_col + block_width] = color1

        # Select possible columns for removal, skipping the first column
        possible_columns = list(range(start_col + 1, start_col + block_width - 1))
        random.shuffle(possible_columns)

        # Select columns to remove, ensuring unique strip lengths
        columns_to_remove = []
        idx = 0
        while idx < len(possible_columns):
            columns_to_remove.append(possible_columns[idx])
            idx += random.randint(2, 3)  # Skip 2 or 3 indices at a time

        # Generate unique removal lengths
        available_lengths = list(range(1, 4))  # Possible unique strip lengths (1, 2, 3)
        random.shuffle(available_lengths)

        strip_lengths = {}
        for i, col in enumerate(columns_to_remove):
            strip_lengths[col] = available_lengths[i % len(available_lengths)]

        # Apply the removal of top strips
        for col, length in strip_lengths.items():
            grid[4-length:4, col] = 0

        # Generate unique bottom strip lengths
        available_bottom_lengths = list(range(1, rows - 5))  # Ensure uniqueness
        random.shuffle(available_bottom_lengths)

        bottom_strip_lengths = {}
        for i, col in enumerate(columns_to_remove):
            bottom_strip_lengths[col] = available_bottom_lengths[i % len(available_bottom_lengths)]

        # Apply the bottom strips
        for col, length in bottom_strip_lengths.items():
            grid[rows - length:rows, col] = color2

        # Optionally add color1 cells below the block
        if add_cell_below:
            intact_columns = [c for c in range(start_col + 1, start_col + block_width - 1) if c not in columns_to_remove]
            if intact_columns:
                num_to_add = min(2, len(intact_columns))
                selected_cols = random.sample(intact_columns, num_to_add)
                for col in selected_cols:
                    grid[4, col] = color1

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        
        color2 = taskvars['object_color2']
        color2_strips = objects.with_color(color2)
        color1 = taskvars['object_color1']
        color1_objects = objects.with_color(color1)
        
        for strip in color2_strips:
            cells = list(strip.cells)
            for r, c, _ in cells:
                output_grid[r, c] = 0
            
            strip_cols = set(c for _, c, _ in cells)
            for col in strip_cols:
                color1_rows = [r for r, c, _ in color1_objects[0].cells if c == col]
                if color1_rows:
                    max_color1_row = max(color1_rows)
                    strip_height = len([1 for r, c, _ in cells if c == col])
                    for r_offset in range(strip_height):
                        new_row = max_color1_row + 1 + r_offset
                        if new_row < grid.shape[0]:
                            output_grid[new_row, col] = color2
        
        return output_grid

