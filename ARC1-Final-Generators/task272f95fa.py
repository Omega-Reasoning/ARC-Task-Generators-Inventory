from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Task272f95faGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['cols']}.",
            "The grid is a raster with {color('raster_color')} rows and columns as separators."
            "The {color('raster_color')} rows and columns are separated by at least one cell, which divides the input grid into nine subgrids, each of these subgrids have different sizes.",
            "The remaining cells of the input grid are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "First copy the input grid to the output grid.",
            "Identify the {color('raster_color')} rows and columns positions in the input grid.",
            "Identify the subgrids to be colored.",
            "The second, fourth, fifth, sixth, and eighth subgrids are colored with five different colors respectively: {color('one')}, {color('two')}, {color('three')}, {color('four')}, and {color('five')}.",
            "The remaining cells remain empty(0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        cols = taskvars['cols']
        raster_color = taskvars['raster_color']
        
        grid = np.zeros((rows, cols), dtype=int)
        
        # First position can be anywhere from 1 to rows-3 (leaving room for second position)
        first_row = random.randint(1, rows//2)
        # Second position must be at least 2 cells after first position
        second_row = random.randint(first_row + 2, rows - 2)
        row_positions = [first_row, second_row]

        # Similar logic for columns
        first_col = random.randint(1, cols//2)
        second_col = random.randint(first_col + 2, cols -2)
        col_positions = [first_col, second_col]
        
        for row in row_positions:
            grid[row, :] = raster_color
        for col in col_positions:
            grid[:, col] = raster_color
        
        return grid
    
    def transform_input(self, grid, taskvars):
        output_grid = grid.copy()
        colors = [taskvars['one'], taskvars['two'], taskvars['three'], taskvars['four'], taskvars['five']]

        # Find the raster color (most common non-zero value)
        non_zero_values = grid[grid != 0]
        raster_color = np.bincount(non_zero_values)[1:].argmax() + 1
        
        # Find row positions where raster lines exist
        row_positions = np.where(np.all(grid == raster_color, axis=1))[0]
        # Find column positions where raster lines exist
        col_positions = np.where(np.all(grid == raster_color, axis=0))[0]
        
        subgrid_coords = [
            (0, row_positions[0], 0, col_positions[0]),
            (0, row_positions[0], col_positions[0] + 1, col_positions[1]),
            (0, row_positions[0], col_positions[1] + 1, grid.shape[1]),
            (row_positions[0] + 1, row_positions[1], 0, col_positions[0]),
            (row_positions[0] + 1, row_positions[1], col_positions[0] + 1, col_positions[1]),
            (row_positions[0] + 1, row_positions[1], col_positions[1] + 1, grid.shape[1]),
            (row_positions[1] + 1, grid.shape[0], 0, col_positions[0]),
            (row_positions[1] + 1, grid.shape[0], col_positions[0] + 1, col_positions[1]),
            (row_positions[1] + 1, grid.shape[0], col_positions[1] + 1, grid.shape[1])
        ]
        
        subgrid_indices = [1, 3, 4, 5, 7]  # Corresponding indices to be colored
        
        for idx, color in zip(subgrid_indices, colors):
            r1, r2, c1, c2 = subgrid_coords[idx]
            output_grid[r1:r2, c1:c2] = color
        
        return output_grid
    
    def create_grids(self):
        rows = random.randint(10, 30)
        cols = random.randint(10, 30)
        raster_color = random.randint(1, 9)
        
        available_colors = list(set(range(1, 10)) - {raster_color})
        random.shuffle(available_colors)
        
        taskvars = {
            'rows': rows,
            'cols': cols,
            'raster_color': raster_color,
            'one': available_colors[0],
            'two': available_colors[1],
            'three': available_colors[2],
            'four': available_colors[3],
            'five': available_colors[4]
        }
        
        num_train = random.randint(2, 3)
        num_test = 1
        
        train_data = [{'input': self.create_input(taskvars, {}), 'output': None} for _ in range(num_train)]
        test_data = [{'input': self.create_input(taskvars, {}), 'output': None} for _ in range(num_test)]
        
        for example in train_data + test_data:
            example['output'] = self.transform_input(example['input'], taskvars)
        
        return taskvars, {'train': train_data, 'test': test_data}

