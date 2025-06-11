from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import retry, create_object, random_cell_coloring
import numpy as np
import random

class Task0e671a1aGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['rows']}.",
            "Each grid contains exactly three colored cells — {color('color1')}, {color('color2')}, and {color('color3')} — placed such that each one is located in a different quadrant of the grid.",
            "All remaining cells are empty (0), and the three colored cells are completely separated from one another.",
            "Ensure no two cells share the same row or column."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying the {color('color1')}, {color('color2')}, and {color('color3')} cells.",
            "Create a path connecting {color('color1')} to {color('color2')}, and then from {color('color2')} to {color('color3')}, using the following rules.",
            "First, from {color('color1')} to {color('color2')}: Move horizontally (left or right) until reaching the column of {color('color2')}, then move vertically (up or down) to reach its exact position.",
            "Next, from {color('color2')} to {color('color3')}: Move horizontally (left or right) until reaching the column of {color('color3')}, then move vertically (up or down) to reach the exact cell of {color('color3')}.",
            "The paths should be drawn using {color('path_color')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        # Generate task variables
        taskvars = {
            'rows': random.randint(10, 30),
            'color1': random.randint(1, 9),
            'color2': random.randint(1, 9),
            'color3': random.randint(1, 9),
            'path_color': random.randint(1, 9)
        }
        
        # Ensure all colors are different
        while taskvars['color2'] == taskvars['color1']:
            taskvars['color2'] = random.randint(1, 9)
        while taskvars['color3'] in [taskvars['color1'], taskvars['color2']]:
            taskvars['color3'] = random.randint(1, 9)
        while taskvars['path_color'] in [taskvars['color1'], taskvars['color2'], taskvars['color3']]:
            taskvars['path_color'] = random.randint(1, 9)
        
        # Generate training and test examples
        num_train = random.randint(3, 6)
        train_examples = []
        test_examples = []
        
        for i in range(num_train + 1):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            
            example = {'input': input_grid, 'output': output_grid}
            if i < num_train:
                train_examples.append(example)
            else:
                test_examples.append(example)
        
        train_test_data = {'train': train_examples, 'test': test_examples}
        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        grid = np.zeros((rows, rows), dtype=int)
        
        def generate_valid_positions():
            # Define the four quadrants
            mid_r = rows // 2
            mid_c = rows // 2
            
            quadrants = [
                (0, mid_r, 0, mid_c),           # top-left quadrant
                (0, mid_r, mid_c, rows),        # top-right quadrant  
                (mid_r, rows, 0, mid_c),        # bottom-left quadrant
                (mid_r, rows, mid_c, rows)      # bottom-right quadrant
            ]
            
            # Randomly select 3 different quadrants
            selected_quadrants = random.sample(quadrants, 3)
            
            # Generate one position in each selected quadrant
            positions = []
            for r_min, r_max, c_min, c_max in selected_quadrants:
                # Add some margin from edges to ensure separation
                margin = 1
                r_start = max(r_min, r_min + margin)
                r_end = min(r_max - 1, r_max - 1 - margin)
                c_start = max(c_min, c_min + margin)
                c_end = min(c_max - 1, c_max - 1 - margin)
                
                # Ensure we have valid ranges
                if r_start > r_end:
                    r_start = r_end = (r_min + r_max - 1) // 2
                if c_start > c_end:
                    c_start = c_end = (c_min + c_max - 1) // 2
                
                r = random.randint(r_start, r_end)
                c = random.randint(c_start, c_end)
                positions.append((r, c))
            
            return positions
        
        def positions_valid(positions):
            # Check that no two positions share row or column
            rows_used = set()
            cols_used = set()
            for r, c in positions:
                if r in rows_used or c in cols_used:
                    return False
                rows_used.add(r)
                cols_used.add(c)
            return True
        
        # Generate valid positions
        positions = retry(generate_valid_positions, positions_valid)
        
        # Place the three colored cells
        colors = [taskvars['color1'], taskvars['color2'], taskvars['color3']]
        for (r, c), color in zip(positions, colors):
            grid[r, c] = color
            
        return grid

    def transform_input(self, grid, taskvars):
        output_grid = grid.copy()
        
        # Find positions of the three colored cells
        color1_pos = None
        color2_pos = None
        color3_pos = None
        
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == taskvars['color1']:
                    color1_pos = (r, c)
                elif grid[r, c] == taskvars['color2']:
                    color2_pos = (r, c)
                elif grid[r, c] == taskvars['color3']:
                    color3_pos = (r, c)
        
        path_color = taskvars['path_color']
        
        # Draw path from color1 to color2
        if color1_pos and color2_pos:
            r1, c1 = color1_pos
            r2, c2 = color2_pos
            
            # Move horizontally first (including the target column)
            if c1 < c2:  # move right
                for c in range(c1 + 1, c2 + 1):  # Fixed: include c2
                    output_grid[r1, c] = path_color
            elif c1 > c2:  # move left
                for c in range(c2, c1):  # Fixed: include c2
                    output_grid[r1, c] = path_color
            
            # Then move vertically (not including the starting row since we already drew there)
            if r1 < r2:  # move down
                for r in range(r1 + 1, r2):  # Don't include r2 since that's the target cell
                    output_grid[r, c2] = path_color
            elif r1 > r2:  # move up
                for r in range(r2 + 1, r1):  # Don't include r2 since that's the target cell
                    output_grid[r, c2] = path_color
        
        # Draw path from color2 to color3
        if color2_pos and color3_pos:
            r2, c2 = color2_pos
            r3, c3 = color3_pos
            
            # Move horizontally first (including the target column)
            if c2 < c3:  # move right
                for c in range(c2 + 1, c3 + 1):  # Fixed: include c3
                    output_grid[r2, c] = path_color
            elif c2 > c3:  # move left
                for c in range(c3, c2):  # Fixed: include c3
                    output_grid[r2, c] = path_color
            
            # Then move vertically (not including the starting row since we already drew there)
            if r2 < r3:  # move down
                for r in range(r2 + 1, r3):  # Don't include r3 since that's the target cell
                    output_grid[r, c3] = path_color
            elif r2 > r3:  # move up
                for r in range(r3 + 1, r2):  # Don't include r3 since that's the target cell
                    output_grid[r, c3] = path_color
        
        return output_grid

