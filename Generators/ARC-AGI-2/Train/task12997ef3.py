from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObject, GridObjects
from Framework.input_library import create_object, retry, enforce_object_width, enforce_object_height, Contiguity
import numpy as np
import random

class Task12997ef3Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids have {vars['rows']} rows and a varying number of columns.",
            "Each grid contains exactly one {color('object_color')} object, which is made of 8-way connected cells, as well as several additional multi-colored cells arranged in a single row or column. All remaining cells are empty (0).",
            "The {color('object_color')} object is shaped and sized to fit within a 3x3 subgrid, with the constraint that each row and column of that subgrid contains at least one {color('object_color')} cell.",
            "The additional multi-colored cells (outside the object) must be at least 2 in number and are placed with exactly one empty (0) cell between each pair of consecutive colored cells, either in a single row or a single column.",
            "Each of these additional multi-colored cells must have a different color.",
            "The row or column containing these additional multi-colored cells must be completely separated from the {color('object_color')} object — there should be no overlap or adjacency between them."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by identifying the {color('object_color')} object as well as the several additional multi-colored cells arranged in a single row or column.",
            "The number of multi-colored cells (n) and their horizontal or vertical arrangement are used to determine the size of the output grid.",
            "If the colored cells are arranged horizontally, the output grid has size 3 x (3 * n) else if the colored cells are arranged vertically, the output grid has size (3 * n) x 3.",
            "The {color('object_color')} object is then copied and pasted n times into the output grid — once for each multi-colored cell.",
            "Finally, the color of each pasted object is updated to match the color of the corresponding multi-colored cell."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        # Initialize task variables
        taskvars = {
            'rows': random.randint(9, 30),
            'object_color': random.randint(1, 9)
        }
        
        # Create training examples with specific constraints
        train_examples = []
        
        # First example: exactly 2 multi-colored cells arranged horizontally
        gridvars1 = {
            'num_colored_cells': 2,
            'arrangement': 'horizontal',
            'colored_cell_colors': self._get_unique_colors(2, exclude=[taskvars['object_color']])
        }
        input1 = self.create_input(taskvars, gridvars1)
        output1 = self.transform_input(input1, taskvars)
        train_examples.append({'input': input1, 'output': output1})
        
        # Second example: exactly 3 multi-colored cells arranged vertically
        gridvars2 = {
            'num_colored_cells': 3,
            'arrangement': 'vertical',
            'colored_cell_colors': self._get_unique_colors(3, exclude=[taskvars['object_color']])
        }
        input2 = self.create_input(taskvars, gridvars2)
        output2 = self.transform_input(input2, taskvars)
        train_examples.append({'input': input2, 'output': output2})
        
        # Additional training examples with random configurations
        for _ in range(random.randint(1, 2)):  # 1-2 more examples for variety
            num_cells = random.randint(2, 4)
            arrangement = random.choice(['horizontal', 'vertical'])
            gridvars = {
                'num_colored_cells': num_cells,
                'arrangement': arrangement,
                'colored_cell_colors': self._get_unique_colors(num_cells, exclude=[taskvars['object_color']])
            }
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Test example
        test_num_cells = random.randint(2, 4)
        test_arrangement = random.choice(['horizontal', 'vertical'])
        test_gridvars = {
            'num_colored_cells': test_num_cells,
            'arrangement': test_arrangement,
            'colored_cell_colors': self._get_unique_colors(test_num_cells, exclude=[taskvars['object_color']])
        }
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}

    def _get_unique_colors(self, num_colors, exclude):
        """Get a list of unique colors excluding specified colors."""
        available_colors = [c for c in range(1, 10) if c not in exclude]
        return random.sample(available_colors, num_colors)

    def create_input(self, taskvars, gridvars):
        def generate_grid():
            rows = taskvars['rows']
            object_color = taskvars['object_color']
            num_colored_cells = gridvars['num_colored_cells']
            arrangement = gridvars['arrangement']
            colored_cell_colors = gridvars['colored_cell_colors']
            
            # Calculate minimum required columns based on arrangement
            if arrangement == 'horizontal':
                min_cols = max(3, 2 * num_colored_cells - 1)  # Space for colored cells with gaps
            else:
                min_cols = 3  # Just need space for object
            
            cols = random.randint(min_cols + 3, 30)  # Extra space for separation
            grid = np.zeros((rows, cols), dtype=int)
            
            # Create 3x3 object with constraint that each row and column has at least one cell
            def create_constrained_object():
                obj = create_object(3, 3, object_color, contiguity=Contiguity.EIGHT)
                return obj
            
            # Ensure object spans all rows and columns of the 3x3 area
            object_3x3 = retry(
                create_constrained_object,
                lambda obj: (np.any(obj != 0, axis=1).all() and  # Each row has at least one colored cell
                           np.any(obj != 0, axis=0).all())        # Each column has at least one colored cell
            )
            
            # Place object in grid with some margin
            obj_start_row = random.randint(1, rows - 4)
            obj_start_col = random.randint(1, cols - 4)
            grid[obj_start_row:obj_start_row+3, obj_start_col:obj_start_col+3] = object_3x3
            
            # Place multi-colored cells
            if arrangement == 'horizontal':
                # Choose a row that's separated from the object
                if obj_start_row >= 2:
                    colored_row = random.randint(0, obj_start_row - 2)
                else:
                    colored_row = random.randint(obj_start_row + 4, rows - 1)
                
                # Place colored cells with gaps
                start_col = random.randint(0, cols - (2 * num_colored_cells - 1))
                for i, color in enumerate(colored_cell_colors):
                    col_pos = start_col + 2 * i
                    if col_pos < cols:
                        grid[colored_row, col_pos] = color
            else:  # vertical
                # Choose a column that's separated from the object
                if obj_start_col >= 2:
                    colored_col = random.randint(0, obj_start_col - 2)
                else:
                    colored_col = random.randint(obj_start_col + 4, cols - 1)
                
                # Place colored cells with gaps
                start_row = random.randint(0, rows - (2 * num_colored_cells - 1))
                for i, color in enumerate(colored_cell_colors):
                    row_pos = start_row + 2 * i
                    if row_pos < rows:
                        grid[row_pos, colored_col] = color
            
            return grid
        
        # Retry until we get a valid grid
        return retry(
            generate_grid,
            lambda g: self._validate_input_grid(g, taskvars, gridvars)
        )

    def _validate_input_grid(self, grid, taskvars, gridvars):
        """Validate that the input grid meets all constraints."""
        object_color = taskvars['object_color']
        num_colored_cells = gridvars['num_colored_cells']
        colored_cell_colors = gridvars['colored_cell_colors']
        
        # Find the main object
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)
        main_objects = objects.with_color(object_color)
        
        if len(main_objects) != 1:
            return False
        
        main_obj = main_objects[0]
        if main_obj.height > 3 or main_obj.width > 3:
            return False
        
        # Check that we have the right number of colored cells
        colored_cells = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) 
                        if grid[r, c] != 0 and grid[r, c] != object_color]
        
        if len(colored_cells) != num_colored_cells:
            return False
        
        # Check that colored cells have the right colors
        found_colors = {grid[r, c] for r, c in colored_cells}
        if found_colors != set(colored_cell_colors):
            return False
        
        return True

    def transform_input(self, grid, taskvars):
        object_color = taskvars['object_color']
        
        # Find the main object
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)
        main_obj = objects.with_color(object_color)[0]
        
        # Find colored cells (excluding the main object)
        colored_cells = []
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] != 0 and grid[r, c] != object_color:
                    colored_cells.append((r, c, grid[r, c]))
        
        n = len(colored_cells)
        
        # Determine arrangement
        if len(set(r for r, c, _ in colored_cells)) == 1:  # Same row
            arrangement = 'horizontal'
            output_grid = np.zeros((3, 3 * n), dtype=int)
        else:  # Same column
            arrangement = 'vertical'  
            output_grid = np.zeros((3 * n, 3), dtype=int)
        
        # Get the object as a 3x3 array
        obj_array = main_obj.to_array()
        # Pad to 3x3 if needed
        if obj_array.shape[0] < 3 or obj_array.shape[1] < 3:
            padded = np.zeros((3, 3), dtype=int)
            padded[:obj_array.shape[0], :obj_array.shape[1]] = obj_array
            obj_array = padded
        
        # Sort colored cells by position to maintain order
        if arrangement == 'horizontal':
            colored_cells.sort(key=lambda x: x[1])  # Sort by column
        else:
            colored_cells.sort(key=lambda x: x[0])  # Sort by row
        
        # Place copies of the object with updated colors
        for i, (_, _, color) in enumerate(colored_cells):
            colored_obj = np.where(obj_array == object_color, color, obj_array)
            
            if arrangement == 'horizontal':
                output_grid[:, i*3:(i+1)*3] = colored_obj
            else:
                output_grid[i*3:(i+1)*3, :] = colored_obj
        
        return output_grid

