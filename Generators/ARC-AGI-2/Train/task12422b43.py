from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObject, GridObjects
from Framework.input_library import create_object, retry, random_cell_coloring
import numpy as np
import random

class Task12422b43Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "Each grid contains a vertical rectangle formed by filling the first several cells of the first column with {color('vertical_block')} color, several colored objects made of 4-way connected cells, and the remaining cells are empty (0).",
            "The {color('vertical_block')} block starts from the first row and can be between 1 to 4 cells long, this should vary across examples.",
            "The colored objects, excluding the {color('vertical_block')} rectangular block, are placed in any columns except the first one and are all vertically connected to each other.",
            "These objects have horizontal block shapes, with possible dimensions such as 1x1, 1x2, 1x3, 2x2 or similar.",
            "The last few rows of the grid must remain empty (0). The number of empty (0) rows should either be equal to the length of the {color('vertical_block')} block or a multiple of it."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying the {color('vertical_block')} vertical rectangle located in the first column, along with the colored objects made of 4-way connected cells located in the interior columns.",
            "Then, all interior columns containing colored objects (excluding the {color('vertical_block')} block) are copied starting from the first row and stopping at the row containing the last {color('vertical_block')} cell.",
            "This copied section is then pasted directly below the existing objects, starting from the first completely empty (0) row, and is repeatedly pasted until the last row of the grid has been filled or reached."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        height = gridvars['height']
        width = gridvars['width']
        vertical_block_length = gridvars['vertical_block_length']
        num_objects = gridvars['num_objects']  # 2 or 3
        vertical_block_color = taskvars['vertical_block']
        
        # Create empty grid
        grid = np.zeros((height, width), dtype=int)
        
        # Place vertical block in first column
        for i in range(vertical_block_length):
            grid[i, 0] = vertical_block_color
        
        # Available colors for objects (excluding vertical block color and background)
        available_colors = [c for c in range(1, 10) if c != vertical_block_color]
        
        # Calculate constraints
        empty_rows_multiplier = gridvars.get('empty_rows_multiplier', random.randint(1, 3))
        required_empty_rows = vertical_block_length * empty_rows_multiplier
        
        # Objects must occupy at least until vertical_block_length-1 row
        min_object_end_row = vertical_block_length - 1
        
        # Objects can extend beyond but must leave required empty rows
        max_object_end_row = height - required_empty_rows - 1
        
        # Sometimes extend beyond vertical block, sometimes not
        extend_beyond = random.choice([True, False])
        if extend_beyond:
            target_object_end_row = random.randint(min_object_end_row, 
                                                 min(max_object_end_row, min_object_end_row + 2))
        else:
            target_object_end_row = min_object_end_row
        
        # Create exactly num_objects (2 or 3) connected objects in interior columns
        # All objects must be vertically connected by sharing at least one common column
        current_row = 0
        
        # Choose a connecting column that will be shared by all objects for vertical connectivity
        connecting_col = random.randint(1, width - 2)  # Interior column only
        
        # Select exactly num_objects colors
        object_colors = random.sample(available_colors, num_objects)
        
        for i, color in enumerate(object_colors):
            # Choose object dimensions
            obj_shapes = [(1, 1), (1, 2), (1, 3), (2, 2)]
            obj_height, obj_width = random.choice(obj_shapes)
            
            # Ensure we have enough rows remaining
            remaining_rows_needed = max(0, target_object_end_row - current_row + 1)
            remaining_objects = len(object_colors) - i
            
            if remaining_objects > 0 and remaining_rows_needed > 0:
                min_height_needed = max(1, remaining_rows_needed // remaining_objects)
                obj_height = max(obj_height, min_height_needed)
            
            # Make sure object fits vertically within constraints
            max_possible_height = max_object_end_row + 1 - current_row
            if max_possible_height <= 0:
                break
            obj_height = min(obj_height, max_possible_height)
            
            # Position object to ensure it includes the connecting column
            # This guarantees vertical 4-way connectivity
            if obj_width == 1:
                # 1-width objects go exactly on the connecting column
                col_start = connecting_col
            else:
                # Multi-width objects must include the connecting column
                # Choose position so connecting_col is within [col_start, col_start + obj_width - 1]
                min_col_start = max(1, connecting_col - obj_width + 1)
                max_col_start = min(connecting_col, width - 1 - obj_width)
                
                if min_col_start <= max_col_start:
                    col_start = random.randint(min_col_start, max_col_start)
                else:
                    # Fallback: reduce object width to fit
                    obj_width = 1
                    col_start = connecting_col
            
            # Place object
            for r in range(current_row, current_row + obj_height):
                for c in range(col_start, col_start + obj_width):
                    if r < height and 1 <= c < width - 1:  # Interior columns only
                        grid[r, c] = color
            
            # Move to next row (vertically connected - no gaps)
            current_row += obj_height
            
            # Stop if we've reached our target
            if current_row > target_object_end_row:
                break
        
        return grid

    def transform_input(self, grid, taskvars):
        height, width = grid.shape
        vertical_block_color = taskvars['vertical_block']
        
        # Create output grid as copy of input
        output = grid.copy()
        
        # Find the length of vertical block
        vertical_block_length = 0
        for i in range(height):
            if grid[i, 0] == vertical_block_color:
                vertical_block_length = i + 1
            else:
                break
        
        # Extract the section to copy (interior columns from row 0 to vertical_block_length-1)
        section_to_copy = grid[:vertical_block_length, 1:]
        
        # Find first completely empty row
        first_empty_row = height
        for r in range(height):
            if np.all(grid[r, :] == 0):
                first_empty_row = r
                break
        
        # Paste the section repeatedly starting from first empty row
        paste_row = first_empty_row
        while paste_row < height:
            # Calculate how much we can paste
            rows_available = height - paste_row
            rows_to_paste = min(vertical_block_length, rows_available)
            
            # Paste the section
            output[paste_row:paste_row + rows_to_paste, 1:] = section_to_copy[:rows_to_paste, :]
            
            paste_row += vertical_block_length
        
        return output

    def create_grids(self):
        # Task variables
        taskvars = {
            'vertical_block': random.randint(1, 9)
        }
        
        # Create training examples
        train_examples = []
        
        # Ensure we have examples with 2 and 3 cell vertical blocks as required
        required_lengths = [2, 3]
        
        for i in range(3):  # Create 3 training examples
            if i < len(required_lengths):
                vertical_block_length = required_lengths[i]
            else:
                vertical_block_length = random.randint(1, 4)
            
            # Determine number of objects (2 or 3)
            num_objects = random.choice([2, 3])
            
            # Determine empty rows multiplier
            empty_rows_multiplier = random.randint(1, 3)
            
            # Create gridvars for this example
            gridvars = {
                'height': random.randint(12, 20),
                'width': random.randint(6, 12),
                'vertical_block_length': vertical_block_length,
                'num_objects': num_objects,
                'empty_rows_multiplier': empty_rows_multiplier
            }
            
            # Ensure minimum width for interior columns
            gridvars['width'] = max(gridvars['width'], 5)  # At least 5 columns
            
            # Adjust height to accommodate all components
            required_empty_rows = vertical_block_length * empty_rows_multiplier
            min_height = vertical_block_length + num_objects + required_empty_rows + 2
            gridvars['height'] = max(gridvars['height'], min_height)
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_vertical_block_length = random.randint(1, 4)
        test_num_objects = random.choice([2, 3])
        test_empty_rows_multiplier = random.randint(2, 4)
        
        test_gridvars = {
            'height': random.randint(15, 25),
            'width': random.randint(8, 15),
            'vertical_block_length': test_vertical_block_length,
            'num_objects': test_num_objects,
            'empty_rows_multiplier': test_empty_rows_multiplier
        }
        
        # Ensure minimum width for interior columns
        test_gridvars['width'] = max(test_gridvars['width'], 5)
        
        # Ensure enough height for test
        test_required_empty_rows = test_vertical_block_length * test_empty_rows_multiplier
        min_height = test_vertical_block_length + test_num_objects + test_required_empty_rows + 2
        test_gridvars['height'] = max(test_gridvars['height'], min_height)
        
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }

