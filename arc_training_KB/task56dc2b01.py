from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, GridObject, GridObjects

class Task56dc2b01Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "They contain a single completely filled column or row with {color('line_color1')} color, forming a single vertical or horizontal {color('line_color1')} line, and an object made of 4-way connected cells of {color('object_color')} color.",
            "The {color('object_color')} object must be separated from the vertical or horizontal {color('line_color1')} line by at least three empty (0) columns or rows, respectively.",
            "The position and shape of the {color('object_color')} object varies across examples.",
            "All remaining cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "Output grids are constructed by copying the input grids and identifying whether the {color('line_color1')} line is horizontal or vertical.",
            "If the line is horizontal, the {color('object_color')} object is moved vertically towards it so they become vertically connected.",
            "If the line is vertical, the {color('object_color')} object is moved horizontally towards it so they become horizontally connected.",
            "Once the {color('object_color')} object is connected to the {color('line_color1')} line, a new line of {color('line_color2')} is added on the opposite side of the object by completely filling a row or column, depending on whether the original {color('line_color1')} line was horizontal or vertical.",
            "The new line is oriented in the same direction as the {color('line_color1')} line—horizontal or vertical.",
            "The transformation preserves the original shape of the {color('object_color')} object."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Initialize task variables
        taskvars = {
            'object_color': random.randint(1, 9),
            'line_color1': 0,
            'line_color2': 0
        }
        
        # Ensure colors are different
        while taskvars['line_color1'] == 0 or taskvars['line_color1'] == taskvars['object_color']:
            taskvars['line_color1'] = random.randint(1, 9)
        
        while taskvars['line_color2'] == 0 or taskvars['line_color2'] == taskvars['object_color'] or taskvars['line_color2'] == taskvars['line_color1']:
            taskvars['line_color2'] = random.randint(1, 9)
        
        # Create training examples
        train_examples = []
        
        # Ensure we have at least one horizontal and one vertical example
        horizontal_added = False
        vertical_added = False
        
        for i in range(random.randint(3, 4)):
            is_horizontal = True
            if i == 0 and not vertical_added:
                is_horizontal = False
                vertical_added = True
            elif i == 1 and not horizontal_added:
                is_horizontal = True
                horizontal_added = True
            else:
                is_horizontal = random.choice([True, False])
            
            gridvars = {'is_horizontal': is_horizontal}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test examples - one horizontal, one vertical
        test_examples = []
        
        # Horizontal test example
        gridvars = {'is_horizontal': True}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        test_examples.append({
            'input': input_grid,
            'output': output_grid
        })
        
        # Vertical test example
        gridvars = {'is_horizontal': False}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        test_examples.append({
            'input': input_grid,
            'output': output_grid
        })
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
    
    def create_input(self, taskvars, gridvars):
        # Determine grid size (between 8x8 and 30x30)
        height = random.randint(8, 30)
        width = random.randint(8, 30)
        
        # Create empty grid
        grid = np.zeros((height, width), dtype=int)
        
        # Determine if line is horizontal or vertical
        is_horizontal = gridvars.get('is_horizontal', random.choice([True, False]))
        
        # Place the line
        line_color = taskvars['line_color1']
        
        if is_horizontal:
            # Horizontal line
            line_position = random.randint(0, height - 1)
            grid[line_position, :] = line_color
        else:
            # Vertical line
            line_position = random.randint(0, width - 1)
            grid[:, line_position] = line_color
        
        # Determine suitable area for the object
        obj_min_size = 4
        obj_max_size = min(10, height//2, width//2)
        min_separation = 3
        
        # Generate object dimensions
        obj_height = random.randint(obj_min_size, obj_max_size)
        obj_width = random.randint(obj_min_size, obj_max_size)
        
        # Determine position for the object (ensuring minimum separation)
        if is_horizontal:
            valid_rows = list(range(0, line_position - obj_height - min_separation + 1)) + \
                        list(range(line_position + min_separation, height - obj_height + 1))
            if not valid_rows:
                # Adjust if no valid position found
                valid_rows = [max(0, line_position - obj_height - min_separation)]
            
            obj_row = random.choice(valid_rows)
            obj_col = random.randint(0, width - obj_width)
        else:
            valid_cols = list(range(0, line_position - obj_width - min_separation + 1)) + \
                         list(range(line_position + min_separation, width - obj_width + 1))
            if not valid_cols:
                # Adjust if no valid position found
                valid_cols = [max(0, line_position - obj_width - min_separation)]
            
            obj_row = random.randint(0, height - obj_height)
            obj_col = random.choice(valid_cols)
        
        # Create and validate the object
        def create_candidate():
            return create_object(
                height=obj_height,
                width=obj_width,
                color_palette=taskvars['object_color'],
                contiguity=Contiguity.FOUR
            )
        
        def is_valid_object(obj):
            # Check if spans multiple rows and columns and has enough cells
            non_zero_count = np.sum(obj != 0)
            has_multi_rows = len(set(np.where(obj != 0)[0])) > 1
            has_multi_cols = len(set(np.where(obj != 0)[1])) > 1
            return non_zero_count >= 5 and has_multi_rows and has_multi_cols
        
        object_matrix = retry(create_candidate, is_valid_object, max_attempts=50)
        
        # Place the object on the grid
        for r in range(obj_height):
            for c in range(obj_width):
                if object_matrix[r, c] != 0:
                    grid[obj_row + r, obj_col + c] = object_matrix[r, c]
        
        return grid
    
    def transform_input(self, grid, taskvars):
        # Copy the input grid
        output = grid.copy()
        
        # Find line color and object color
        line_color = taskvars['line_color1']
        object_color = taskvars['object_color']
        new_line_color = taskvars['line_color2']
        
        # Check if the line is horizontal or vertical by counting occurrences in rows vs cols
        row_counts = np.sum(grid == line_color, axis=1)
        col_counts = np.sum(grid == line_color, axis=0)
        
        is_horizontal = np.max(row_counts) > np.max(col_counts)
        
        # Find the line position
        if is_horizontal:
            line_position = np.argmax(row_counts)
        else:
            line_position = np.argmax(col_counts)
        
        # Find the object
        objects = find_connected_objects(grid, diagonal_connectivity=False)
        object_cells = objects.with_color(object_color)
        
        if len(object_cells) == 0:
            # No object found, return input
            return grid
        
        # Get the main object
        obj = object_cells[0]
        
        # Cut the object from the grid
        obj.cut(output)
        
        # Determine how to move the object to connect with the line
        if is_horizontal:
            # Find the object's position
            obj_rows = [r for r, _, _ in obj.cells]
            obj_top = min(obj_rows)
            obj_bottom = max(obj_rows)
            
            # Determine direction to move
            if obj_top > line_position:
                # Move object upward to connect
                distance = obj_top - line_position - 1
                obj.translate(-distance, 0)
                
                # Add new horizontal line at the bottom
                new_line_row = obj_bottom - distance + 1
                if 0 <= new_line_row < output.shape[0]:
                    output[new_line_row, :] = new_line_color
            else:
                # Move object downward to connect
                distance = line_position - obj_bottom - 1
                obj.translate(distance, 0)
                
                # Add new horizontal line at the top
                new_line_row = obj_top + distance - 1
                if 0 <= new_line_row < output.shape[0]:
                    output[new_line_row, :] = new_line_color
        else:
            # Find the object's position
            obj_cols = [c for _, c, _ in obj.cells]
            obj_left = min(obj_cols)
            obj_right = max(obj_cols)
            
            # Determine direction to move
            if obj_left > line_position:
                # Move object leftward to connect
                distance = obj_left - line_position - 1
                obj.translate(0, -distance)
                
                # Add new vertical line at the right
                new_line_col = obj_right - distance + 1
                if 0 <= new_line_col < output.shape[1]:
                    output[:, new_line_col] = new_line_color
            else:
                # Move object rightward to connect
                distance = line_position - obj_right - 1
                obj.translate(0, distance)
                
                # Add new vertical line at the left
                new_line_col = obj_left + distance - 1
                if 0 <= new_line_col < output.shape[1]:
                    output[:, new_line_col] = new_line_color
        
        # Paste the object back
        obj.paste(output)
        
        return output

