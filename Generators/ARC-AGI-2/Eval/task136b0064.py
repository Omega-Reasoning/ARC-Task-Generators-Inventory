from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import create_object, Contiguity
from Framework.transformation_library import BorderBehavior, find_connected_objects

class Task136b0064Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids have {vars['col']} columns and a varying number of rows, restricted to values of the form 4*m + 3, where m is a positive integer greater than 2 and less than 6.",
            "The grid contains several single-colored objects, one {color('vertical')} vertical line, and a single {color('cell_color')} cell. All other cells are empty (0).",
            "Each single-colored object fits within a 3x3 grid and can be one of the following four types.",
            "Type 1: defined as [[{color('type1')}, 0, {color('type1')}], [{color('type1')}, 0, {color('type1')}], [{color('type1')}, {color('type1')}, {color('type1')}]].",
            "Type 2: defined as [[{color('type2')}, {color('type2')}, 0], [{color('type2')}, 0, {color('type2')}], [0, {color('type2')}, 0]].",
            "Type 3: defined as [[{color('type3')}, 0, {color('type3')}], [0, {color('type3')}, 0], [0, {color('type3')}, 0]].",
            "Type 4: defined as[[{color('type4')}, {color('type4')}, {color('type4')}], [0, {color('type4')}, 0], [{color('type4')}, 0, {color('type4')}]].",
            "First, create the {color('vertical')} vertical line by filling the entire middle column of the grid with {color('vertical')} color.",
            "Then, divide the grid into 3x3 subgrids starting from the first column and ending just before the {color('vertical')} line.",
            "Group the columns in sets of 3: columns 0–2 form the first group, and if space allows, leave column 3 empty and start the next group from column 4.",
            "Similarly, group the rows: rows 0–2 form the first group, leave row 3 empty, and continue the same pattern for any additional 3-row groups.",
            "The first 3x3 subgrid starts from position (0, 0) and ends at (2, 2).",
            "Each 3x3 subgrid receives one object.",
            "The single {color('cell_color')} cell should appear in the first row and can be placed in columns ranging from {(vars['col'] // 2) + 2} to {(3 * (vars['col']+1) // 4)}.",
            "A running score is used to decide the placement of single-colored objects. The score starts at the value which defines the position of the {color('cell_color')} cell (counted from the first column after the vertical line as 1, 2, 3, etc.).",
            "Before adding each object (type 1 = −1, type 2 = +2, type 4 = −3), its score is added to the current total. If the new score stays within the valid range (from 0 to {(vars['col']+1)//2 - 1}), the object is added; otherwise, it’s replaced with another object type which keeps the score valid."
        ]

        transformation_reasoning_chain = [
            "The output grid is smaller than the input grid; it is constructed by copying the right half of the input grid, excluding the {color('vertical')} middle column.",
            "First, identify all single-colored objects on the left side of the {color('vertical')} column from the input grid. These objects are placed in 3x3 subgrids and can be of the following four types.",
            "Type 1: defined as [[{color('type1')}, 0, {color('type1')}], [{color('type1')}, 0, {color('type1')}], [{color('type1')}, {color('type1')}, {color('type1')}]].",
            "Type 2: defined as [[{color('type2')}, {color('type2')}, 0], [{color('type2')}, 0, {color('type2')}], [0, {color('type2')}, 0]].",
            "Type 3: defined as [[{color('type3')}, 0, {color('type3')}], [0, {color('type3')}, 0], [0, {color('type3')}, 0]].",
            "Type 4: defined as [[{color('type4')}, {color('type4')}, {color('type4')}], [0, {color('type4')}, 0], [{color('type4')}, 0, {color('type4')}]].",
            "Based on the type of each object, vertical or horizontal strips are generated starting vertically below the {color('cell_color')} cell.",
            "For example, if the identified object is Type 1, it results in a horizontal strip two cells wide using {color('type1')} color. This strip starts exactly below the previously added cell or directly below the {color('cell_color')} cell if it is the first one and extends towards the left side.",
            "Similarly, if the identified object is Type 2, it results in a horizontal strip three cells wide using {color('type2')} color. This strip starts exactly below the previously added cell or directly below the {color('cell_color')} cell if it is the first one and extends towards the right side.",
            "If the identified object is Type 3, it results in a vertical strip i.e two cells long using {color('type3')} color. This strip starts exactly below the previously added cell or directly below the {color('cell_color')} cell if it is the first one and extends in the downward direction.",
            "If the identified object is Type 4, it results in a horizontal strip four cells wide using {color('type4')} color. This strip starts exactly below the previously added cell or directly below the {color('cell_color')} cell if it is the first one and extends towards the left side."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        # Randomly select unique colors for objects
        colors = random.sample(range(1, 10), 6)
        
        taskvars = {
            'col': random.choice([15, 23]),  # Only 15 or 23 columns allowed
            'type1': colors[0],
            'type2': colors[1],
            'type3': colors[2],
            'type4': colors[3],
            'cell_color': colors[4],
            'vertical': colors[5]
        }
        
        # Generate train cases according to constraints
        train_data = []
        
        # Case 1: First object with type 1, below it is type 2
        grid_vars1 = {'first_object': 'type1', 'second_object': 'type2'}
        input_grid1 = self.create_input(taskvars, grid_vars1)
        output_grid1 = self.transform_input(input_grid1, taskvars)
        train_data.append({'input': input_grid1, 'output': output_grid1})
        
        # Case 2: First object with type 2, below it is type 2 again
        grid_vars2 = {'first_object': 'type2', 'second_object': 'type2'}
        input_grid2 = self.create_input(taskvars, grid_vars2)
        output_grid2 = self.transform_input(input_grid2, taskvars)
        train_data.append({'input': input_grid2, 'output': output_grid2})
        
        # Case 3: First object with type 3, below it is type 1
        grid_vars3 = {'first_object': 'type3', 'second_object': 'type1'}
        input_grid3 = self.create_input(taskvars, grid_vars3)
        output_grid3 = self.transform_input(input_grid3, taskvars)
        train_data.append({'input': input_grid3, 'output': output_grid3})
        
        # Optionally add a 4th case with different composition
        if random.choice([True, False]):
            grid_vars4 = {'first_object': random.choice(['type1', 'type2', 'type3']), 
                         'second_object': random.choice(['type1', 'type2', 'type3', 'type4'])}
            input_grid4 = self.create_input(taskvars, grid_vars4)
            output_grid4 = self.transform_input(input_grid4, taskvars)
            train_data.append({'input': input_grid4, 'output': output_grid4})
        
        # Test case
        grid_vars_test = {'first_object': random.choice(['type1', 'type2', 'type3']), 
                        'second_object': random.choice(['type1', 'type2', 'type3', 'type4'])}
        test_input = self.create_input(taskvars, grid_vars_test)
        test_output = self.transform_input(test_input, taskvars)
        
        test_data = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_data, 'test': test_data}

    def create_input(self, taskvars, gridvars):
        # Calculate grid dimensions
        col = taskvars['col']
        m = random.randint(3, 5)  # m between 3 and 5 inclusive
        row = 4 * m + 3
        
        # Create empty grid
        grid = np.zeros((row, col), dtype=int)
        
        # Create vertical line in the middle column
        middle_col = col // 2
        grid[:, middle_col] = taskvars['vertical']
        
        # Define object types
        object_types = {
            'type1': np.array([
                [taskvars['type1'], 0, taskvars['type1']],
                [taskvars['type1'], 0, taskvars['type1']],
                [taskvars['type1'], taskvars['type1'], taskvars['type1']]
            ]),
            'type2': np.array([
                [taskvars['type2'], taskvars['type2'], 0],
                [taskvars['type2'], 0, taskvars['type2']],
                [0, taskvars['type2'], 0]
            ]),
            'type3': np.array([
                [taskvars['type3'], 0, taskvars['type3']],
                [0, taskvars['type3'], 0],
                [0, taskvars['type3'], 0]
            ]),
            'type4': np.array([
                [taskvars['type4'], taskvars['type4'], taskvars['type4']],
                [0, taskvars['type4'], 0],
                [taskvars['type4'], 0, taskvars['type4']]
            ])
        }
        
        # Determine number of 3x3 subgrids we can fit on the left side
        left_columns = middle_col
        max_col_groups = left_columns // 4 + (1 if left_columns % 4 > 0 else 0)
        max_row_groups = (row - 3) // 4 + 1
        total_objects_needed = max_col_groups * max_row_groups
        
        # Define function to get the score contribution of each object type
        def get_object_score(obj_type):
            if obj_type == 'type1':
                return -1
            elif obj_type == 'type2':
                return 2
            elif obj_type == 'type4':
                return -3
            else:  # type3 doesn't contribute to the score
                return 0
        
        # Set the preferred position for cell_color: 2 cells right of vertical line
        preferred_pos = middle_col + 2
        cell_offset = preferred_pos - middle_col  # Should be 2
        
        # Define the valid score range
        valid_min = 0
        valid_max = (col + 1) // 2 - 1
        
        # Start with the first two objects (from gridvars or random)
        first_object_type = gridvars.get('first_object', random.choice(['type1', 'type2', 'type3']))
        
        # Ensure the first object isn't type4
        if first_object_type == 'type4':
            first_object_type = random.choice(['type1', 'type2', 'type3'])
        
        # Check if first object would result in a valid score
        first_score = get_object_score(first_object_type)
        first_total = first_score + cell_offset
        if not (valid_min <= first_total <= valid_max) and 'first_object' not in gridvars:
            # Try to find a valid alternative
            for alt_type in ['type1', 'type2', 'type3']:
                if alt_type != first_object_type:
                    alt_score = get_object_score(alt_type)
                    alt_total = alt_score + cell_offset
                    if valid_min <= alt_total <= valid_max:
                        first_object_type = alt_type
                        first_score = alt_score
                        first_total = alt_total
                        break
        
        # Get the second object (from gridvars or random)
        second_object_type = gridvars.get('second_object', random.choice(['type1', 'type2', 'type3', 'type4']))
        
        # Apply constraints for second object
        if first_object_type == 'type1' and second_object_type in ['type1', 'type4']:
            second_object_type = random.choice(['type2', 'type3'])
        
        if second_object_type in ['type1', 'type4'] and first_object_type != 'type2':
            second_object_type = 'type2'
        
        # Check if second object would maintain a valid score
        second_score = get_object_score(second_object_type)
        combined_score = first_score + second_score
        combined_total = combined_score + cell_offset
        
        if not (valid_min <= combined_total <= valid_max) and 'second_object' not in gridvars:
            # Try to find a valid alternative that respects the constraints
            valid_alternatives = []
            for alt_type in ['type1', 'type2', 'type3', 'type4']:
                # Skip if it violates the constraints
                if (first_object_type == 'type1' and alt_type in ['type1', 'type4']) or \
                   (alt_type in ['type1', 'type4'] and first_object_type != 'type2'):
                    continue
                
                alt_score = get_object_score(alt_type)
                alt_total = first_score + alt_score + cell_offset
                if valid_min <= alt_total <= valid_max:
                    valid_alternatives.append(alt_type)
            
            if valid_alternatives:
                second_object_type = random.choice(valid_alternatives)
                second_score = get_object_score(second_object_type)
                combined_score = first_score + second_score
        
        # Start building our sequence with the first two objects
        objects_sequence = [first_object_type, second_object_type]
        current_score = combined_score
        
        # Track type3 count for the large grid constraint
        is_large_grid = (col == 23)
        type3_count = objects_sequence.count('type3')
        type3_limit_reached = is_large_grid and type3_count >= 2
        
        # Now build rest of the sequence while strictly ensuring score validity at each step
        attempts = 0
        max_attempts = 1000
        
        while len(objects_sequence) < total_objects_needed and attempts < max_attempts:
            # Get available types based on constraints
            available_types = ['type1', 'type2', 'type3', 'type4']
            last_obj = objects_sequence[-1]
            second_last_obj = objects_sequence[-2] if len(objects_sequence) >= 2 else None
            
            # Apply constraint 3: If col is 23, limit type3 to at most 2
            if is_large_grid and type3_limit_reached:
                if 'type3' in available_types:
                    available_types.remove('type3')
            
            # Apply constraint 1: Never put more than one type1 or type4 consecutively
            if last_obj in ['type1', 'type4']:
                if 'type1' in available_types:
                    available_types.remove('type1')
                if 'type4' in available_types:
                    available_types.remove('type4')
            
            # Apply constraint 2: Always have a type2 before any type1 or type4
            if last_obj != 'type2':
                if 'type1' in available_types:
                    available_types.remove('type1')
                if 'type4' in available_types:
                    available_types.remove('type4')
            
            # Apply constraint 4: No triple occurrence of type1, type2, type4
            if second_last_obj is not None:
                last_two = set([last_obj, second_last_obj])
                if len(last_two.intersection({'type1', 'type2', 'type4'})) == 2:
                    remaining_critical = list({'type1', 'type2', 'type4'} - last_two)
                    if remaining_critical and remaining_critical[0] in available_types:
                        available_types.remove(remaining_critical[0])
            
            # If no valid types after applying constraints, relax them
            if not available_types:
                # Try relaxing the constraint against consecutive type1/type4
                if last_obj in ['type1', 'type4']:
                    available_types = ['type1', 'type4']
                    
                # If that doesn't help, try relaxing the type2 before type1/type4 constraint
                if not available_types and last_obj != 'type2':
                    available_types = ['type1', 'type4']
                    
                # If still no valid types, use type3 (which doesn't impact score)
                if not available_types and not type3_limit_reached:
                    available_types = ['type3']
                    
                # If we can't use type3 either, try type2 (which increases score)
                if not available_types:
                    available_types = ['type2']
            
            # Filter types that would make the score invalid
            valid_types = []
            for next_type in available_types:
                next_score = current_score + get_object_score(next_type)
                next_total = next_score + cell_offset
                if valid_min <= next_total <= valid_max:
                    valid_types.append(next_type)
            
            # If no valid types found, try to use type3 (doesn't change score)
            if not valid_types and not type3_limit_reached:
                if current_score + cell_offset >= valid_min and current_score + cell_offset <= valid_max:
                    valid_types = ['type3']
            
            # If still no valid types, try more aggressive corrections
            if not valid_types:
                # If score is too low, try type2 to increase it
                if current_score + cell_offset < valid_min:
                    next_score = current_score + get_object_score('type2')
                    next_total = next_score + cell_offset
                    if next_total <= valid_max:  # Only add if it doesn't exceed max
                        valid_types = ['type2']
                
                # If score is too high, try type1 or type4 to decrease it
                elif current_score + cell_offset > valid_max:
                    # Try type1 first (smaller decrease)
                    next_score = current_score + get_object_score('type1')
                    next_total = next_score + cell_offset
                    if next_total >= valid_min:
                        valid_types = ['type1']
                    else:
                        # Try type4 (larger decrease)
                        next_score = current_score + get_object_score('type4')
                        next_total = next_score + cell_offset
                        if next_total >= valid_min:
                            valid_types = ['type4']
            
            # If we found valid types, choose one randomly
            if valid_types:
                next_type = random.choice(valid_types)
                objects_sequence.append(next_type)
                current_score += get_object_score(next_type)
                
                # Update type3 counter if needed
                if next_type == 'type3':
                    type3_count += 1
                    if is_large_grid and type3_count >= 2:
                        type3_limit_reached = True
            else:
                # If we couldn't find any valid next object, restart with different options
                attempts += 1
                
                # Reset sequence but keep the fixed first and second objects
                objects_sequence = [first_object_type, second_object_type]
                current_score = get_object_score(first_object_type) + get_object_score(second_object_type)
                type3_count = objects_sequence.count('type3')
                type3_limit_reached = is_large_grid and type3_count >= 2
        
        # If we couldn't build a complete valid sequence, fill with type3 objects
        while len(objects_sequence) < total_objects_needed:
            objects_sequence.append('type3')
        
        # Now place all objects in the grid
        object_idx = 0
        
        # First, place objects in the first column (top to bottom)
        for i in range(min(max_row_groups, len(objects_sequence))):
            row_start = i * 4
            if row_start + 3 <= row:
                grid[row_start:row_start+3, 0:3] = object_types[objects_sequence[object_idx]]
                object_idx += 1
                
        # Then fill any additional columns
        for j in range(1, max_col_groups):
            col_start = j * 4
            if col_start + 3 <= left_columns:
                for i in range(max_row_groups):
                    if object_idx < len(objects_sequence):
                        row_start = i * 4
                        if row_start + 3 <= row:
                            grid[row_start:row_start+3, col_start:col_start+3] = object_types[objects_sequence[object_idx]]
                            object_idx += 1
        
        # Place the cell_color at the preferred position
        grid[0, preferred_pos] = taskvars['cell_color']
        
        # Calculate final score for debugging
        final_score = current_score + cell_offset
        # print(f"Final sequence: {objects_sequence}")
        # print(f"Final score: {final_score} (valid range: [0, {valid_max}])")
        # print(f"Cell position: {preferred_pos} (middle_col: {middle_col}, offset: {cell_offset})")
        
        return grid

    def transform_input(self, grid, taskvars):
        # Find the middle column with vertical line
        middle_col = None
        for col in range(grid.shape[1]):
            if np.all(grid[:, col] == taskvars['vertical']):
                middle_col = col
                break
        
        if middle_col is None:
            raise ValueError("Vertical line not found in grid")
        
        # Find the cell_color position
        cell_row, cell_col = None, None
        for col in range(middle_col + 1, grid.shape[1]):
            if grid[0, col] == taskvars['cell_color']:
                cell_row, cell_col = 0, col
                break
        
        if cell_row is None:
            raise ValueError("Cell color not found in first row on right side")
        
        # Extract objects from the left side of the vertical line in the correct order
        left_grid = grid[:, :middle_col]
        objects = []
        
        # Determine number of 3x3 subgrids we can fit on the left side
        left_width = middle_col
        max_col_groups = left_width // 4 + (1 if left_width % 4 > 0 else 0)
        max_row_groups = (grid.shape[0] - 3) // 4 + 1
        
        # Process objects in the correct order: first column from top to bottom,
        # then second column from top to bottom, and so on
        for j in range(max_col_groups):
            col_start = j * 4
            if col_start + 3 <= left_width:
                for i in range(max_row_groups):
                    row_start = i * 4
                    if row_start + 3 <= grid.shape[0]:
                        subgrid = left_grid[row_start:row_start+3, col_start:col_start+3]
                        
                        # Identify object type
                        if np.array_equal(subgrid, np.array([
                            [taskvars['type1'], 0, taskvars['type1']],
                            [taskvars['type1'], 0, taskvars['type1']],
                            [taskvars['type1'], taskvars['type1'], taskvars['type1']]
                        ])):
                            objects.append('type1')
                        elif np.array_equal(subgrid, np.array([
                            [taskvars['type2'], taskvars['type2'], 0],
                            [taskvars['type2'], 0, taskvars['type2']],
                            [0, taskvars['type2'], 0]
                        ])):
                            objects.append('type2')
                        elif np.array_equal(subgrid, np.array([
                            [taskvars['type3'], 0, taskvars['type3']],
                            [0, taskvars['type3'], 0],
                            [0, taskvars['type3'], 0]
                        ])):
                            objects.append('type3')
                        elif np.array_equal(subgrid, np.array([
                            [taskvars['type4'], taskvars['type4'], taskvars['type4']],
                            [0, taskvars['type4'], 0],
                            [taskvars['type4'], 0, taskvars['type4']]
                        ])):
                            objects.append('type4')
        
        # Create output grid (copy right half of input, excluding vertical line)
        right_grid = grid[:, middle_col+1:]
        output_grid = np.zeros_like(right_grid)
        
        # Place the cell_color in the first row
        output_col = cell_col - (middle_col + 1)
        if 0 <= output_col < output_grid.shape[1]:
            output_grid[0, output_col] = taskvars['cell_color']
        else:
            # Fallback if position is out of bounds
            output_grid[0, output_grid.shape[1] // 2] = taskvars['cell_color']
            output_col = output_grid.shape[1] // 2
        
        # Starting point is directly below the cell_color
        current_row = 1
        current_col = output_col
        max_rows = output_grid.shape[0]
        max_cols = output_grid.shape[1]
        
        # Add strips based on the objects identified in order
        for obj_type in objects:
            # Check if we've reached the bottom of the grid
            if current_row >= max_rows:
                break  # Stop adding more strips
                
            if obj_type == 'type1':
                # Horizontal strip two cells wide, extending LEFT from current position
                strip_width = 2
                
                # Calculate start column (ensuring we stay within grid boundaries)
                start_col = max(0, current_col - (strip_width - 1))
                
                # If we need to adjust where the strip starts
                if start_col == 0 and current_col > 0:
                    # We hit the left boundary, so we're only placing what fits
                    actual_width = current_col + 1
                else:
                    actual_width = strip_width
                
                # Place the strip
                if current_row < max_rows:
                    for c in range(actual_width):
                        col_idx = current_col - c
                        if 0 <= col_idx < max_cols:
                            output_grid[current_row, col_idx] = taskvars['type1']
                    
                    # Update current position for next strip
                    current_col = start_col
                    current_row += 1
            
            elif obj_type == 'type2':
                # Horizontal strip three cells wide, extending RIGHT from current position
                strip_width = 3
                
                # Calculate end column (ensuring we stay within grid boundaries)
                end_col = min(max_cols - 1, current_col + (strip_width - 1))
                
                # If we need to adjust where the strip ends
                if end_col == max_cols - 1 and current_col < max_cols - 1:
                    # We hit the right boundary, so we're only placing what fits
                    actual_width = end_col - current_col + 1
                else:
                    actual_width = strip_width
                
                # Place the strip
                if current_row < max_rows:
                    for c in range(actual_width):
                        col_idx = current_col + c
                        if 0 <= col_idx < max_cols:
                            output_grid[current_row, col_idx] = taskvars['type2']
                    
                    # Update current position for next strip
                    current_col = end_col
                    current_row += 1
            
            elif obj_type == 'type3':
                # Vertical strip two cells long, extending DOWN from current position
                strip_height = 2
                
                # Calculate end row (ensuring we stay within grid boundaries)
                end_row = min(max_rows - 1, current_row + (strip_height - 1))
                
                # If we need to adjust where the strip ends
                if end_row == max_rows - 1 and current_row < max_rows - 1:
                    # We hit the bottom boundary, so we're only placing what fits
                    actual_height = end_row - current_row + 1
                else:
                    actual_height = strip_height
                
                # Place the strip
                if 0 <= current_col < max_cols:
                    for r in range(actual_height):
                        row_idx = current_row + r
                        if row_idx < max_rows:
                            output_grid[row_idx, current_col] = taskvars['type3']
                    
                    # Update current position for next strip
                    current_row = end_row + 1
                    # current_col stays the same
            
            elif obj_type == 'type4':
                # Horizontal strip four cells wide, extending LEFT from current position
                strip_width = 4
                
                # Calculate start column (ensuring we stay within grid boundaries)
                start_col = max(0, current_col - (strip_width - 1))
                
                # If we need to adjust where the strip starts
                if start_col == 0 and current_col > 0:
                    # We hit the left boundary, so we're only placing what fits
                    actual_width = current_col + 1
                else:
                    actual_width = strip_width
                
                # Place the strip
                if current_row < max_rows:
                    for c in range(actual_width):
                        col_idx = current_col - c
                        if 0 <= col_idx < max_cols:
                            output_grid[current_row, col_idx] = taskvars['type4']
                    
                    # Update current position for next strip
                    current_col = start_col
                    current_row += 1
        
        return output_grid
