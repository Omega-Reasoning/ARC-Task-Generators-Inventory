from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import random_cell_coloring
from Framework.transformation_library import find_connected_objects

class Task46f33fceGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}..",
            "They contain several colored (1-9) cells, with all remaining being empty (0).",
            "The colored cells use the following colors: {color('cell_color1')}, {color('cell_color2')}, {color('cell_color3')}, {color('cell_color4')}, and sometimes a single cell has a different color (1-9).",
            "All odd rows and columns must be completely empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are of size {2*vars['grid_size']}x{2*vars['grid_size']}.",
            "They are constructed by completely dividing the input grid into {vars['NumberofBlocks']}, 2x2 blocks, starting from top-left corner and going till the end.",
            "Each 2x2 block is expanded into a 4x4 block in the output grid.",
            "If a colored cell is present in a 2x2 block in the input grid, the corresponding 4x4 block in the output grid is entirely filled with the same color, otherwise its empty (0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Initialize task variables
        taskvars = {}
        
        # Random even grid size between 6 and 14
        taskvars['grid_size'] = random.randint(3, 7) * 2
        
        # Calculate number of 2x2 blocks
        taskvars['NumberofBlocks'] = (taskvars['grid_size'] * taskvars['grid_size']) // 4
        
        # Choose 4 different colors (1-9)
        colors = random.sample(range(1, 10), 4)
        taskvars['cell_color1'] = colors[0]
        taskvars['cell_color2'] = colors[1]
        taskvars['cell_color3'] = colors[2]
        taskvars['cell_color4'] = colors[3]
        
        # Generate 3-4 train examples and 1 test example
        num_train_examples = random.randint(3, 4)
        
        train_examples = []
        # Generate each training example
        for i in range(num_train_examples):
            # One of the training examples should have an extra cell with a different color
            add_extra_color = (i == 0)  # Add to first training example
            
            input_grid = self.create_input(taskvars, {'add_extra_color': add_extra_color})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example
        test_input = self.create_input(taskvars, {'add_extra_color': False})
        test_output = self.transform_input(test_input, taskvars)
        
        return taskvars, {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        add_extra_color = gridvars.get('add_extra_color', False)
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # List of valid positions (odd rows and columns only)
        valid_positions = [(r, c) for r in range(grid_size) for c in range(grid_size) 
                          if r % 2 == 1 and c % 2 == 1]
        
        # Shuffle valid positions
        random.shuffle(valid_positions)
        
        # Determine number of cells to color.
        # The count should vary between 2 and taskvars['grid_size'], but cannot exceed available valid positions.
        max_allowed = min(taskvars['grid_size'], len(valid_positions))
        num_colored_cells = random.randint(2, max_allowed) if max_allowed >= 2 else max_allowed

        # List of predefined colors to prefer using
        available_colors = [
            taskvars['cell_color1'],
            taskvars['cell_color2'],
            taskvars['cell_color3'],
            taskvars['cell_color4']
        ]

        colors_to_place = []

        # If we need at least 4 or more colored cells, ensure each predefined color appears at least once.
        if num_colored_cells >= 4:
            # start with the four unique colors
            colors_to_place.extend(available_colors)
            # fill the remaining slots by sampling from the predefined colors (allow repeats)
            for _ in range(num_colored_cells - 4):
                colors_to_place.append(random.choice(available_colors))
        else:
            # If fewer than 4 cells, choose a random subset of the predefined colors without repeats
            colors_to_place = random.sample(available_colors, num_colored_cells)

        # Shuffle the positions and the colors_to_place to randomize placement
        random.shuffle(colors_to_place)

        for i in range(min(num_colored_cells, len(valid_positions))):
            r, c = valid_positions[i]
            grid[r, c] = colors_to_place[i]

        # If add_extra_color requested, replace one additional free position (if any) with a color
        # that is not among the predefined colors.
        if add_extra_color:
            # find an extra position index after the ones we already used
            extra_pos_index = num_colored_cells
            if extra_pos_index < len(valid_positions):
                r, c = valid_positions[extra_pos_index]
                # Choose a random color (1-9) that's not one of the predefined colors
                extra_colors = [i for i in range(1, 10) if i not in available_colors]
                if extra_colors:
                    extra_color = random.choice(extra_colors)
                    grid[r, c] = extra_color
        
        return grid
    
    def transform_input(self, grid, taskvars):
        input_size = taskvars['grid_size']
        output_size = 2 * input_size
        
        # Create empty output grid
        output = np.zeros((output_size, output_size), dtype=int)
        
        # Process each 2x2 block in the input
        for r in range(0, input_size, 2):
            for c in range(0, input_size, 2):
                # Get the 2x2 block
                block = grid[r:r+2, c:c+2]
                
                # Find colored cells in the block
                colors = [color for color in block.flatten() if color != 0]
                
                # If any cells are colored, fill the corresponding 4x4 block
                if colors:
                    # Use the first colored cell's color (in case there are multiple)
                    block_color = colors[0]
                    
                    # Calculate the starting position of the 4x4 block in output
                    out_r = r * 2
                    out_c = c * 2
                    
                    # Fill the 4x4 block with the color
                    output[out_r:out_r+4, out_c:out_c+4] = block_color
        
        return output

