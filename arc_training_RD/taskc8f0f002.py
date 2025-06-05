from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import create_object, Contiguity, retry

class Taskc8f0f002Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids have exactly 3 rows and varying column widths.",
            "The grid is fully filled with exactly three patterns using {color('pattern_color1')}, {color('pattern_color2')}, and {color('pattern_color3')} colors."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.",
            "One of the three pattern colors is changed to {color('replace_color')} in the output grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        # Fixed row size of 3, varying column width
        height = 3
        width = random.randint(4, 12)  # Random width between 4-12
        
        # Get the three pattern colors from taskvars
        pattern_color1 = taskvars['pattern_color1']
        pattern_color2 = taskvars['pattern_color2']
        pattern_color3 = taskvars['pattern_color3']
        colors = [pattern_color1, pattern_color2, pattern_color3]
        
        # Initialize grid with background
        grid = np.zeros((height, width), dtype=int)
        
        # Create rectangular block patterns with exactly 3 colors
        self._create_block_patterns(grid, colors, height, width)
        
        return grid
    
    def _create_block_patterns(self, grid, colors, height, width):
        """Create rectangular block patterns with exactly 3 colors in a 3-row grid"""
        
        # Since we only have 3 rows, use smaller block sizes
        block_height = 1  # Single row blocks work well for 3-row grids
        block_width = random.randint(1, 2)
        
        # Fill grid with rectangular blocks using only the 3 colors
        color_index = 0
        for r in range(0, height, block_height):
            for c in range(0, width, block_width):
                # Calculate actual block boundaries
                end_r = min(r + block_height, height)
                end_c = min(c + block_width, width)
                
                # Assign color to this block (cycle through the 3 colors)
                current_color = colors[color_index % 3]
                grid[r:end_r, c:end_c] = current_color
                
                # Move to next color
                color_index += 1
        
        # Add some variation by creating larger blocks randomly
        num_large_blocks = random.randint(1, 2)  # Fewer large blocks for smaller grid
        for _ in range(num_large_blocks):
            # Create a larger block (max 2x2 or 3x2 for 3-row grid)
            large_height = random.randint(1, 2)  # Max 2 rows for variation
            large_width = random.randint(2, min(3, width//2))
            
            # Find random position for the large block
            start_r = random.randint(0, max(1, height - large_height))
            start_c = random.randint(0, max(1, width - large_width))
            
            # Choose one of the 3 colors
            block_color = random.choice(colors)
            grid[start_r:start_r+large_height, start_c:start_c+large_width] = block_color
    
    def transform_input(self, input_grid, taskvars):
        # Copy the input grid
        output_grid = np.copy(input_grid)
        
        # Get the colors and replacement color from taskvars
        pattern_colors = [taskvars['pattern_color1'], taskvars['pattern_color2'], taskvars['pattern_color3']]
        replace_color = taskvars['replace_color']
        
        # Choose one of the three pattern colors to replace
        old_color = random.choice(pattern_colors)
        
        # Replace all instances of old_color with replace_color
        output_grid[input_grid == old_color] = replace_color
        
        return output_grid
    
    def create_grids(self):
        # Choose 3 distinct colors for patterns
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        
        pattern_color1 = available_colors[0]
        pattern_color2 = available_colors[1]
        pattern_color3 = available_colors[2]
        
        # Choose replacement color (different from the 3 pattern colors)
        replace_color = available_colors[3]
        
        # Create task variables
        taskvars = {
            'pattern_color1': pattern_color1,
            'pattern_color2': pattern_color2,
            'pattern_color3': pattern_color3,
            'replace_color': replace_color
        }
        
        # Helper for color formatting
        def color_name(color_id):
            color_map = {
                0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow", 
                5: "gray", 6: "magenta", 7: "orange", 8: "cyan", 9: "brown"
            }
            return color_map.get(color_id, f"color_{color_id}")
        
        def color_fmt(key):
            color_id = taskvars[key]
            return f"{color_name(color_id)} ({color_id})"
        
        # Replace placeholders in reasoning chains
        self.input_reasoning_chain = [
            chain.replace("{color('pattern_color1')}", color_fmt('pattern_color1'))
                 .replace("{color('pattern_color2')}", color_fmt('pattern_color2'))
                 .replace("{color('pattern_color3')}", color_fmt('pattern_color3'))
            for chain in self.input_reasoning_chain
        ]
        self.transformation_reasoning_chain = [
            chain.replace("{color('replace_color')}", color_fmt('replace_color'))
            for chain in self.transformation_reasoning_chain
        ]
        
        # Generate 3-5 training examples
        num_train = random.randint(3, 5)
        train_pairs = []
        
        for _ in range(num_train):
            gridvars = {}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate one test example
        test_gridvars = {}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)
