from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Taskae4f1146Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are 9x9 fixed size.",
            "The grid consists of sub-grids of 3x3 spread across the main grid, these sub-grids may or may not have patterns within them. These patterns can be just single cells or scattered cells, or an actual pattern.",
            "These sub-grids consists of two colors namely {{color(\"base_color\")}} color color which contributes to the base color of the sub-grid and {{color(\"pattern_color\")}} color which contributes to the formation of patterns within the sub-grid."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is 3x3 fixed size.",
            "The output grid is formed by identifying that one sub-grid from the input grid which has the maximum patterns."
        ]
        
        taskvars_definitions = {
            "grid_Size": "Size of the square grid 9x9.",
            "base_color": "Integer between 1 and 9", 
            "pattern_color": "Integer between 1 and 9"}
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, gridvars):
        """Create a 9x9 input grid with 4 sub-grids (3x3 each), with three distinct colors.
        Input grid is always 9x9 fixed size."""
        # Create a 9x9 grid filled with zeros
        grid = np.zeros((9, 9), dtype=int)
        
        # Get colors from gridvars
        base_color = gridvars.get('base_color', random.randint(1, 9))
        pattern_color = gridvars.get('pattern_color', random.randint(1, 9))
        
        # Ensure the two colors are different
        while pattern_color == base_color:
            pattern_color = random.randint(1, 9)
        
        # Create a 9x9 grid filled with main grid color
        grid = np.zeros((9, 9), dtype=int)
        
        # Define 4 fixed positions for sub-grids (top-left, top-right, bottom-left, bottom-right)
        subgrid_positions = [
            (0, 0),  # top-left
            (0, 6),  # top-right
            (6, 0),  # bottom-left
            (6, 6)   # bottom-right
        ]
        
        # Randomize the order (for variety)
        random.shuffle(subgrid_positions)
        
        # Create a random number of patterns in each 3x3 sub-grid
        subgrid_pattern_counts = []
        
        for idx, (start_i, start_j) in enumerate(subgrid_positions):
            # Fill the 3x3 subgrid with base_color first
            for i in range(3):
                for j in range(3):
                    grid[start_i + i, start_j + j] = base_color
            
            # Randomly decide density of pattern cells in this sub-grid
            pattern_density = random.uniform(0.1, 0.8)
            
            # Apply patterns to this 3x3 sub-grid
            pattern_count = 0
            for i in range(3):
                for j in range(3):
                    if random.random() < pattern_density:
                        grid[start_i + i, start_j + j] = pattern_color
                        pattern_count += 1
            
            subgrid_pattern_counts.append(pattern_count)
        
        return grid
    
    def transform_input(self, inp, gridvars):
        """Extract the 3x3 sub-grid with the maximum number of pattern cells.
        Output grid is always 3x3 fixed size."""
        # Remove grid_size initialization since output is always 3x3
        base_color = gridvars['base_color']
        pattern_color = gridvars['pattern_color']
        
        # Identify the colors if not provided
        if base_color is None or pattern_color is None:
            # Get non-zero colors (excluding background)
            non_zero_colors = [col for col in np.unique(inp) if col != 0]
            
            if len(non_zero_colors) >= 2:
                # Count occurrences of each color
                counts = {color: np.sum(inp == color) for color in non_zero_colors}
                # Sort colors by frequency
                sorted_colors = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                
                # The most common color is likely the subgrid base color
                # The second most common is likely the pattern color
                base_color = sorted_colors[0][0]
                pattern_color = sorted_colors[1][0]
            elif len(non_zero_colors) == 1:
                # If only one color is present, use it as both
                base_color = non_zero_colors[0]
                pattern_color = non_zero_colors[0]
            else:
                # Default fallback (should not happen)
                base_color = 1
                pattern_color = 2
        
        # Define the positions of the 4 sub-grids
        subgrid_positions = [
            (0, 0),  # top-left
            (0, 6),  # top-right
            (6, 0),  # bottom-left
            (6, 6)   # bottom-right
        ]
        
        # Find the sub-grid with the maximum pattern cells
        max_patterns = -1
        max_subgrid = None
        
        for start_i, start_j in subgrid_positions:
            subgrid = inp[start_i:start_i+3, start_j:start_j+3]
            
            # Skip empty subgrids
            if np.all(subgrid == 0):
                continue
                
            pattern_count = np.sum(subgrid == pattern_color)
            
            if pattern_count > max_patterns:
                max_patterns = pattern_count
                max_subgrid = subgrid.copy()
        
        # If no patterns found, return an arbitrary non-empty subgrid
        if max_subgrid is None:
            for start_i, start_j in subgrid_positions:
                subgrid = inp[start_i:start_i+3, start_j:start_j+3]
                if not np.all(subgrid == 0):
                    max_subgrid = subgrid.copy()
                    break
        
        # If still no subgrid found (very unlikely), create one
        if max_subgrid is None:
            max_subgrid = np.full((3, 3), base_color)
            
        return max_subgrid


    def create_grids(self):
        """Create training and test pairs with diverse patterns in 9x9 input grids"""
        color_pool = list(range(1, 10))
        random.shuffle(color_pool)
        
        base_color = color_pool[0]
        pattern_color = color_pool[1]
            
        gridvars = {
            'base_color': base_color, 
            'pattern_color': pattern_color
        }   
        
        # Generate 3-5 training pairs
        num_train = random.randint(3, 5)
        train_pairs = []
        
        for _ in range(num_train):
            input_grid = self.create_input(gridvars)    
            output_grid = self.transform_input(input_grid, gridvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Create test pair
        test_input = self.create_input(gridvars)
        test_output = self.transform_input(test_input, gridvars)
        test_examples = [GridPair(input=test_input, output=test_output)]
        
        # Create task variables for display
        taskvars = {
            'input_grid_size': '9x9',  # Input grid size
            'base_color': base_color,
            'pattern_color': pattern_color,
            'output_grid_size': '3x3'  # Output grid size
        }
        
        return taskvars, TrainTestData(train=train_pairs, test=test_examples)


