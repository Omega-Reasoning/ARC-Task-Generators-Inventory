from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Task54d9e175Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes, with the number of rows being (4n − 1) and the number of columns being (4m − 1), where m and n are integers greater than 1 and less than 8.",
            "They contain {color('divider_color')} completely filled rows and columns inserted after every 3 rows and columns, acting as dividers.",
            "Once the dividers have been added, the grid is divided into multiple 3x3 subgrids, with each subgrid having its exact center cell colored.",
            "The center cells can only use one of the following colors: {color('cell_color1')}, {color('cell_color2')}, {color('cell_color3')}, or {color('cell_color4')}.",
            "All other cells remain empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and filling each 3x3 subgrid separated by {color('divider_color')} dividers.",
            "Each 3x3 subgrid is filled entirely with a color chosen based on the color of its central cell in the input grid.",
            "The {color('cell_color1')}, {color('cell_color2')}, {color('cell_color3')}, and {color('cell_color4')} change to {color('cell_color5')}, {color('cell_color6')}, {color('cell_color7')}, and {color('cell_color8')} respectively.",
            "The {color('divider_color')}  dividers remain unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Initialize task variables
        color_values = random.sample(range(1, 10), 9)  # Select 9 unique colors
        
        taskvars = {
            'divider_color': color_values[0],
            'cell_color1': color_values[1],
            'cell_color2': color_values[2],
            'cell_color3': color_values[3],
            'cell_color4': color_values[4],
            'cell_color5': color_values[5],
            'cell_color6': color_values[6],
            'cell_color7': color_values[7],
            'cell_color8': color_values[8]
        }
        
        # Generate random number of train examples (3-6)
        nr_train_examples = random.randint(3, 6)
        
        # Create train and test data
        return taskvars, self.create_grids_default(nr_train_examples, 1, taskvars)
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        # Determine grid dimensions (4n-1) x (4m-1)
        n = random.randint(2, 7)  # n between 2 and 7
        m = random.randint(2, 7)  # m between 2 and 7
        
        rows = 4 * n - 1
        cols = 4 * m - 1
        
        # Initialize empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Add dividers after every 3 rows and columns
        for i in range(3, rows, 4):
            grid[i, :] = taskvars['divider_color']
        
        for j in range(3, cols, 4):
            grid[:, j] = taskvars['divider_color']
        
        # Add colored center cells in each 3x3 subgrid
        center_colors = [
            taskvars['cell_color1'], 
            taskvars['cell_color2'], 
            taskvars['cell_color3'], 
            taskvars['cell_color4']
        ]
        
        for i in range(n):
            for j in range(m):
                # Calculate center position of each 3x3 subgrid
                center_row = i * 4 + 1
                center_col = j * 4 + 1
                
                # Set center cell to a random color from the color options
                grid[center_row, center_col] = random.choice(center_colors)
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        # Create a copy of input grid
        output = grid.copy()
        
        # Mapping from input colors to output colors
        color_mapping = {
            taskvars['cell_color1']: taskvars['cell_color5'],
            taskvars['cell_color2']: taskvars['cell_color6'],
            taskvars['cell_color3']: taskvars['cell_color7'],
            taskvars['cell_color4']: taskvars['cell_color8']
        }
        
        rows, cols = grid.shape
        
        # Process each 3x3 subgrid
        for i in range(0, rows, 4):
            for j in range(0, cols, 4):
                # Skip if we're out of bounds
                if i + 2 >= rows or j + 2 >= cols:
                    continue
                
                # Get center cell color
                center_color = grid[i + 1, j + 1]
                
                # Skip if center is not one of our colors
                if center_color not in color_mapping:
                    continue
                
                # Fill the 3x3 subgrid with the mapped color
                output_color = color_mapping[center_color]
                
                for r in range(3):
                    for c in range(3):
                        output[i + r, j + c] = output_color
        
        return output

