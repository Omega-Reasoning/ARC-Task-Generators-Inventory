from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, random_cell_coloring, Contiguity
from transformation_library import find_connected_objects, GridObject

class ARCTask99b1bc43Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has dimension {vars['rows']} X {vars['cols']}.",
            "The grid is divided into two parts by a row which has cells of color {color('divider')}.",
            "The top part of the grid has a 4-way connected object of color {color('top_color')}.",
            "The top part of the grid has a 4-way connected object of color {color('bottom_color')}.",
            "The remaining cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has dimensions {vars['rows']}/2 - 1 * {vars['cols']}.",
            "The output grid cell is assigned {color('fill')} exactly when its top and bottom half colors differ (i.e., top half is colored and bottom half is not, or vice versa)—in other words, the coloring is determined by the XOR of the top and bottom halves.",
            "The remaining cells are empty(0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Initialize task variables with random values
        taskvars = {
            'rows': random.choice([11, 13, 15]),  # Odd numbers between 10-15
            'cols': random.randint(10, 15),
            'divider': random.choice([5, 6, 7, 8, 9]),  # Different color for divider
            'top_color': random.choice([1, 2, 3, 4]),  # Color for top object
            'bottom_color': random.choice([1, 2, 3, 4]),  # Color for bottom object
            'fill': random.choice([1, 2, 3, 4])  # Color for output grid
        }
        
        # Ensure all colors are different
        while len(set([taskvars['divider'], taskvars['top_color'], taskvars['bottom_color'], taskvars['fill']])) < 4:
            taskvars['divider'] = random.choice([5, 6, 7, 8, 9])
            taskvars['top_color'] = random.choice([1, 2, 3, 4])
            taskvars['bottom_color'] = random.choice([1, 2, 3, 4])
            taskvars['fill'] = random.choice([1, 2, 3, 4])
        
        # Generate 3-4 training examples and 1 test example
        num_train = random.randint(3, 4)
        
        train_data = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})
        
        test_data = []
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_data.append({'input': test_input, 'output': test_output})
        
        return taskvars, {'train': train_data, 'test': test_data}
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        divider_color = taskvars['divider']
        top_color = taskvars['top_color']
        bottom_color = taskvars['bottom_color']
        
        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Calculate divider row (roughly in the middle)
        divider_row = rows // 2
        
        # Fill divider row with divider color
        grid[divider_row, :] = divider_color
        
        # Create top object
        top_height = divider_row
        top_object = create_object(
            height=top_height,
            width=cols,
            color_palette=top_color,
            contiguity=Contiguity.FOUR,
            background=0
        )
        
        # Create bottom object
        bottom_height = rows - divider_row - 1
        bottom_object = create_object(
            height=bottom_height,
            width=cols,
            color_palette=bottom_color,
            contiguity=Contiguity.FOUR,
            background=0
        )
        
        # Place top object in the top part of the grid
        for r in range(top_height):
            for c in range(cols):
                if top_object[r, c] != 0:
                    grid[r, c] = top_object[r, c]
        
        # Place bottom object in the bottom part of the grid
        for r in range(bottom_height):
            for c in range(cols):
                if bottom_object[r, c] != 0:
                    grid[divider_row + 1 + r, c] = bottom_object[r, c]
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        divider_color = taskvars['divider']
        fill_color = taskvars['fill']
        
        # Find the divider row
        divider_row = None
        for r in range(rows):
            if np.all(grid[r, :] == divider_color):
                divider_row = r
                break
        
        if divider_row is None:
            # Fallback if divider row not found
            divider_row = rows // 2
        
        # Calculate output grid dimensions - rows/2 - 1 * cols
        output_rows = (rows // 2) - 1
        output_grid = np.zeros((output_rows, cols), dtype=int)
        
        # Apply XOR logic between top and bottom halves
        for r in range(output_rows):
            for c in range(cols):
                top_half = grid[r, c] != 0
                bottom_half = grid[divider_row + 1 + r, c] != 0
                
                # XOR operation
                if top_half != bottom_half:
                    output_grid[r, c] = fill_color
        
        return output_grid