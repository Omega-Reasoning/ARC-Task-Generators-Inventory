from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
import numpy as np
import random

class HorizontalLineToTriangleTaskGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain one horizontal line of {color(\"object_color\")} color and empty (0) cells.",
            "The horizontal line always starts from column 1 of any row.",
            "The size of the horizontal line across the grids varies."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by transforming the input grid into an inverted right-angled triangle.",
            "The horizontal line from the input remains unchanged and forms the widest part of the triangle.",
            "Above the line, cells are filled with {color(\"fill_color1\")} to form the top part of the triangle, with each row getting wider as it approaches the horizontal line.",
            "Below the line, cells are filled with {color(\"fill_color2\")} to form the bottom part of the triangle, with each row getting narrower as it moves away from the horizontal line."
        ]
        
        taskvars_definitions = {
            "object_color": "Color of the horizontal line (between 1 and 9)",
            "fill_color1": "Color to fill above the line (between 1 and 9)",
            "fill_color2": "Color to fill below the line (between 1 and 9)"
        }
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, gridvars=None):
        if gridvars is None:
            gridvars = {}
        
        # Random grid dimensions, within specified constraints
        height = random.randint(5, 20)
        width = random.randint(5, 20)
        
        # Get colors from gridvars or generate random colors
        object_color = gridvars.get('object_color', random.randint(1, 9))
        
        # Create an empty grid
        grid = np.zeros((height, width), dtype=int)
        
        # Randomly choose a row for the horizontal line
        line_row = random.randint(1, height - 2)  # Avoid placing at the very top or bottom
        
        # Decide the length of the horizontal line (from column 1 to a random endpoint)
        line_length = random.randint(max(3, width // 4), width - 1)
        
        # Draw the horizontal line
        for col in range(1, line_length + 1):
            grid[line_row, col] = object_color
        
        return grid
    
    def transform_input(self, input_grid, gridvars=None):
        if gridvars is None:
            gridvars = {}
        
        # Get the fill colors from gridvars
        fill_color1 = gridvars.get('fill_color1', random.randint(1, 9))
        fill_color2 = gridvars.get('fill_color2', random.randint(1, 9))
        
        # Create a completely new output grid filled with zeros
        output_grid = np.zeros_like(input_grid)
        
        # Find the horizontal line
        objects = find_connected_objects(input_grid)
        
        if len(objects) == 0:
            return output_grid  # No objects found, return empty grid
        
        # Assuming the horizontal line is the only object
        line_object = objects[0]
        
        # Get all the coordinates of the line to find the row it's on and its length
        line_coords = line_object.coords
        line_row = min(r for r, _ in line_coords)  # The row of the line
        line_length = max(c for _, c in line_coords)  # The rightmost column of the line
        
        # Copy the horizontal line to the output grid (preserving its original color)
        for r, c, color in line_object:
            output_grid[r, c] = color
        
        # Calculate how many rows we need above and below
        # We want only enough rows to make one complete triangle
        rows_above = min(line_row, line_length - 1)  # At most line_length-1 rows above
        rows_below = min(input_grid.shape[0] - line_row - 1, line_length - 1)  # At most line_length-1 rows below
        
        # Fill above the line with fill_color1 (top part of triangle)
        for offset in range(1, rows_above + 1):
            row = line_row - offset
            # The width decreases as we go up
            width = line_length - offset
            
            for col in range(1, width + 1):
                output_grid[row, col] = fill_color1
        
        # Fill below the line with fill_color2 (bottom part of triangle)
        for offset in range(1, rows_below + 1):
            row = line_row + offset
            # The width decreases as we go down
            width = line_length - offset
            
            for col in range(1, width + 1):
                output_grid[row, col] = fill_color2
        
        return output_grid
    
    def create_grids(self):
        # Generate 3-5 pairs of training grids with consistent colors
        num_train_pairs = random.randint(3, 5)
        
        # First, choose the colors to use consistently across examples
        object_color = random.randint(1, 9)
        
        # Make sure fill colors are different from object color and from each other
        fill_color1 = random.choice([c for c in range(1, 10) if c != object_color])
        fill_color2 = random.choice([c for c in range(1, 10) if c != object_color and c != fill_color1])
        
        gridvars = {
            'object_color': object_color,
            'fill_color1': fill_color1,
            'fill_color2': fill_color2
        }
        
        # Generate train pairs
        train_pairs = []
        for _ in range(num_train_pairs):
            input_grid = self.create_input(gridvars)
            output_grid = self.transform_input(input_grid, gridvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate a test pair with the same colors but different dimensions/layout
        test_input = self.create_input(gridvars)
        test_output = self.transform_input(test_input, gridvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return gridvars, TrainTestData(train=train_pairs, test=test_pairs)

# Test the task generator
if __name__ == "__main__":
    generator = HorizontalLineToTriangleTaskGenerator()
    gridvars, train_test_data = generator.create_grids()
    print(f"Object color: {gridvars['object_color']}")
    print(f"Fill color 1: {gridvars['fill_color1']}")
    print(f"Fill color 2: {gridvars['fill_color2']}")
    ARCTaskGenerator.visualize_train_test_data(train_test_data)