from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject, GridObjects
import numpy as np
import random

class Taskb9b7f026Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} × {vars['m']}.", 
            "The grid contains multiple rectangular blocks of different colors (between 1 to 9).", 
            "Exactly one of these rectangles contains a square hole (2x2, 3x3, or 4x4) inside it.",
            "The other rectangles are solid (no holes)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is always 1x1.",
            "Output the color of the rectangle that contains the hole."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict):
        # Use the dimensions from taskvars
        grid_height = taskvars['n']
        grid_width = taskvars['m']
        
        # Create empty grid
        grid = np.zeros((grid_height, grid_width), dtype=int)
        
        # Number of rectangles (3-5)
        num_rectangles = random.randint(3, 5)
        
        # Choose random colors for rectangles
        colors = random.sample(range(1, 10), num_rectangles)
        
        # Decide which rectangle will have the hole
        hole_rectangle_index = random.randint(0, num_rectangles - 1)
        
        # Place rectangles in a simple grid layout
        rectangles_per_row = int(np.ceil(np.sqrt(num_rectangles)))
        spacing = 2
        
        available_height = (grid_height - spacing * (rectangles_per_row + 1)) // rectangles_per_row
        available_width = (grid_width - spacing * (rectangles_per_row + 1)) // rectangles_per_row
        
        rect_index = 0
        for i in range(rectangles_per_row):
            for j in range(rectangles_per_row):
                if rect_index >= num_rectangles:
                    break
                
                # Determine rectangle size
                if rect_index == hole_rectangle_index:
                    # Make sure this one is large enough for a hole (at least 8x8)
                    min_size = 8
                    max_size = min(12, available_height, available_width)
                    if max_size < min_size:
                        max_size = min_size
                    rect_height = random.randint(min_size, max_size)
                    rect_width = random.randint(min_size, max_size)
                else:
                    # Other rectangles can be smaller
                    min_size = 3
                    max_size = min(8, available_height, available_width)
                    if max_size < min_size:
                        max_size = min_size
                    rect_height = random.randint(min_size, max_size)
                    rect_width = random.randint(min_size, max_size)
                
                # Calculate position
                rect_row = spacing + i * (available_height + spacing)
                rect_col = spacing + j * (available_width + spacing)
                
                # Make sure we don't go out of bounds
                if rect_row + rect_height >= grid_height:
                    rect_height = grid_height - rect_row - 1
                if rect_col + rect_width >= grid_width:
                    rect_width = grid_width - rect_col - 1
                
                if rect_height < 3 or rect_width < 3:
                    continue
                
                color = colors[rect_index]
                
                # Fill the rectangle
                for r in range(rect_row, rect_row + rect_height):
                    for c in range(rect_col, rect_col + rect_width):
                        grid[r, c] = color
                
                # Add hole if this is the chosen rectangle
                if rect_index == hole_rectangle_index and rect_height >= 8 and rect_width >= 8:
                    # Determine hole size: 2x2, 3x3, or 4x4
                    max_hole_size = min(rect_height - 4, rect_width - 4, 4)
                    possible_sizes = [s for s in [2, 3, 4] if s <= max_hole_size]
                    hole_size = random.choice(possible_sizes) if possible_sizes else 2
                    
                    # Position the hole at least 2 cells away from rectangle edges
                    max_hole_row = rect_row + rect_height - hole_size - 2
                    max_hole_col = rect_col + rect_width - hole_size - 2
                    
                    if max_hole_row > rect_row + 2 and max_hole_col > rect_col + 2:
                        hole_row = random.randint(rect_row + 2, max_hole_row)
                        hole_col = random.randint(rect_col + 2, max_hole_col)
                        
                        # Create the hole
                        for r in range(hole_row, hole_row + hole_size):
                            for c in range(hole_col, hole_col + hole_size):
                                grid[r, c] = 0
                
                rect_index += 1
        
        return grid
    
    def transform_input(self, input_grid, taskvars):
        # Find connected objects in the grid
        objects = find_connected_objects(input_grid, diagonal_connectivity=False, background=0)
        rows, cols = input_grid.shape
        
        hollow_color = None
        
        # Examine each object to find the one with a hollow region
        for obj in objects.objects:
            # Get the object's color
            color = next(iter(obj.colors))
            if color == 0:
                continue
                
            # Get object's bounding box
            min_r = min(r for r, c, _ in obj.cells)
            max_r = max(r for r, c, _ in obj.cells)
            min_c = min(c for r, c, _ in obj.cells)
            max_c = max(c for r, c, _ in obj.cells)
            
            # Check interior cells for hollow region
            found_hollow = False
            for r in range(min_r + 1, max_r):
                for c in range(min_c + 1, max_c):
                    if input_grid[r, c] == 0:  # Empty cell
                        # Check if surrounded by this object's color
                        if r > 0 and r < rows - 1 and c > 0 and c < cols - 1:
                            surrounding_cells = [
                                input_grid[r-1, c-1], input_grid[r-1, c], input_grid[r-1, c+1],
                                input_grid[r, c-1],                        input_grid[r, c+1],
                                input_grid[r+1, c-1], input_grid[r+1, c], input_grid[r+1, c+1]
                            ]
                            
                            if all(cell == color for cell in surrounding_cells):
                                hollow_color = color
                                found_hollow = True
                                break
                if found_hollow:
                    break
            
            if found_hollow:
                break
        
        if hollow_color is None:
            raise ValueError("No shape with hollow region found in the input grid")
        
        # Create 1x1 output grid with the hollow rectangle's color
        output_grid = np.array([[hollow_color]], dtype=int)
        return output_grid
    
    def create_grids(self):
        # Set larger fixed grid dimensions for all examples to ensure shapes fit
        grid_height = random.randint(25, 30)
        grid_width = random.randint(25, 30)
        
        # General taskvars that apply to all examples
        general_taskvars = {
            'n': grid_height,
            'm': grid_width
        }
        
        # Generate 3-5 train examples
        num_train = random.randint(3, 5)
        train_pairs = []
        
        for _ in range(num_train):
            gridvars = {}
            input_grid = self.create_input(general_taskvars, gridvars)
            output_grid = self.transform_input(input_grid, general_taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Create test example
        test_gridvars = {}
        test_input = self.create_input(general_taskvars, test_gridvars)
        test_output = self.transform_input(test_input, general_taskvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return general_taskvars, TrainTestData(train=train_pairs, test=test_pairs)


# Test code
if __name__ == "__main__":
    generator = Taskb9b7f026Generator()
    taskvars, train_test_data = generator.create_grids()
    
    print("Task Variables:", taskvars)
    ARCTaskGenerator.visualize_train_test_data(train_test_data)