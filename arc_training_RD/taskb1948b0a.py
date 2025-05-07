from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, random_cell_coloring, Contiguity
from transformation_library import find_connected_objects, GridObject, GridObjects

class Taskb1948b0aGenerator(ARCTaskGenerator):
    def __init__(self):
        self.input_reasoning_chain = [
            "Input grids are of size MxN.",
            "The grid is completely filled with two colors namely {{color(\"object_color\")}} color and {{color(\"fill_color\")}} color.",
            "The objects can form any random shape."
        ]
        
        self.transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid",
            "The {{color(\"object_color\")}} color cells remain as it is but the {{color(\"fill_color\")}} color cells change to {{color(\"replace_color\")}} color color."
        ]
        
        taskvars_definitions = {
            "object_color": "object_color used in the grid",
            "fill_color": "fill_color that gets replaced",
            "replace_color": "replace_color color that replaces green"
        }
        
        super().__init__(self.input_reasoning_chain, self.transformation_reasoning_chain)
    
    def create_input(self, vars):
        # Get colors from vars
        object_color = vars["object_color"]
        fill_color = vars["fill_color"]
        
        # Create a grid of random size between 5x5 and 15x15
        rows = random.randint(5, 15)
        cols = random.randint(5, 15)
        
        # Initialize grid with the fill_color (now we'll add object_color instead of the reverse)
        grid = np.full((rows, cols), fill_color, dtype=int)
        
        # Determine coverage percentage for object_color (30-70% of the grid)
        coverage = random.uniform(0.3, 0.7)
        cells_to_color = int(rows * cols * coverage)
        
        # Create a mask of random positions for object_color
        positions = []
        for r in range(rows):
            for c in range(cols):
                positions.append((r, c))
        
        # Randomly select positions for object_color
        selected_positions = random.sample(positions, min(cells_to_color, len(positions)))
        
        # Apply object_color to selected positions
        for r, c in selected_positions:
            grid[r, c] = object_color
        
        # Ensure that objects have some coherent structure by applying cellular automaton rules
        # This creates more natural-looking shapes rather than pure noise
        for _ in range(3):  # Apply smoothing iterations
            new_grid = grid.copy()
            for r in range(rows):
                for c in range(cols):
                    # Count neighbors of object_color
                    neighbor_count = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == object_color:
                                neighbor_count += 1
                    
                    # Apply cellular automaton rule:
                    # - If a cell has many neighbors of object_color, it becomes object_color
                    # - If a cell has few neighbors of object_color, it becomes fill_color
                    if neighbor_count >= 5:  # More strict for becoming object_color
                        new_grid[r, c] = object_color
                    elif neighbor_count <= 2:  # More lenient for becoming fill_color
                        new_grid[r, c] = fill_color
            
            grid = new_grid
                    
        return grid
    
    def transform_input(self, input_grid, vars):
        output_grid = input_grid.copy()
        
        # Get colors from vars
        fill_color = vars["fill_color"]
        replace_color = vars["replace_color"]
        
        # Replace fill_color cells with replace_color
        output_grid[output_grid == fill_color] = replace_color
        
        return output_grid
    
    def create_grids(self):
        # Define random colors for the task (all different)
        colors = random.sample(range(1, 10), 3)
        taskvars = {
            "object_color": colors[0],
            "fill_color": colors[1],
            "replace_color": colors[2]
        }
        
        # Create random number of training examples
        num_train_examples = random.randint(3, 5)
        train_pairs = []
        
        for _ in range(num_train_examples):
            input_grid = self.create_input(taskvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Create test pair
        test_input = self.create_input(taskvars)
        test_output = self.transform_input(test_input, taskvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        # Return taskvars and TrainTestData object
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)


