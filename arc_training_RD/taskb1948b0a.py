from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Taskb1948b0aGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "The grid is completely filled with two colors namely {color('object_color')} and {color('fill_color')}.",
            "The objects can form any random shape."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid",
            "The {color('object_color')} cells remain as it is but the {color('fill_color')} cells change to {color('replace_color')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create a grid filled with two colors in random shapes."""
        # Get colors from taskvars
        object_color = taskvars["object_color"]
        fill_color = taskvars["fill_color"]
        
        # Create a grid of random size between 5x5 and 15x15
        rows = random.randint(5, 15)
        cols = random.randint(5, 15)
        
        # Initialize grid with the fill_color
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

        # Ensure at least one object_color cell is present
        if not np.any(grid == object_color):
            rand_r = random.randint(0, rows - 1)
            rand_c = random.randint(0, cols - 1)
            grid[rand_r, rand_c] = object_color

        return grid
    
    def transform_input(self, grid, taskvars):
        """Transform input by replacing fill_color cells with replace_color."""
        output_grid = grid.copy()
        
        # Get colors from taskvars
        fill_color = taskvars["fill_color"]
        replace_color = taskvars["replace_color"]
        
        # Replace fill_color cells with replace_color
        output_grid[output_grid == fill_color] = replace_color
        
        return output_grid

    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        # Define random colors for the task (all different)
        colors = random.sample(range(1, 10), 3)
        taskvars = {
            "object_color": colors[0],
            "fill_color": colors[1],
            "replace_color": colors[2]
        }

        # Generate 3-5 training examples with same task variables
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        for _ in range(num_train_examples):
            gridvars = {}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example with same task variables
        test_gridvars = {}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }

# Test code
if __name__ == "__main__":
    generator = Taskb1948b0aGenerator()
    taskvars, train_test_data = generator.create_grids()
    
    print("Task Variables:", taskvars)
    ARCTaskGenerator.visualize_train_test_data(train_test_data)