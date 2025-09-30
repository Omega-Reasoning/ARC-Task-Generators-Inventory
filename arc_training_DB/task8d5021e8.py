from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity
from transformation_library import find_connected_objects, GridObject, GridObjects

class ARCTask8d5021e8Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid is a square grid with dimension {vars['rows']} X {vars['cols']}.",
            "There is one 4-way object present in the input grid of color in_color(between 1-9).",
            "The remaining cells are empty(0)"
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the has twice the width and three times the height of the input grid",
            "First copy the input grid to the output grid.",
            "Reflect the input grid along the y-axis at the left edge, and stack it horizontally along the left edge, called initial_grid.",
            "Reflect the initial_grid vertically along the x-axis at the bottom edge and stack it with initial_grid, called intermediate_grid.",
            "Stack the initial_grid along the bottom egde of the intermediate_grid, this is your output grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Setup task variables
        rows = random.randint(3, 10)  # Keeping smaller to ensure output is within 30x30
        cols = random.randint(3, 10)
        
        # Create 3-4 training examples with different colors
        num_train_examples = random.randint(3, 4)
        
        # Ensure we use different colors across examples
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        
        # Define task variables
        taskvars = {
            'rows': rows,
            'cols': cols
        }
        
        # Generate train and test data
        train_data = []
        for i in range(num_train_examples):
            in_color = available_colors[i % len(available_colors)]
            gridvars = {'in_color': in_color}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_data.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Test example
        test_in_color = available_colors[num_train_examples % len(available_colors)]
        test_gridvars = {'in_color': test_in_color}
        
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_data = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {'train': train_data, 'test': test_data}
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        in_color = gridvars['in_color']
        
        # Create a grid with a 4-way connected object
        # We'll keep trying until we get an object that doesn't fill the entire grid
        def generate_valid_grid():
            # Create random 4-way connected object
            grid = create_object(
                height=rows,
                width=cols,
                color_palette=in_color,
                contiguity=Contiguity.FOUR,
                background=0
            )
            
            # Check if the object doesn't fill the entire grid
            return grid if np.any(grid == 0) else None
        
        # Keep trying until we get a valid grid
        grid = None
        max_attempts = 100
        for _ in range(max_attempts):
            grid = generate_valid_grid()
            if grid is not None:
                break
        
        if grid is None:
            # Fallback - create grid where we ensure some empty cells
            grid = np.zeros((rows, cols), dtype=int)
            num_cells = rows * cols
            cells_to_fill = random.randint(1, num_cells - 1)  # Leave at least one empty cell
            
            # Start with a single cell for 4-way connectivity
            r, c = random.randint(0, rows-1), random.randint(0, cols-1)
            grid[r, c] = in_color
            filled_cells = 1
            
            # Keep adding adjacent cells
            while filled_cells < cells_to_fill:
                candidates = []
                for i in range(rows):
                    for j in range(cols):
                        if grid[i, j] == 0:
                            # Check if adjacent to existing object (4-way)
                            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                ni, nj = i + dr, j + dc
                                if 0 <= ni < rows and 0 <= nj < cols and grid[ni, nj] == in_color:
                                    candidates.append((i, j))
                                    break
                
                if not candidates:
                    break  # No more candidates to grow the object
                
                r, c = random.choice(candidates)
                grid[r, c] = in_color
                filled_cells += 1
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        
        # Output grid has twice the width and three times the height
        output = np.zeros((3 * rows, 2 * cols), dtype=int)
        
        # Step 1: Copy the input grid to top right of output grid
        output[:rows, cols:] = grid
        
        # Step 2: Reflect along y-axis (horizontally)
        for r in range(rows):
            for c in range(cols):
                output[r, cols-1-c] = grid[r, c]
        
        # Now we have the initial_grid (twice the width)
        initial_grid = output[:rows, :]
        
        # Step 3: Reflect initial_grid along x-axis (vertically)
        for r in range(rows):
            output[2*rows-1-r, :] = initial_grid[r, :]
        
        # Step 4: Copy initial_grid to bottom of output grid
        output[2*rows:, :] = initial_grid
        
        return output
