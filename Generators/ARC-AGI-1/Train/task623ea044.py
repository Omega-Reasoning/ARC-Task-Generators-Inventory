from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import retry
from transformation_library import find_connected_objects

class Task623ea044Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} x {vars['cols']}.",
            "They contain exactly one colored (1–9) cell, with all other cells being empty (0).",
            "This colored cell is never placed on the border and is always located in the interior of the grid.",
            "Both the color and position of the cell vary across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying the single-colored cell.",
            "This cell is then extended diagonally in all four directions using the same color.",
            "The diagonal extension continues in each direction until it reaches the edge of the grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Define the task variables
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)
        
        taskvars = {
            'rows': rows,
            'cols': cols
        }
        
        # Generate 4 training examples and 1 test example
        train_examples = []
        
        # Keep track of used colors and positions to ensure they're different
        used_colors = set()
        used_positions = set()
        
        # Create 4 training examples
        for _ in range(4):
            # Choose a different color for each example
            color = retry(
                lambda: random.randint(1, 9),
                lambda c: c not in used_colors
            )
            used_colors.add(color)
            
            # Create grid variables for this example
            gridvars = {'color': color, 'used_positions': used_positions}
            
            # Create input grid and transform it
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            # Update used positions
            objects = find_connected_objects(input_grid)
            if objects.objects:
                r, c, _ = next(iter(objects[0].cells))
                used_positions.add((r, c))
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create 1 test example with a different color and position
        test_color = retry(
            lambda: random.randint(1, 9),
            lambda c: c not in used_colors
        )
        test_gridvars = {'color': test_color, 'used_positions': used_positions}
        
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
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        color = gridvars['color']
        used_positions = gridvars.get('used_positions', set())
        
        # Create an empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Select a random position for the colored cell, avoiding borders and used positions
        def generate_position():
            r = random.randint(1, rows - 2)  # Avoid borders
            c = random.randint(1, cols - 2)  # Avoid borders
            return (r, c)
        
        position = retry(
            generate_position,
            lambda pos: pos not in used_positions,
            max_attempts=100
        )
        
        # Place the colored cell
        r, c = position
        grid[r, c] = color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        rows, cols = grid.shape
        output_grid = grid.copy()
        
        # Find the colored cell
        colored_cells = np.where(grid > 0)
        if len(colored_cells[0]) == 0:
            return output_grid  # No colored cells, return unchanged
        
        r, c = colored_cells[0][0], colored_cells[1][0]
        color = grid[r, c]
        
        # Extend diagonally in all four directions
        # Top-left direction
        i, j = r - 1, c - 1
        while i >= 0 and j >= 0:
            output_grid[i, j] = color
            i -= 1
            j -= 1
        
        # Top-right direction
        i, j = r - 1, c + 1
        while i >= 0 and j < cols:
            output_grid[i, j] = color
            i -= 1
            j += 1
        
        # Bottom-left direction
        i, j = r + 1, c - 1
        while i < rows and j >= 0:
            output_grid[i, j] = color
            i += 1
            j -= 1
        
        # Bottom-right direction
        i, j = r + 1, c + 1
        while i < rows and j < cols:
            output_grid[i, j] = color
            i += 1
            j += 1
        
        return output_grid
