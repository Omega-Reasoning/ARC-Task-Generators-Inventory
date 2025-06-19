from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Taska5f85a15Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are squares of size {vars['rows']}x{vars['cols']}.",
            "They contain one or more same-colored diagonal lines, with the remaining cells being empty (0).",
            "The diagonal lines are in the direction which is parallel to the main diagonal (top-left to bottom-right).",
            "The diagonal lines may not necessarily be on the main diagonal but parallel to it.",
            "Incase of more than one diagonal line, all diagonal lines must be completely separated from each other by having at least one empty (0) diagonal line between two consecutive colored diagonal lines.",
            "If there are more than one diagonal lines, the color of each line should be the same within the grid but must vary across grids."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying the colored diagonal lines.",
            "Once identified, for each diagonal line, start from the second cell and change its color to a different color (color 6).", 
            "Repeat this process by changing the color of all alternating cells along the diagonal line, continuing until no more cells can be recolored."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Create 3-4 training examples
        num_train_examples = random.randint(3, 4)
        train_pairs = []
        
        # Generate different grid sizes for each example
        train_grid_sizes = random.sample(range(15, 25), num_train_examples)
        
        # Define test grid size for taskvars
        test_grid_size = random.randint(15, 20)
        
        # Initialize task variables with test grid size
        taskvars = {
            'rows': test_grid_size,
            'cols': test_grid_size
        }
        
        # Create training examples
        for i in range(num_train_examples):
            # Generate a random line color for this grid (not 6)
            available_colors = [c for c in range(1, 10) if c != 6]
            line_color = random.choice(available_colors)
            
            gridvars = {
                "size": train_grid_sizes[i],
                "line_color": line_color,
                "has_main_diag": random.choice([True, False]),
                "num_diags": random.randint(1, 3)
            }
            
            input_grid = self.create_input(gridvars)
            output_grid = self.transform_input(input_grid)
            
            # Use GridPair instead of dictionary
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Create test example with the size specified in taskvars
        # Generate random colors for the test example (not 6)
        available_colors = [c for c in range(1, 10) if c != 6]
        test_line_color = random.choice(available_colors)
        
        test_gridvars = {
            "size": test_grid_size,
            "line_color": test_line_color,
            "has_main_diag": random.choice([True, False]),
            "num_diags": random.randint(1, 3)
        }
        
        test_input = self.create_input(test_gridvars)
        test_output = self.transform_input(test_input)
        
        # Use GridPair for test example
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        # Return TrainTestData object
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)
    
    def create_input(self, gridvars: dict[str, any]) -> np.ndarray:
        # Extract variables
        grid_size = gridvars["size"]
        line_color = gridvars["line_color"]
        has_main_diag = gridvars["has_main_diag"]
        num_diags = gridvars["num_diags"]
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Find valid diagonal offsets that would give at least 3 cells on each diagonal
        valid_offsets = [k for k in range(-grid_size+3, grid_size-2) if abs(k) <= grid_size-3]
        
        # If we need the main diagonal, make sure 0 is in the selected offsets
        if has_main_diag:
            offsets = [0]
            if 0 in valid_offsets:
                valid_offsets.remove(0)
            num_diags -= 1
        else:
            offsets = []
        
        # Ensure diagonal lines are separated by at least one empty diagonal
        if num_diags > 0:
            # If we already have the main diagonal, exclude adjacent diagonals
            if has_main_diag:
                if 1 in valid_offsets:
                    valid_offsets.remove(1)
                if -1 in valid_offsets:
                    valid_offsets.remove(-1)
            
            # Select remaining offsets ensuring separation
            while num_diags > 0 and valid_offsets:
                offset = random.choice(valid_offsets)
                offsets.append(offset)
                valid_offsets.remove(offset)
                
                # Remove adjacent diagonals to ensure separation
                if offset+1 in valid_offsets:
                    valid_offsets.remove(offset+1)
                if offset-1 in valid_offsets:
                    valid_offsets.remove(offset-1)
                
                num_diags -= 1
        
        # Draw the diagonal lines
        for offset in offsets:
            if offset >= 0:
                # Diagonals below or on the main diagonal
                for i in range(grid_size - offset):
                    grid[i, i + offset] = line_color
            else:
                # Diagonals above the main diagonal
                for i in range(grid_size + offset):
                    grid[i - offset, i] = line_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray) -> np.ndarray:
        """
        Transform the input grid by recoloring alternating cells on each diagonal line.
        """
        # Create a copy of the input grid
        output = grid.copy()
        
        # Get the line color from the input grid
        line_color = None
        for val in np.unique(grid):
            if val != 0:
                line_color = val
                break
        
        if line_color is None:
            return output
        
        # Always use color 6 for the alternating cells
        fill_color = 6
        
        # Find all diagonal lines
        grid_size = grid.shape[0]
        
        # Process each diagonal (both main and parallel)
        for offset in range(-grid_size+1, grid_size):
            if offset >= 0:
                # Diagonals below or on the main diagonal
                cells = [(i, i+offset) for i in range(grid_size-offset)]
            else:
                # Diagonals above the main diagonal
                cells = [(i-offset, i) for i in range(grid_size+offset)]
            
            # Filter valid cells and check if this is a colored diagonal
            valid_cells = [(r, c) for r, c in cells if 0 <= r < grid_size and 0 <= c < grid_size]
            diagonal_values = [grid[r, c] for r, c in valid_cells]
            
            # Skip empty diagonals or diagonals with mixed colors
            if not diagonal_values or all(val == 0 for val in diagonal_values):
                continue
                
            # Identify non-zero cells and check if they all have the same color
            non_zero_cells = [(r, c) for r, c in valid_cells if grid[r, c] != 0]
            non_zero_colors = {grid[r, c] for r, c in non_zero_cells}
            
            if len(non_zero_colors) > 1:
                continue
                
            # Ensure the diagonal has at least 3 cells before applying transformation
            if len(non_zero_cells) >= 3:
                # Apply the alternating color pattern: change every second cell to fill_color
                for i in range(1, len(non_zero_cells), 2):
                    r, c = non_zero_cells[i]
                    output[r, c] = fill_color
                
        return output