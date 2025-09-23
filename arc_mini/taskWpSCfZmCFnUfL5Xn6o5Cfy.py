from arc_task_generator import ARCTaskGenerator, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple

class taskWpSCfZmCFnUfL5Xn6o5CfyGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids can have different sizes.",
            "Each input grid contains several same-colored cells; all other cells are empty (0).",
            "The colored cells are mostly separated from each other, but some may be diagonally connected.",
            "The color and positions of the cells vary across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and for each colored cell in the input, add up to four {color('new_cell')} cells orthogonally adjacent (top, bottom, left, right), forming a plus shape.",
            "Do not overwrite existing non-zero cells; if a target location is already occupied, leave it as is.",
            "Respect grid boundaries and only add cells that lie within the grid.",
            "The original colored cells remain unchanged.",
            "If multiple expansions target the same location, place a single {color('new_cell')} cell there."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """Create input grid with scattered colored cells."""
        grid_size = gridvars.get('grid_size', random.randint(5, 15))
        cell_color = gridvars.get('cell_color', random.randint(1, 9))
        if gridvars.get('num_cells') is None:
            num_cells = random.randint(2, min(8, grid_size))
        else:
            num_cells = gridvars['num_cells']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Place scattered colored cells
        placed_cells = 0
        attempts = 0
        max_attempts = 100
        
        while placed_cells < num_cells and attempts < max_attempts:
            row = random.randint(0, grid_size - 1)
            col = random.randint(0, grid_size - 1)
            
            # Place cell if position is empty
            if grid[row, col] == 0:
                grid[row, col] = cell_color
                placed_cells += 1
                
                # Optionally add diagonally connected cells (30% chance)
                if random.random() < 0.3 and placed_cells < num_cells:
                    # Try to place a diagonally adjacent cell
                    diagonal_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                    random.shuffle(diagonal_dirs)
                    
                    for dr, dc in diagonal_dirs:
                        new_row, new_col = row + dr, col + dc
                        if (0 <= new_row < grid_size and 
                            0 <= new_col < grid_size and 
                            grid[new_row, new_col] == 0):
                            grid[new_row, new_col] = cell_color
                            placed_cells += 1
                            break
            
            attempts += 1
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform input by adding plus-shaped expansions around colored cells."""
        output_grid = grid.copy()
        new_cell_color = taskvars['new_cell']
        
        # Find all non-zero cells in the input
        colored_cells = np.where(grid != 0)
        
        # For each colored cell, try to add plus-shaped expansion
        for row, col in zip(colored_cells[0], colored_cells[1]):
            # Define orthogonal directions: up, down, left, right
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            for dr, dc in directions:
                new_row = row + dr
                new_col = col + dc
                
                # Check if the new position is within grid boundaries
                if (0 <= new_row < grid.shape[0] and 
                    0 <= new_col < grid.shape[1]):
                    
                    # Only place new_cell if the position is empty (0)
                    if output_grid[new_row, new_col] == 0:
                        output_grid[new_row, new_col] = new_cell_color
        
        return output_grid

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """Create training and test grids with task variables."""
        # Initialize task variables
        taskvars = {
            'new_cell': random.randint(1, 9)  # Color for plus expansions
        }
        
        # Generate training examples (3-5 examples)
        num_train = random.randint(3, 5)
        train_examples = []
        
        for _ in range(num_train):
            # Create diverse grid variables for each example
            gridvars = {
                'grid_size': random.randint(5, 15),
                'cell_color': random.choice([c for c in range(1, 10) if c != taskvars['new_cell']]),
                'num_cells': None  # Will be determined in create_input
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        test_gridvars = {
            'grid_size': random.randint(5, 15),
            'cell_color': random.choice([c for c in range(1, 10) if c != taskvars['new_cell']]),
            'num_cells': None
        }
        
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data


