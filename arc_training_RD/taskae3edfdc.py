from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects, GridObject
from input_library import create_object, retry, random_cell_coloring, Contiguity
import numpy as np
import random

class Taskae3edfdcGenerator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are squares and of the same size.",
            "The grid contains 2 main colored cells and can be placed anywhere in the grid.",
            "The two main cells are covered with one type of cells namely scattered cells (<=4), where the scattered cells are again grouped by two colors.",
            "Scattered cells are of a different color group than the main cells, * These scattered cell are grouped into two distinct colors, one for each main cell.They are positioned strictly horizontal or vertical to their associated main cell. Not adjacent (no touching) to their main cell or to each other.",
            "Each scattered cell is uniquely associated with one main cell.",
            "A scattered cell is always placed such that:",
            "It is horizontal or vertical to the main cell (but not diagonal).",
            "There is at least one empty cell between it and the main cell."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "For each main cell, Identify its corresponding scattered cells based on Color grouping and Horizontal/vertical orientation.",
            "Attach each scattered cell adjacent to its main cell in the same direction as in the input: If a scattered cell is 2 cells right of the main cell, attach it 1 cell right in the output. Similarly, vertical relationships are preserved.",
            "The main cell remains unchanged. Each scattered cell is attached once, directly forming a \"plus\" pattern around the main cell. Do not create a series or chain of scattered cells."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create input grid with main cells and their associated scattered cells."""
        grid_size = gridvars['grid_size']
        main_cell1_color = taskvars['main_cell1']
        main_cell2_color = taskvars['main_cell2']
        scattered_color1 = taskvars['scattered_main_1']
        scattered_color2 = taskvars['scattered_main_2']
        
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Place main cells ensuring they don't conflict
        def place_main_cells():
            positions = []
            for _ in range(50):  # Try multiple times
                pos1 = (random.randint(2, grid_size-3), random.randint(2, grid_size-3))
                pos2 = (random.randint(2, grid_size-3), random.randint(2, grid_size-3))
                
                # Ensure main cells are not too close to each other
                if abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) >= 4:
                    positions = [pos1, pos2]
                    break
            
            if not positions:
                # Fallback positions
                positions = [(2, 2), (grid_size-3, grid_size-3)]
            
            return positions
        
        main_positions = place_main_cells()
        main_pos1, main_pos2 = main_positions
        
        # Place main cells
        grid[main_pos1[0], main_pos1[1]] = main_cell1_color
        grid[main_pos2[0], main_pos2[1]] = main_cell2_color
        
        # Place scattered cells for each main cell
        for main_idx, (main_pos, scattered_color) in enumerate(zip(main_positions, [scattered_color1, scattered_color2])):
            num_scattered = random.randint(1, 4)
            
            # Get valid positions for scattered cells (horizontal/vertical, not adjacent)
            valid_positions = self._get_valid_scattered_positions(grid, main_pos, grid_size)
            
            if len(valid_positions) >= num_scattered:
                selected_positions = random.sample(valid_positions, num_scattered)
                
                for pos in selected_positions:
                    grid[pos[0], pos[1]] = scattered_color
        
        return grid
    
    def _get_valid_scattered_positions(self, grid, main_pos, grid_size):
        """Get valid positions for scattered cells around a main cell"""
        valid_positions = []
        main_r, main_c = main_pos
        
        # Check horizontal and vertical directions
        directions = [
            # Horizontal (left and right)
            [(0, -i) for i in range(2, grid_size)],  # Left
            [(0, i) for i in range(2, grid_size)],   # Right
            # Vertical (up and down)
            [(-i, 0) for i in range(2, grid_size)],  # Up
            [(i, 0) for i in range(2, grid_size)]    # Down
        ]
        
        for direction in directions:
            for dr, dc in direction:
                r, c = main_r + dr, main_c + dc
                
                # Check bounds
                if 0 <= r < grid_size and 0 <= c < grid_size:
                    # Check if position is empty and not adjacent to other objects
                    if grid[r, c] == 0 and self._is_position_valid(grid, (r, c), grid_size):
                        valid_positions.append((r, c))
                        break  # Only take first valid position in each direction
        
        return valid_positions
    
    def _is_position_valid(self, grid, pos, grid_size):
        """Check if position is valid (not adjacent to existing non-background cells)"""
        r, c = pos
        
        # Check 8-connected neighbors
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    if grid[nr, nc] != 0:
                        return False
        return True
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """Transform input by attaching scattered cells adjacent to their main cells"""
        output_grid = grid.copy()
        grid_size = grid.shape[0]
        
        # Get colors from taskvars
        main_cell1_color = taskvars['main_cell1']
        main_cell2_color = taskvars['main_cell2']
        scattered_color1 = taskvars['scattered_main_1']
        scattered_color2 = taskvars['scattered_main_2']
        
        # Find main cells and their positions
        main_positions = []
        main_colors = [main_cell1_color, main_cell2_color]
        scattered_colors = [scattered_color1, scattered_color2]
        
        for main_color in main_colors:
            for r in range(grid_size):
                for c in range(grid_size):
                    if grid[r, c] == main_color:
                        main_positions.append((r, c))
                        break
                else:
                    continue
                break
        
        # Process each main cell and its scattered cells
        for main_idx, (main_pos, main_color, scattered_color) in enumerate(
            zip(main_positions, main_colors, scattered_colors)):
            
            main_r, main_c = main_pos
            
            # Find all scattered cells of this color
            scattered_positions = []
            for r in range(grid_size):
                for c in range(grid_size):
                    if grid[r, c] == scattered_color:
                        scattered_positions.append((r, c))
            
            # Remove scattered cells from their original positions
            for scat_r, scat_c in scattered_positions:
                output_grid[scat_r, scat_c] = 0
            
            # Attach scattered cells adjacent to main cell
            for scat_r, scat_c in scattered_positions:
                # Determine direction from main to scattered cell
                dr = scat_r - main_r
                dc = scat_c - main_c
                
                # Normalize direction to get adjacent position
                if dr != 0:
                    # Vertical relationship
                    new_dr = 1 if dr > 0 else -1
                    new_r = main_r + new_dr
                    new_c = main_c
                else:
                    # Horizontal relationship
                    new_dc = 1 if dc > 0 else -1
                    new_r = main_r
                    new_c = main_c + new_dc
                
                # Place scattered cell adjacent to main cell if position is valid
                if (0 <= new_r < grid_size and 0 <= new_c < grid_size and 
                    output_grid[new_r, new_c] == 0):
                    output_grid[new_r, new_c] = scattered_color
        
        return output_grid
    
    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        # Set consistent grid size and colors for all grids
        grid_size = random.randint(8, 15)  # Ensure enough space for scattered cells
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        
        # Store task variables
        taskvars = {
            "main_cell1": available_colors[0],
            "main_cell2": available_colors[1],
            "scattered_main_1": available_colors[2],
            "scattered_main_2": available_colors[3]
        }
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        for _ in range(num_train_examples):
            gridvars = {'grid_size': grid_size}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {'grid_size': grid_size}
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