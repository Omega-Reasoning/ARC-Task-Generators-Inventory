from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects, GridObject
from input_library import create_object, retry, random_cell_coloring, Contiguity
import numpy as np
import random

class Taskae3edfdcGenerator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are squares and of size {vars['grid_size']} x {vars['grid_size']}.",
            "The grid contains 2 main colored cells in {color('main_cell1')} and {color('main_cell2')} colors and can be placed anywhere in the grid.",
            "The two main cells are covered with scattered cells, where the scattered cells are grouped by two colors {color('scattered_main_1')} and {color('scattered_main_2')}.",
            "Each main cell has at least 2 scattered cells associated with it, and at least one main cell has scattered cells in all 4 directions (up, down, left, right).",
            "Scattered cells are positioned far away from their main cells, creating clear visual separation between main cells and their scattered cells.",
            "Each scattered cell is uniquely associated with one main cell.",
            "A scattered cell is always positioned either horizontally or vertically—but not diagonally—from the main cell, at a distance of 4 to 8 empty cells away.",
            "Scattered cells appear genuinely scattered across the grid with significant spacing from their main cells."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "For each main cell in {color('main_cell1')} and {color('main_cell2')} colors, identify its corresponding scattered cells (in {color('scattered_main_1')} and {color('scattered_main_2')} colors respectively) based on color grouping and horizontal/vertical orientation.",
            "Attach each scattered cell adjacent to its main cell in the same direction as in the input: If a scattered cell is 6 cells right of the main cell, attach it 1 cell right in the output. Similarly, vertical relationships are preserved.",
            "The main cell remains unchanged. Each scattered cell is attached once, directly forming a \"plus\" pattern around the main cell."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        taskvars = {}
        
        # Set larger grid size to accommodate far-scattered cells
        grid_size = random.randint(16, 20)  # Slightly larger for 4-way structure
        
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        
        # Store task variables
        taskvars['grid_size'] = grid_size
        taskvars['main_cell1'] = available_colors[0]
        taskvars['main_cell2'] = available_colors[1]
        taskvars['scattered_main_1'] = available_colors[2]
        taskvars['scattered_main_2'] = available_colors[3]
        
        nr_train_examples = random.randint(3, 5)
        nr_test_examples = 1
        
        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)
        return taskvars, train_test_data
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create input grid with main cells and far-scattered cells."""
        grid_size = taskvars['grid_size']
        main_cell1_color = taskvars['main_cell1']
        main_cell2_color = taskvars['main_cell2']
        scattered_color1 = taskvars['scattered_main_1']
        scattered_color2 = taskvars['scattered_main_2']
        
        max_attempts = 100
        
        for attempt in range(max_attempts):
            grid = np.zeros((grid_size, grid_size), dtype=int)
            
            # Place main cells in center region
            main_positions = self._place_main_cells_centrally(grid_size)
            main_pos1, main_pos2 = main_positions
            
            # Place main cells
            grid[main_pos1[0], main_pos1[1]] = main_cell1_color
            grid[main_pos2[0], main_pos2[1]] = main_cell2_color
            
            # Randomly decide which main cell gets 4-way structure
            main_with_4way = random.choice([0, 1])
            
            success = True
            
            # Place scattered cells for each main cell
            for main_idx, (main_pos, scattered_color) in enumerate(zip(main_positions, [scattered_color1, scattered_color2])):
                
                if main_idx == main_with_4way:
                    # This main cell gets 4-way structure (4 scattered cells)
                    scattered_positions = self._get_4way_scattered_positions(grid, main_pos, grid_size)
                    
                    if len(scattered_positions) == 4:  # Must have all 4 directions
                        for pos in scattered_positions:
                            grid[pos[0], pos[1]] = scattered_color
                    else:
                        success = False
                        break
                else:
                    # This main cell gets 2-3 scattered cells
                    num_scattered = random.randint(2, 3)
                    scattered_positions = self._get_partial_scattered_positions(grid, main_pos, grid_size, num_scattered)
                    
                    if len(scattered_positions) >= num_scattered:
                        for pos in scattered_positions:
                            grid[pos[0], pos[1]] = scattered_color
                    else:
                        success = False
                        break
            
            if success:
                # Verify structure requirements
                scattered_count1 = np.sum(grid == scattered_color1)
                scattered_count2 = np.sum(grid == scattered_color2)
                
                # Check if we have the required 4-way structure
                has_4way = self._verify_4way_structure(grid, main_positions[main_with_4way], 
                                                     [scattered_color1, scattered_color2][main_with_4way])
                
                if scattered_count1 >= 2 and scattered_count2 >= 2 and has_4way:
                    return grid
        
        # Fallback: create a grid with guaranteed structure
        return self._create_fallback_grid(taskvars)
    
    def _place_main_cells_centrally(self, grid_size):
        """Place main cells in central region with room for 4-way scattering"""
        center = grid_size // 2
        
        for _ in range(50):
            # Keep main cells well within center to allow 4-way scattering
            pos1 = (random.randint(center-4, center+4), random.randint(center-4, center+4))
            pos2 = (random.randint(center-4, center+4), random.randint(center-4, center+4))
            
            # Ensure main cells are reasonably separated
            if abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) >= 6:
                return [pos1, pos2]
        
        # Fallback positions
        return [(center-3, center-3), (center+3, center+3)]
    
    def _get_4way_scattered_positions(self, grid, main_pos, grid_size):
        """Get scattered positions in all 4 directions from main cell"""
        main_r, main_c = main_pos
        scattered_positions = []
        
        # Define all 4 directions (must have all)
        directions = [
            (0, 1),   # Right
            (0, -1),  # Left  
            (1, 0),   # Down
            (-1, 0),  # Up
        ]
        
        for dr, dc in directions:
            # Try distances 4-8 cells away
            position_found = False
            for distance in range(4, 9):
                new_r = main_r + (dr * distance)
                new_c = main_c + (dc * distance)
                
                # Check if position is valid and within bounds
                if (0 <= new_r < grid_size and 0 <= new_c < grid_size and 
                    grid[new_r, new_c] == 0):
                    
                    scattered_positions.append((new_r, new_c))
                    position_found = True
                    break
            
            # If we couldn't place in this direction, return incomplete
            if not position_found:
                return []  # Failed to get 4-way structure
        
        return scattered_positions
    
    def _get_partial_scattered_positions(self, grid, main_pos, grid_size, num_scattered):
        """Get scattered positions (partial, not necessarily 4-way)"""
        main_r, main_c = main_pos
        scattered_positions = []
        
        # Define directions
        directions = [
            (0, 1),   # Right
            (0, -1),  # Left  
            (1, 0),   # Down
            (-1, 0),  # Up
        ]
        
        # Shuffle directions to randomize placement
        random.shuffle(directions)
        
        for direction in directions:
            if len(scattered_positions) >= num_scattered:
                break
                
            dr, dc = direction
            
            # Use far distances (4 to 8 cells away)
            for distance in range(4, 9):
                new_r = main_r + (dr * distance)
                new_c = main_c + (dc * distance)
                
                # Check if position is valid and within bounds
                if (0 <= new_r < grid_size and 0 <= new_c < grid_size and 
                    grid[new_r, new_c] == 0):
                    
                    scattered_positions.append((new_r, new_c))
                    break  # Found a good position in this direction
        
        return scattered_positions
    
    def _verify_4way_structure(self, grid, main_pos, scattered_color):
        """Verify that a main cell has scattered cells in all 4 directions"""
        main_r, main_c = main_pos
        
        directions_found = set()
        
        # Check all positions in the grid for scattered cells of this color
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == scattered_color:
                    # Determine direction relative to main cell
                    dr = r - main_r
                    dc = c - main_c
                    
                    if dr == 0 and dc > 0:
                        directions_found.add('right')
                    elif dr == 0 and dc < 0:
                        directions_found.add('left')
                    elif dr > 0 and dc == 0:
                        directions_found.add('down')
                    elif dr < 0 and dc == 0:
                        directions_found.add('up')
        
        # Check if all 4 directions are present
        return len(directions_found) == 4
    
    def _create_fallback_grid(self, taskvars):
        """Create a fallback grid with guaranteed 4-way structure"""
        grid_size = taskvars['grid_size']
        main_cell1_color = taskvars['main_cell1']
        main_cell2_color = taskvars['main_cell2']
        scattered_color1 = taskvars['scattered_main_1']
        scattered_color2 = taskvars['scattered_main_2']
        
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Place main cells in center
        center = grid_size // 2
        main_pos1 = (center-3, center-3)
        main_pos2 = (center+3, center+3)
        
        grid[main_pos1[0], main_pos1[1]] = main_cell1_color
        grid[main_pos2[0], main_pos2[1]] = main_cell2_color
        
        # Give main cell 1 the 4-way structure
        # Place scattered cells in all 4 directions from main cell 1
        grid[main_pos1[0], main_pos1[1] + 5] = scattered_color1  # Right
        grid[main_pos1[0], main_pos1[1] - 5] = scattered_color1  # Left
        grid[main_pos1[0] + 5, main_pos1[1]] = scattered_color1  # Down
        grid[main_pos1[0] - 5, main_pos1[1]] = scattered_color1  # Up
        
        # Give main cell 2 partial structure (2 directions)
        grid[main_pos2[0], main_pos2[1] - 6] = scattered_color2  # Left
        grid[main_pos2[0] - 4, main_pos2[1]] = scattered_color2  # Up
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """Transform input by attaching scattered cells adjacent to their main cells"""
        output_grid = grid.copy()
        grid_size = taskvars['grid_size']
        
        # Get colors from taskvars
        main_cell1_color = taskvars['main_cell1']
        main_cell2_color = taskvars['main_cell2']
        scattered_color1 = taskvars['scattered_main_1']
        scattered_color2 = taskvars['scattered_main_2']
        
        # Find main cells and their positions - FIXED
        main_positions = []
        main_colors = [main_cell1_color, main_cell2_color]
        scattered_colors = [scattered_color1, scattered_color2]
        
        for main_color in main_colors:
            found = False
            for r in range(grid_size):
                for c in range(grid_size):
                    if grid[r, c] == main_color:
                        main_positions.append((r, c))
                        found = True
                        break
                if found:
                    break
        
        # Process each main cell and its scattered cells
        for main_pos, main_color, scattered_color in zip(main_positions, main_colors, scattered_colors):
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
            
            # Attach scattered cells adjacent to main cell in proper directions
            directions_used = set()
            
            for scat_r, scat_c in scattered_positions:
                # Determine direction from main to scattered cell
                dr = scat_r - main_r
                dc = scat_c - main_c
                
                # Determine which direction and get adjacent position
                if dr == 0 and dc > 0:  # Right
                    new_r, new_c = main_r, main_c + 1
                    direction = 'right'
                elif dr == 0 and dc < 0:  # Left
                    new_r, new_c = main_r, main_c - 1
                    direction = 'left'
                elif dr > 0 and dc == 0:  # Down
                    new_r, new_c = main_r + 1, main_c
                    direction = 'down'
                elif dr < 0 and dc == 0:  # Up
                    new_r, new_c = main_r - 1, main_c
                    direction = 'up'
                else:
                    continue  # Skip invalid directions
                
                # Place scattered cell adjacent to main cell if position is valid and direction not used
                if (0 <= new_r < grid_size and 0 <= new_c < grid_size and 
                    output_grid[new_r, new_c] == 0 and direction not in directions_used):
                    output_grid[new_r, new_c] = scattered_color
                    directions_used.add(direction)
        
        return output_grid