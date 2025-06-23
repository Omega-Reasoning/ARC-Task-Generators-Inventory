from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import create_object, Contiguity, random_cell_coloring

class Taskaedd82e4Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square grids of size MxM.",
            "The grid consists of patterns in {color('pattern_color')} color and scattered single cells in the same color."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid.",
            "The patterns remain in {color('pattern_color')} color",
            "Only the scattered single cells change to {color('cell_color')} color"
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create a square grid with patterns and scattered cells all in one color"""
        pattern_color = taskvars['pattern_color']
        grid_size = gridvars['grid_size']
        num_patterns = gridvars.get('num_patterns', 2)
        scatter_density = gridvars.get('scatter_density', 0.02)
            
        max_attempts = 10
        for attempt in range(max_attempts):
            # Initialize empty grid
            grid = np.zeros((grid_size, grid_size), dtype=int)
            
            # Create a few pattern objects
            for _ in range(num_patterns):
                pattern_size = random.randint(2, min(4, grid_size // 2))
                pattern = create_object(
                    height=pattern_size,
                    width=pattern_size,
                    color_palette=pattern_color,
                    contiguity=Contiguity.EIGHT,
                    background=0
                )
                
                # Find a random position to place the pattern
                max_pos = grid_size - pattern_size
                if max_pos <= 0:
                    continue  # Skip if pattern doesn't fit
                    
                pos_r = random.randint(0, max_pos)
                pos_c = random.randint(0, max_pos)
                
                # Place the pattern on the grid
                pattern_obj = GridObject.from_array(pattern, offset=(pos_r, pos_c))
                pattern_obj.paste(grid)
            
            # Add scattered cells with same color
            random_cell_coloring(
                grid=grid,
                color_palette=pattern_color,
                density=scatter_density,
                background=0,
                overwrite=False
            )
            
            # Check if we have at least one individual cell
            all_objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)
            individual_cells = [obj for obj in all_objects if len(obj) == 1]
            
            if len(individual_cells) > 0:
                return grid
            
            # If no individual cells, force add at least one
            # Find empty positions that are not adjacent to existing colored cells
            empty_positions = []
            for r in range(grid_size):
                for c in range(grid_size):
                    if grid[r, c] == 0:
                        # Check if this position is isolated (not adjacent to any colored cell)
                        is_isolated = True
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = r + dr, c + dc
                                if (0 <= nr < grid_size and 0 <= nc < grid_size and 
                                    grid[nr, nc] != 0):
                                    is_isolated = False
                                    break
                            if not is_isolated:
                                break
                        if is_isolated:
                            empty_positions.append((r, c))
            
            # Place at least one individual cell
            if empty_positions:
                num_individual = random.randint(1, min(3, len(empty_positions)))
                selected_positions = random.sample(empty_positions, num_individual)
                for r, c in selected_positions:
                    grid[r, c] = pattern_color
                return grid
            
            # If we can't find isolated positions, try with less restrictive placement
            empty_positions = [(r, c) for r in range(grid_size) for c in range(grid_size) if grid[r, c] == 0]
            if empty_positions:
                # Just place individual cells in empty positions
                num_individual = random.randint(1, min(3, len(empty_positions)))
                selected_positions = random.sample(empty_positions, num_individual)
                for r, c in selected_positions:
                    grid[r, c] = pattern_color
                
                # Verify we actually created individual cells
                all_objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)
                individual_cells = [obj for obj in all_objects if len(obj) == 1]
                if len(individual_cells) > 0:
                    return grid
        
        # If all attempts failed, create a simple grid with guaranteed individual cells
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Place a simple 2x2 pattern in the center
        center = grid_size // 2
        if center > 0 and center < grid_size - 1:
            grid[center:center+2, center:center+2] = pattern_color
        
        # Place individual cells in corners
        corners = [(1, 1), (1, grid_size-2), (grid_size-2, 1), (grid_size-2, grid_size-2)]
        valid_corners = [(r, c) for r, c in corners if 0 <= r < grid_size and 0 <= c < grid_size]
        if valid_corners:
            num_corners = random.randint(1, min(2, len(valid_corners)))
            selected_corners = random.sample(valid_corners, num_corners)
            for r, c in selected_corners:
                grid[r, c] = pattern_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """Transform input grid by changing only scattered cells color"""
        pattern_color = taskvars['pattern_color']
        cell_color = taskvars['cell_color']
        
        output_grid = grid.copy()  # Copy to preserve patterns
        
        # Find all connected objects in the input grid
        all_objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)
        
        # Change color of only scattered cells
        for obj in all_objects:
            if len(obj) == 1:  # Scattered cell
                r, c, _ = next(iter(obj.cells))
                output_grid[r, c] = cell_color
        
        return output_grid
    
    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        # Generate two distinct colors
        pattern_color = random.randint(1, 9)
        cell_color = random.choice([c for c in range(1, 10) if c != pattern_color])
        
        # Store task variables
        taskvars = {
            'pattern_color': pattern_color,
            'cell_color': cell_color,
        }
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Generate grid sizes
        grid_sizes = [random.randint(5, 20) for _ in range(num_train_examples + 1)]
        
        # Create training examples
        for i in range(num_train_examples):
            gridvars = {
                'grid_size': grid_sizes[i],
                'num_patterns': random.randint(1, 3),
                'scatter_density': 0.02
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {
            'grid_size': grid_sizes[-1],
            'num_patterns': random.randint(1, 3),
            'scatter_density': 0.02
        }
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