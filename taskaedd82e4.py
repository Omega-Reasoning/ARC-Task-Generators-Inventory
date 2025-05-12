from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import create_object, Contiguity, random_cell_coloring

class Taskaedd82e4Generator(ARCTaskGenerator):
    def __init__(self):
        self.input_reasoning_chain = [
            "Input grids are square grids of size MxM.",
            "The grid consists of patterns in {{color(\"pattern_color\")}} color and scattered single cells in the same color."
        ]
        
        self.transformation_reasoning_chain = [
            "The output grid is created by copying the input grid.",
            "The patterns remain in {{color(\"pattern_color\")}} color",
            "Only the scattered single cells change to {{color(\"cell_color\")}} color"
        ]
        
        taskvars_definitions = {
            'pattern_color': 'Color of patterns (stays same in input and output)',
            'cell_color': 'Color of scattered cells in output grid'
        }
        
        super().__init__(self.input_reasoning_chain, self.transformation_reasoning_chain)

    def create_input(self, grid_size, pattern_color, num_patterns=2, scatter_density=0.02):
        """Create a square grid with patterns and scattered cells all in one color"""
        # Initialize empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Create a few pattern objects
        for _ in range(num_patterns):
            pattern_size = random.randint(2, min(4, grid_size // 2))
            pattern = create_object(
                height=pattern_size,
                width=pattern_size,
                color_palette=pattern_color,  # Use pattern_color for patterns
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
            color_palette=pattern_color,  # Use pattern_color for scattered cells
            density=scatter_density,  # Reduced density
            background=0,
            overwrite=False
        )
        
        return grid
    
    def transform_input(self, input_grid, pattern_color, cell_color):
        """Transform input grid by changing only scattered cells color"""
        output_grid = input_grid.copy()  # Copy to preserve patterns
        
        # Find all connected objects in the input grid
        all_objects = find_connected_objects(input_grid, diagonal_connectivity=True, background=0)
        
        # Change color of only scattered cells
        for obj in all_objects:
            if len(obj) == 1:  # Scattered cell
                r, c, _ = next(iter(obj.cells))
                output_grid[r, c] = cell_color
        
        return output_grid
    
    def create_grids(self):
        # Generate two distinct colors
        pattern_color = random.randint(1, 9)
        cell_color = random.choice([c for c in range(1, 10) if c != pattern_color])
        
        # Generate training pairs
        train_pairs = []
        num_examples = random.randint(3, 5)
        
        for _ in range(num_examples):
            grid_size = random.randint(5, 20)
            input_grid = self.create_input(
                grid_size=grid_size,
                pattern_color=pattern_color,
                num_patterns=random.randint(1, 3),
                scatter_density=0.02
            )
            
            output_grid = self.transform_input(
                input_grid=input_grid,
                pattern_color=pattern_color,
                cell_color=cell_color
            )
            
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate test example
        test_grid_size = random.randint(5, 20)
        test_input = self.create_input(
            grid_size=test_grid_size,
            pattern_color=pattern_color,
            num_patterns=random.randint(1, 3),
            scatter_density=0.02
        )
        test_output = self.transform_input(test_input, pattern_color, cell_color)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        taskvars = {
            'grid_size': f"{test_grid_size}x{test_grid_size}",
            'pattern_color': pattern_color,
            'cell_color': cell_color
        }
        
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)
