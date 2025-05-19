from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, GridObjects, GridObject
from input_library import create_object, random_cell_coloring, Contiguity

class Tasktaskc8f0f002Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid consists of squares of different sizes.",
            f"Each square has its outline filled with the {'block_color'} color.",
            "The squares can be of sizes from 1x1 up to 9x9."
        ]
        
        transformation_reasoning_chain = [
            f"The output grid maintains the same square outlines with {'block_color'} color.",
            "Each square inner region is filled based on its size:",
            f"For a square of size NxN, its interior is filled with {'inner_color_N'} color",
            "Different sized squares get different interior colors"
        ]
        
        # Generate dynamic taskvars for all possible square sizes
        taskvars_definitions = {
            "block_color": {
                "description": "color used for square outlines",
                "type": "int",
                "range": [1, 9]
            }
        }
        
        # Add inner color definitions for each possible square size
        for size in range(1, 10):
            taskvars_definitions[f"inner_color_{size}"] = {
                "description": f"color for {size}x{size} square interiors",
                "type": "int",
                "range": [1, 9]
            }
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, gridvars):
        grid_size = random.randint(10, 20)
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        block_color = gridvars.get("block_color", random.randint(1, 9))
        
        # Decide on number of squares (2-4)
        num_squares = random.randint(2, 4)
        
        # Keep track of occupied areas to avoid overlaps
        occupied_areas = []
        
        for _ in range(num_squares):
            # Create a square of size 3x3 to 6x6
            size = random.randint(3, 6)
            
            # Try to find a non-overlapping position
            for attempt in range(20):
                start_row = random.randint(1, grid_size - size - 1)
                start_col = random.randint(1, grid_size - size - 1)
                
                area = (start_row-1, start_col-1, start_row+size+1, start_col+size+1)
                
                if not any(self._areas_overlap(area, existing) for existing in occupied_areas):
                    occupied_areas.append(area)
                    
                    # Draw the square outline
                    for r in range(start_row, start_row + size):
                        for c in range(start_col, start_col + size):
                            if (r == start_row or r == start_row + size - 1 or
                                c == start_col or c == start_col + size - 1):
                                grid[r, c] = block_color
                    break
        
        return grid
    
    def _areas_overlap(self, area1, area2):
        # Check if two areas (specified as (r1, c1, r2, c2)) overlap
        r1_1, c1_1, r2_1, c2_1 = area1
        r1_2, c1_2, r2_2, c2_2 = area2
        
        # If one rectangle is to the left of the other
        if c2_1 < c1_2 or c2_2 < c1_1:
            return False
        
        # If one rectangle is above the other
        if r2_1 < r1_2 or r2_2 < r1_1:
            return False
        
        # Otherwise, they overlap
        return True

    def transform_input(self, input_grid, gridvars):
        output_grid = np.copy(input_grid)
        block_color = gridvars.get("block_color")
        
        # Create dynamic inner_colors dictionary
        inner_colors = {}
        for size in range(1, 10):
            color = gridvars.get(f"inner_color_{size}")
            if color is not None:
                inner_colors[size] = color
        
        # Find all connected objects (the outlines)
        objects = find_connected_objects(input_grid, diagonal_connectivity=False, background=0)
        
        # For each object, fill its inner region based on size
        for obj in objects.objects:
            r_slice, c_slice = obj.bounding_box
            size = r_slice.stop - r_slice.start
            
            # Skip if not a square or invalid size
            if size not in inner_colors or size != (c_slice.stop - c_slice.start):
                continue
                
            # Fill inner region with color corresponding to square size
            inner_color = inner_colors[size]
            for r in range(r_slice.start + 1, r_slice.stop - 1):
                for c in range(c_slice.start + 1, c_slice.stop - 1):
                    output_grid[r, c] = inner_color
        
        return output_grid

    def create_grids(self):
        # First select block_color
        block_color = random.randint(1, 9)
        
        # Select different colors for inner regions (excluding block_color)
        available_colors = list(set(range(1, 10)) - {block_color})
        inner_colors = random.sample(available_colors, min(8, len(available_colors)))
        
        # Create gridvars dictionary
        gridvars = {
            "block_color": block_color
        }
        
        # Assign colors for each square size
        for size, color in enumerate(inner_colors, start=1):
            gridvars[f"inner_color_{size}"] = color
        
        # Generate training examples
        num_train_examples = random.randint(3, 5)
        train_pairs = []
        
        for _ in range(num_train_examples):
            input_grid = self.create_input(gridvars)
            output_grid = self.transform_input(input_grid, gridvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate test example
        test_input = self.create_input(gridvars)
        test_output = self.transform_input(test_input, gridvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]

        return gridvars, TrainTestData(train=train_pairs, test=test_pairs)

