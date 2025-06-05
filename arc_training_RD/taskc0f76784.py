from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, GridObjects, GridObject
from input_library import create_object, random_cell_coloring, Contiguity

class Taskc0f76784Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid consists of squares of different sizes.",
            "Each square has its outline filled with {color('block_color')} color.",
            "The squares can be of sizes from 3x3 up to 5x5 (with inner regions 1x1 to 3x3)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid maintains the same square outlines with {color('block_color')} color.",
            "Each square inner region is filled based on its size:",
            "For a 3x3 square (1x1 inner), fill interior with {color('inner_color_1')} color",
            "For a 4x4 square (2x2 inner), fill interior with {color('inner_color_2')} color", 
            "For a 5x5 square (3x3 inner), fill interior with {color('inner_color_3')} color"
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        grid_size = random.randint(12, 18)
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        block_color = taskvars["block_color"]
        
        # Decide on number of squares (2-4)
        num_squares = random.randint(2, 4)
        
        # Keep track of occupied areas to avoid overlaps
        occupied_areas = []
        
        # Possible square sizes: 3x3, 4x4, 5x5 (inner regions 1x1, 2x2, 3x3)
        possible_sizes = [3, 4, 5]
        
        for _ in range(num_squares):
            # Create a square of size 3x3 to 5x5
            size = random.choice(possible_sizes)
            
            # Try to find a non-overlapping position
            for attempt in range(30):
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

    def transform_input(self, input_grid, taskvars):
        output_grid = np.copy(input_grid)
        block_color = taskvars["block_color"]
        
        # Map square size to inner color
        size_to_inner_color = {
            3: taskvars["inner_color_1"],  # 3x3 square -> 1x1 inner
            4: taskvars["inner_color_2"],  # 4x4 square -> 2x2 inner  
            5: taskvars["inner_color_3"]   # 5x5 square -> 3x3 inner
        }
        
        # Find all connected objects (the outlines)
        objects = find_connected_objects(input_grid, diagonal_connectivity=False, background=0)
        
        # For each object, fill its inner region based on size
        for obj in objects.objects:
            r_slice, c_slice = obj.bounding_box
            height = r_slice.stop - r_slice.start
            width = c_slice.stop - c_slice.start
            
            # Skip if not a square or invalid size
            if height != width or height not in size_to_inner_color:
                continue
                
            # Fill inner region with color corresponding to square size
            inner_color = size_to_inner_color[height]
            for r in range(r_slice.start + 1, r_slice.stop - 1):
                for c in range(c_slice.start + 1, c_slice.stop - 1):
                    output_grid[r, c] = inner_color
        
        return output_grid

    def create_grids(self):
        # First select block_color
        block_color = random.randint(1, 9)
        
        # Select 3 different colors for inner regions (excluding block_color)
        available_colors = list(set(range(1, 10)) - {block_color})
        inner_colors = random.sample(available_colors, 3)
        
        # Create taskvars dictionary
        taskvars = {
            "block_color": block_color,
            "inner_color_1": inner_colors[0],  # For 3x3 squares (1x1 inner)
            "inner_color_2": inner_colors[1],  # For 4x4 squares (2x2 inner)
            "inner_color_3": inner_colors[2]   # For 5x5 squares (3x3 inner)
        }
        
        # Helper for reasoning chain formatting
        def color_name(color_id):
            color_map = {
                1: "blue", 2: "red", 3: "green", 4: "yellow", 5: "gray",
                6: "magenta", 7: "orange", 8: "cyan", 9: "brown"
            }
            return color_map.get(color_id, f"color_{color_id}")
        
        def color_fmt(key):
            color_id = taskvars[key]
            return f"{color_name(color_id)} ({color_id})"
        
        # Replace placeholders in reasoning chains
        self.input_reasoning_chain = [
            chain.replace("{color('block_color')}", color_fmt('block_color'))
            for chain in self.input_reasoning_chain
        ]
        self.transformation_reasoning_chain = [
            chain.replace("{color('block_color')}", color_fmt('block_color'))
                 .replace("{color('inner_color_1')}", color_fmt('inner_color_1'))
                 .replace("{color('inner_color_2')}", color_fmt('inner_color_2'))
                 .replace("{color('inner_color_3')}", color_fmt('inner_color_3'))
            for chain in self.transformation_reasoning_chain
        ]
        
        # Generate training examples
        num_train_examples = random.randint(3, 5)
        train_pairs = []
        
        for _ in range(num_train_examples):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]

        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)

