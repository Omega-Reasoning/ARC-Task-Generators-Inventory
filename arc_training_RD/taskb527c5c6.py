from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects, GridObject
from input_library import retry, random_cell_coloring
import numpy as np
import random

class TaskGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are squares of different sizes.",
            "Each input grid contains two intersecting bars where one vertical bar and one horizontal bar, both made of {color('block_color')} color.",
            "Each bar has a single pointer cell of {color('pointer_color')} color embedded in it. This pointer cell lies within the bar but is not at its end it is offset inward, typically around the center of the bar length. The pointer cell acts as a marker or pointer for that bar.",
            "The bars are placed such that their red marker cells are aligned or close to forming a T-junction (a shared region or near-intersection)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is a copy of the input grid.",
            "Each block bends at its pointer cell to form a T-shaped structure.",
            "From the pointer cell, A path of {color('pointer_color')} cells extends perpendicularly to the blocks original orientation. Alongside this path, {color('block_color')} cells extend in parallel to preserve the blocks thickness.",
            "The number of {color('block_color')} cells added along the new direction is equal to the number of rows or columns in the original block, excluding the one containing the pointer cell. Example A vertical block that spans 7 rows extends horizontally for 6 steps from its pointer cell.",
            "No two blocks including their extensions ever overlap. Extensions are placed in directions that maintain full separation, even if the original blocks were intersecting or nearby."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create a grid with separate horizontal and vertical bars with pointers."""
        block_color = taskvars["block_color"]
        pointer_color = taskvars["pointer_color"]
        
        # Fixed grid size for reliability
        size = 22
        grid = np.zeros((size, size), dtype=int)
        
        # Create vertical bar first
        v_length = random.randint(6, 8)
        v_thickness = random.randint(2, 3)
        v_start_row = random.randint(2, 6)
        v_start_col = random.randint(3, 8)
        
        # Ensure vertical bar fits
        if v_start_row + v_length >= size - 2:
            v_start_row = size - v_length - 3
        if v_start_col + v_thickness >= size - 2:
            v_start_col = size - v_thickness - 3
        
        # Draw vertical bar
        grid[v_start_row:v_start_row + v_length, 
             v_start_col:v_start_col + v_thickness] = block_color
        
        # Create horizontal bar with separation from vertical bar
        h_length = random.randint(6, 8)
        h_thickness = random.randint(2, 3)
        
        # Position horizontal bar with minimum separation of 4 cells
        min_separation = 4
        attempts = 0
        placed = False
        
        while attempts < 50 and not placed:
            h_start_row = random.randint(2, size - h_thickness - 3)
            h_start_col = random.randint(2, size - h_length - 3)
            
            # Check if horizontal bar is far enough from vertical bar
            v_center_row = v_start_row + v_length // 2
            v_center_col = v_start_col + v_thickness // 2
            h_center_row = h_start_row + h_thickness // 2
            h_center_col = h_start_col + h_length // 2
            
            distance = max(abs(v_center_row - h_center_row), abs(v_center_col - h_center_col))
            
            if distance >= min_separation:
                # Check that bars don't actually overlap
                v_area = set()
                for r in range(v_start_row, v_start_row + v_length):
                    for c in range(v_start_col, v_start_col + v_thickness):
                        v_area.add((r, c))
                
                h_area = set()
                for r in range(h_start_row, h_start_row + h_thickness):
                    for c in range(h_start_col, h_start_col + h_length):
                        h_area.add((r, c))
                
                if not v_area.intersection(h_area):
                    placed = True
            
            attempts += 1
        
        if not placed:
            # Fallback: place horizontal bar in a guaranteed non-overlapping position
            h_start_row = v_start_row + v_length + 3
            h_start_col = 3
            if h_start_row + h_thickness >= size:
                h_start_row = max(2, v_start_row - h_thickness - 3)
        
        # Draw horizontal bar
        grid[h_start_row:h_start_row + h_thickness, 
             h_start_col:h_start_col + h_length] = block_color
        
        # Place pointer in vertical bar
        v_pointer_row = v_start_row + v_length // 2
        v_pointer_col = v_start_col + v_thickness // 2
        
        # Determine safe extension direction for vertical bar
        # Check which direction has more space and won't conflict with horizontal bar
        space_left = v_pointer_col
        space_right = size - v_pointer_col - 1
        
        # Check potential collision with horizontal bar if extending
        left_collision = any(h_start_row <= r < h_start_row + h_thickness 
                           for r in range(v_start_row, v_start_row + v_length)
                           if any(0 <= v_pointer_col - i <= h_start_col + h_length - 1 
                                 for i in range(1, v_length)))
        
        right_collision = any(h_start_row <= r < h_start_row + h_thickness 
                            for r in range(v_start_row, v_start_row + v_length)
                            if any(h_start_col <= v_pointer_col + i <= h_start_col + h_length - 1 
                                  for i in range(1, v_length)))
        
        if not left_collision and space_left >= v_length:
            v_pointer_dir = -1  # Extend left
        elif not right_collision and space_right >= v_length:
            v_pointer_dir = 1   # Extend right
        else:
            v_pointer_dir = -1 if space_left > space_right else 1
        
        grid[v_pointer_row, v_pointer_col] = pointer_color
        
        # Place pointer in horizontal bar
        h_pointer_row = h_start_row + h_thickness // 2
        h_pointer_col = h_start_col + h_length // 2
        
        # Determine safe extension direction for horizontal bar
        space_up = h_pointer_row
        space_down = size - h_pointer_row - 1
        
        # Check potential collision with vertical bar if extending
        up_collision = any(v_start_col <= c < v_start_col + v_thickness 
                         for c in range(h_start_col, h_start_col + h_length)
                         if any(v_start_row <= h_pointer_row - i <= v_start_row + v_length - 1 
                               for i in range(1, h_length)))
        
        down_collision = any(v_start_col <= c < v_start_col + v_thickness 
                           for c in range(h_start_col, h_start_col + h_length)
                           if any(v_start_row <= h_pointer_row + i <= v_start_row + v_length - 1 
                                 for i in range(1, h_length)))
        
        if not up_collision and space_up >= h_length:
            h_pointer_dir = -1  # Extend up
        elif not down_collision and space_down >= h_length:
            h_pointer_dir = 1   # Extend down
        else:
            h_pointer_dir = -1 if space_up > space_down else 1
        
        grid[h_pointer_row, h_pointer_col] = pointer_color
        
        # Store metadata for transformation
        self.h_block_area = (h_start_row, h_start_col, h_thickness, h_length)
        self.v_block_area = (v_start_row, v_start_col, v_length, v_thickness)
        self.h_pointer_pos = (h_pointer_row, h_pointer_col)
        self.v_pointer_pos = (v_pointer_row, v_pointer_col)
        self.h_pointer_dir = h_pointer_dir
        self.v_pointer_dir = v_pointer_dir
        self.grid_size = size
        
        return grid

    def transform_input(self, grid, taskvars):
        """Transform input by extending blocks and pointers to form T-shapes with limited extension."""
        block_color = taskvars["block_color"]
        pointer_color = taskvars["pointer_color"]
        
        output_grid = grid.copy()
        
        # Extend vertical block and its pointer HORIZONTALLY
        v_start_row, v_start_col, v_length, v_thickness = self.v_block_area
        v_pointer_row, v_pointer_col = self.v_pointer_pos
        
        # Calculate extension distance: number of rows excluding pointer row
        v_extension_distance = v_length - 1  # Exclude the row containing the pointer
        
        # Extend pointer path horizontally
        for i in range(1, v_extension_distance + 1):
            new_col = v_pointer_col + (i * self.v_pointer_dir)
            if 0 <= new_col < self.grid_size and output_grid[v_pointer_row, new_col] == 0:
                output_grid[v_pointer_row, new_col] = pointer_color
        
        # Extend the remaining rows of vertical block alongside pointer
        for r in range(v_start_row, v_start_row + v_length):
            if r != v_pointer_row:  # Skip the pointer row
                for i in range(1, v_extension_distance + 1):
                    new_col = v_pointer_col + (i * self.v_pointer_dir)
                    if 0 <= new_col < self.grid_size and output_grid[r, new_col] == 0:
                        output_grid[r, new_col] = block_color
        
        # Extend horizontal block and its pointer VERTICALLY
        h_start_row, h_start_col, h_thickness, h_length = self.h_block_area
        h_pointer_row, h_pointer_col = self.h_pointer_pos
        
        # Calculate extension distance: number of columns excluding pointer column
        h_extension_distance = h_length - 1  # Exclude the column containing the pointer
        
        # Extend pointer path vertically
        for i in range(1, h_extension_distance + 1):
            new_row = h_pointer_row + (i * self.h_pointer_dir)
            if 0 <= new_row < self.grid_size and output_grid[new_row, h_pointer_col] == 0:
                output_grid[new_row, h_pointer_col] = pointer_color
        
        # Extend the remaining columns of horizontal block alongside pointer
        for c in range(h_start_col, h_start_col + h_length):
            if c != h_pointer_col:  # Skip the pointer column
                for i in range(1, h_extension_distance + 1):
                    new_row = h_pointer_row + (i * self.h_pointer_dir)
                    if 0 <= new_row < self.grid_size and output_grid[new_row, c] == 0:
                        output_grid[new_row, c] = block_color
        
        return output_grid

    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        # Randomly choose colors
        block_color = random.randint(1, 9)
        pointer_color = random.choice([c for c in range(1, 10) if c != block_color])
        
        taskvars = {
            "block_color": block_color,
            "pointer_color": pointer_color,
        }
        
        # Generate 3-5 training examples with same task variables
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        for _ in range(num_train_examples):
            gridvars = {}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example with same task variables
        test_gridvars = {}
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

