from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, retry, Contiguity
from transformation_library import find_connected_objects

class Task22168020Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input is a square grid with size given by {vars['rows']} x {vars['rows']}.",
            "The grid contains one or more monochromatic objects (colors 1-9) and empty background cells (0).",
            "Each object uses a unique color (no two objects share the same color).",
            "Each object consists of a 2x2 base square plus two symmetric diagonal arms that extend upward from the top-left and top-right corners.",
            "Each arm extends diagonally by the same length for that object (allowed lengths: 1, 2, or 3), and all cells of an object (square and arms) have the object color.",
            "Objects are placed fully inside the grid and do not overlap or share cells; all other grid cells are background (0)."
        ]

        transformation_reasoning_chain = [
            "The output grid has the same dimensions as the input and starts as a copy of the input.",
            "Identify each 8-way (diagonally allowed) connected monochromatic object in the input.",
            "For each object, process every row that intersects the object: find the leftmost and rightmost object cells on that row.",
            "Fill all cells between the leftmost and rightmost columns (inclusive) on that row with the object color.",
            "The result is a solid horizontal fill between an object arms and base, producing a filled, gap-free shape for each object."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        grid = np.zeros((rows, rows), dtype=int)
        
        # Try to place as many objects as possible (up to 9, limited by available colors)
        max_possible_objects = min(9, rows // 5)  # Rough estimate based on grid size
        num_objects = random.randint(1, max_possible_objects)
        
        used_positions = set()
        filled_areas = set()  # Track areas that will be filled in transformation
        used_colors = set()  # Track colors already used
        
        max_attempts = 100
        objects_placed = 0
        
        for _ in range(num_objects):
            # Select a color that hasn't been used yet
            available_colors = [c for c in range(1, 10) if c not in used_colors]
            if not available_colors:
                break  # No more unique colors available
            
            color = random.choice(available_colors)
            arm_length = random.randint(1, 3)
            
            valid = False
            attempts = 0
            while not valid and attempts < max_attempts:
                attempts += 1
                start_row = random.randint(arm_length, rows - 2)
                start_col = random.randint(arm_length, rows - 3)
                
                object_cells = set()
                area_cells = set()  # Track all cells that will be filled after transformation
                
                # Add 2x2 square cells
                for r in range(2):
                    for c in range(2):
                        object_cells.add((start_row + r, start_col + c))
                        area_cells.add((start_row + r, start_col + c))
                
                # Add diagonal arm cells and calculate fill area
                valid_arms = True
                for i in range(1, arm_length + 1):
                    left_arm = (start_row - i, start_col - i)
                    right_arm = (start_row - i, start_col + 1 + i)
                    
                    # Verify arms are within grid bounds
                    if (left_arm[0] < 0 or left_arm[1] < 0 or 
                        right_arm[0] < 0 or right_arm[1] >= rows):
                        valid_arms = False
                        break
                    
                    object_cells.add(left_arm)
                    object_cells.add(right_arm)
                    
                    # Add all cells in the row between arms to area_cells
                    for c in range(left_arm[1], right_arm[1] + 1):
                        area_cells.add((left_arm[0], c))
                
                # Add filled area for the 2x2 square rows
                for r in range(start_row, start_row + 2):
                    row_cells_in_obj = [c for row, c in object_cells if row == r]
                    if row_cells_in_obj:
                        min_c = min(row_cells_in_obj)
                        max_c = max(row_cells_in_obj)
                        for c in range(min_c, max_c + 1):
                            area_cells.add((r, c))
                
                if not valid_arms:
                    continue
                
                # Check if either the object or its filled area overlaps with existing objects
                if (not any(pos in used_positions for pos in object_cells) and 
                    not any(pos in filled_areas for pos in area_cells)):
                    valid = True
                    used_positions.update(object_cells)
                    filled_areas.update(area_cells)
                    used_colors.add(color)  # Mark this color as used
                    for r, c in object_cells:
                        grid[r, c] = color
                    objects_placed += 1
            
            if attempts >= max_attempts:
                # If we can't place this object, continue trying with other colors
                continue
        
        return grid

    def transform_input(self, grid, taskvars):
        output_grid = grid.copy()
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0, monochromatic=True)
        
        for obj in objects:
            coords = list(obj.coords)
            color = list(obj.colors)[0]
            
            # Find min and max row
            min_row = min(r for r, c in coords)
            max_row = max(r for r, c in coords)
            
            # For each row in the object
            for row in range(min_row, max_row + 1):
                # Find all points in this row
                points_in_row = [c for r, c in coords if r == row]
                
                if points_in_row:
                    # Find leftmost and rightmost columns
                    left_col = min(points_in_row)
                    right_col = max(points_in_row)
                    
                    # Fill all cells between left and right (inclusive)
                    for col in range(left_col, right_col + 1):
                        output_grid[row, col] = color
        
        return output_grid

    def create_grids(self):
        taskvars = {
            'rows': random.choice([i for i in range(10, 30) if i % 2 == 0])
        }
        
        train_data = []
        for _ in range(random.randint(3, 4)):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})
        
        test_input_grid = self.create_input(taskvars, {})
        test_output_grid = self.transform_input(test_input_grid, taskvars)
        test_data = [{'input': test_input_grid, 'output': test_output_grid}]
        
        return taskvars, {'train': train_data, 'test': test_data}