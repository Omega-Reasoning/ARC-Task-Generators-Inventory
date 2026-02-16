from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, List, Any, Tuple

from input_library import create_object, Contiguity, random_cell_coloring
from transformation_library import find_connected_objects, BorderBehavior

class Task4c4377d9Generator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain a completely filled grid using two distinct colors, with the colors varying across examples.",
            "One color serves as the background, while the other forms one or more objects made of 8-way connected cells.",
            "One of these objects is always positioned so that it touches the top edge of the grid."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are of size {2*vars['rows']}x{vars['cols']}.",
            "They are constructed by copying the input grid and pasting it into the lower half of the output grid.",
            "The newly added lower half is then reflected onto the upper half of the grid, using the vertical center as the line of reflection."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        
        rows = random.randint(5, 15)  # Keep reasonable size
        cols = random.randint(5, 30)
        
        taskvars = {'rows': rows, 'cols': cols}
        
        # Generate 3-4 train examples with different colors
        num_train_examples = random.randint(3, 4)
        train_examples = []
        
        used_colors = set()
        
        for _ in range(num_train_examples):
            # Choose two distinct colors that haven't been used together before
            while True:
                fg_color = random.randint(1, 9)
                bg_color = random.randint(1, 9)
                if fg_color != bg_color and (fg_color, bg_color) not in used_colors:
                    used_colors.add((fg_color, bg_color))
                    break
            
            gridvars = {'fg_color': fg_color, 'bg_color': bg_color}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example with different colors
        while True:
            fg_color = random.randint(1, 9)
            bg_color = random.randint(1, 9)
            if fg_color != bg_color and (fg_color, bg_color) not in used_colors:
                break
        
        gridvars = {'fg_color': fg_color, 'bg_color': bg_color}
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        
        fg_color = gridvars['fg_color']
        bg_color = gridvars['bg_color']
        
        # Create a grid filled with background color
        grid = np.full((rows, cols), bg_color, dtype=int)
        
        # Decide if we want one or two objects
        num_objects = random.choice([1, 2])
        
        # Calculate the number of cells to fill (between 1/3 and 1/2 of total cells)
        total_cells = rows * cols
        min_cells = int(total_cells / 3)
        max_cells = int(total_cells / 2)
        
        # First, create an object that touches the top edge
        top_row_indices = [(0, c) for c in range(cols)]
        start_point = random.choice(top_row_indices)
        grid[start_point] = fg_color
        
        # Grow this object
        cells_to_fill = random.randint(min_cells // 2, max_cells // 2 if num_objects == 2 else max_cells)
        cells_filled = 1  # Start with one cell filled
        
        for _ in range(min(rows*cols, 1000)):  # Limit iterations
            if cells_filled >= cells_to_fill:
                break
                
            # Find all objects
            objects = find_connected_objects(grid, diagonal_connectivity=True, background=bg_color)
            
            # Find cells adjacent to existing objects
            all_neighbors = set()
            for obj in objects:
                for r, c, _ in obj.cells:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < rows and 0 <= nc < cols and 
                                grid[nr, nc] == bg_color):
                                all_neighbors.add((nr, nc))
            
            if not all_neighbors:
                break
                
            # Add a random neighboring cell
            r, c = random.choice(list(all_neighbors))
            grid[r, c] = fg_color
            cells_filled += 1
        
        # If we need a second object, create it (making sure it doesn't touch the first one)
        if num_objects == 2:
            objects = find_connected_objects(grid, diagonal_connectivity=True, background=bg_color)
            
            # Make sure we have at least one object
            if objects:
                first_obj = objects[0]
                
                # Find a valid starting point for second object (not touching first object)
                valid_start_points = []
                for r in range(rows):
                    for c in range(cols):
                        if grid[r, c] == bg_color:
                            # Check if this point is not adjacent to the first object
                            is_valid = True
                            for dr in [-1, 0, 1]:
                                for dc in [-1, 0, 1]:
                                    nr, nc = r + dr, c + dc
                                    if (0 <= nr < rows and 0 <= nc < cols and 
                                        (nr, nc) in first_obj.coords):
                                        is_valid = False
                                        break
                                if not is_valid:
                                    break
                            if is_valid:
                                valid_start_points.append((r, c))
                
                if valid_start_points:
                    # Start the second object
                    r, c = random.choice(valid_start_points)
                    grid[r, c] = fg_color
                    cells_filled += 1
                    
                    # Grow the second object
                    cells_to_fill = random.randint(min(min_cells // 3, 2), min(max_cells // 3, rows*cols//10))
                    
                    for _ in range(min(rows*cols, 500)):  # Limit iterations
                        if cells_filled >= cells_to_fill + (min_cells // 2):
                            break
                            
                        objects = find_connected_objects(grid, diagonal_connectivity=True, background=bg_color)
                        
                        # If we somehow ended up with more than 2 objects, stop
                        if len(objects) > 2:
                            break
                            
                        # Find the second object
                        second_obj = None
                        for obj in objects:
                            if not any((0, c) in obj.coords for c in range(cols)):
                                second_obj = obj
                                break
                        
                        if not second_obj:
                            break
                            
                        # Find neighbors of second object that don't touch the first object
                        valid_neighbors = set()
                        for r, c, _ in second_obj.cells:
                            for dr in [-1, 0, 1]:
                                for dc in [-1, 0, 1]:
                                    nr, nc = r + dr, c + dc
                                    if (0 <= nr < rows and 0 <= nc < cols and 
                                        grid[nr, nc] == bg_color):
                                        
                                        # Check if this neighbor would touch the first object
                                        touches_first = False
                                        for ddr in [-1, 0, 1]:
                                            for ddc in [-1, 0, 1]:
                                                nnr, nnc = nr + ddr, nc + ddc
                                                if (0 <= nnr < rows and 0 <= nnc < cols and
                                                    (nnr, nnc) in first_obj.coords):
                                                    touches_first = True
                                                    break
                                            if touches_first:
                                                break
                                                
                                        if not touches_first:
                                            valid_neighbors.add((nr, nc))
                        
                        if not valid_neighbors:
                            break
                            
                        # Add a random valid neighbor
                        r, c = random.choice(list(valid_neighbors))
                        grid[r, c] = fg_color
                        cells_filled += 1
        
        # Verify that we have exactly one or two objects
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=bg_color)
        
        # If we still don't have the right number of objects, try again with a simpler approach
        retry_attempts = 0
        while len(objects) > 2 and retry_attempts < 5:
            # Reset grid
            grid = np.full((rows, cols), bg_color, dtype=int)
            
            # Create first object (simpler approach)
            for r in range(rows//3):
                for c in range(cols//3, 2*cols//3):
                    if random.random() < 0.7:  # Probability to fill a cell
                        grid[r, c] = fg_color
            
            # Ensure top edge is touched
            grid[0, cols//2] = fg_color
            
            # If we want two objects, create a second one away from the first
            if num_objects == 2:
                for r in range(2*rows//3, rows):
                    for c in range(cols//3, 2*cols//3):
                        if random.random() < 0.7:  # Probability to fill a cell
                            grid[r, c] = fg_color
            
            objects = find_connected_objects(grid, diagonal_connectivity=True, background=bg_color)
            retry_attempts += 1
        
        # Final check - if we still don't have the right number, create a very simple solution
        if len(objects) != num_objects:
            grid = np.full((rows, cols), bg_color, dtype=int)
            
            # Create a simple rectangular object touching the top
            height = min(rows//2, 4)
            width = min(cols//2, 4)
            col_start = random.randint(0, cols-width)
            grid[0:height, col_start:col_start+width] = fg_color
            
            # If we need two objects, add a second one
            if num_objects == 2:
                height2 = min(rows//3, 3)
                width2 = min(cols//3, 3)
                row_start = random.randint(height+2, rows-height2-1)
                col_start = random.randint(0, cols-width2)
                grid[row_start:row_start+height2, col_start:col_start+width2] = fg_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        
        # Create a new grid with double the height
        output_grid = np.zeros((2 * rows, cols), dtype=int)
        
        # Copy the input grid to the bottom half
        output_grid[rows:, :] = grid
        
        # Reflect bottom half to top half (with vertical center as reflection line)
        for r in range(rows):
            for c in range(cols):
                output_grid[rows - r - 1, c] = output_grid[rows + r, c]
        
        return output_grid

