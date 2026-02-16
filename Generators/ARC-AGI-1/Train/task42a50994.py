from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple, List
from Framework.input_library import create_object, random_cell_coloring, Contiguity
from Framework.transformation_library import find_connected_objects

class Task42a50994Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain same-colored (1-9) and empty (0) cells.",
            "The colored cells appear mostly as single cells and sometimes as objects made by 8-way connected cells.",
            "The colors of the grid vary across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identfying all colored cells that appear as single cells.",
            "Once identified, remove all single-colored cells from the output grid while keeping the colored objects unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'rows': random.randint(6, 30),
            'cols': random.randint(6, 30)
        }
        
        # Generate 3-4 training examples and 1 test example
        num_train_examples = random.randint(3, 4)
        
        # Create training and test grids
        train_pairs = []
        used_colors = []
        
        for _ in range(num_train_examples):
            # Choose different color for each example
            available_colors = [c for c in range(1, 10) if c not in used_colors]
            color = random.choice(available_colors)
            used_colors.append(color)
            
            # Generate input grid
            gridvars = {'color': color}
            input_grid = self.create_input(taskvars, gridvars)
            
            # Transform input grid to generate output grid
            output_grid = self.transform_input(input_grid, taskvars)
            
            # Verify that the output is not empty
            if np.any(output_grid != 0):
                train_pairs.append({'input': input_grid, 'output': output_grid})
            else:
                # Try again with the same color
                used_colors.pop()
                _-=1
        
        # Create test example with a different color
        available_colors = [c for c in range(1, 10) if c not in used_colors]
        test_color = random.choice(available_colors) if available_colors else random.randint(1, 9)
        
        gridvars = {'color': test_color}
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        # Ensure test output is not empty
        while not np.any(test_output != 0):
            test_input = self.create_input(taskvars, gridvars)
            test_output = self.transform_input(test_input, taskvars)
        
        test_pairs = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_pairs, 'test': test_pairs}
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows, cols = taskvars['rows'], taskvars['cols']
        color = gridvars['color']
        
        # Initialize empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Ensure we create at least 3 multi-cell objects
        num_connected_objects = random.randint(3, 8)
        
        # Create connected objects first
        for _ in range(num_connected_objects):
            # Create a small object (2-4 cells)
            obj_size = random.randint(2, 4)
            
            # Create a connected object directly using predefined patterns
            object_patterns = [
                # 2-cell patterns
                np.array([[color, color]]),
                np.array([[color], [color]]),
                np.array([[color, color], [0, 0]]),
                np.array([[0, 0], [color, color]]),
                np.array([[color, 0], [color, 0]]),
                np.array([[0, color], [0, color]]),
                np.array([[color, 0], [0, color]]),
                np.array([[0, color], [color, 0]]),
                
                # 3-cell patterns
                np.array([[color, color, color]]),
                np.array([[color], [color], [color]]),
                np.array([[color, color], [color, 0]]),
                np.array([[color, color], [0, color]]),
                np.array([[color, 0], [color, color]]),
                np.array([[0, color], [color, color]]),
                np.array([[color, 0, 0], [0, color, color]]),
                np.array([[0, 0, color], [color, color, 0]]),
                
                # 4-cell patterns
                np.array([[color, color], [color, color]]),
                np.array([[color, color, color, color]]),
                np.array([[color], [color], [color], [color]]),
                np.array([[color, color, 0], [0, color, color]]),
                np.array([[0, color, color], [color, color, 0]]),
                np.array([[color, 0], [color, color], [0, color]]),
                np.array([[0, color], [color, color], [color, 0]])
            ]
            
            # Filter patterns based on desired size
            valid_patterns = [p for p in object_patterns if np.sum(p == color) <= obj_size]
            obj = random.choice(valid_patterns)
            
            h, w = obj.shape
            
            # Try to place the object at a random position without overlapping
            max_attempts = 50
            placed = False
            
            for _ in range(max_attempts):
                r = random.randint(0, rows - h)
                c = random.randint(0, cols - w)
                
                # Check if placement area is empty
                placement_area = grid[r:r+h, c:c+w]
                if np.all(placement_area == 0):
                    # Place the object
                    for i in range(h):
                        for j in range(w):
                            if obj[i, j] != 0:
                                grid[r+i, c+j] = obj[i, j]
                    placed = True
                    break
            
            # If we couldn't place it after max attempts, force placement (possibly overlapping)
            if not placed:
                r = random.randint(0, rows - h)
                c = random.randint(0, cols - w)
                for i in range(h):
                    for j in range(w):
                        if obj[i, j] != 0 and grid[r+i, c+j] == 0:
                            grid[r+i, c+j] = obj[i, j]
        
        # Verify we have at least 3 connected objects
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)
        connected_objects = [obj for obj in objects if len(obj) > 1]
        
        # If we don't have enough connected objects, try again
        if len(connected_objects) < 3:
            return self.create_input(taskvars, gridvars)
        
        # Now add single cells (5-15 cells)
        num_singles = random.randint(5, 15)
        singles_added = 0
        max_attempts = 100
        
        for _ in range(max_attempts):
            if singles_added >= num_singles:
                break
                
            r = random.randint(0, rows - 1)
            c = random.randint(0, cols - 1)
            
            # Check if cell is empty and not adjacent to any colored cell
            if grid[r, c] == 0:
                is_isolated = True
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != 0:
                            is_isolated = False
                            break
                    if not is_isolated:
                        break
                
                # If isolated, place a single cell
                if is_isolated:
                    grid[r, c] = color
                    singles_added += 1
        
        # If we couldn't place enough singles, add some randomly
        if singles_added < num_singles:
            empty_positions = list(zip(*np.where(grid == 0)))
            random.shuffle(empty_positions)
            for r, c in empty_positions[:num_singles - singles_added]:
                grid[r, c] = color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Create a copy of the input grid
        output_grid = grid.copy()
        
        # Find connected objects
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)
        
        # Keep only objects with more than one cell
        for obj in objects:
            if len(obj) == 1:  # This is a single cell
                r, c, _ = next(iter(obj.cells))
                output_grid[r, c] = 0  # Remove from output grid
        
        return output_grid

