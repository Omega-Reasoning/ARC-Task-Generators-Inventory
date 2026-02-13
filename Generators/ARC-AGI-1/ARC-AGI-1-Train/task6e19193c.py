from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, List, Any, Tuple

class Task6e19193cGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']} x {vars['grid_size']}.",
            "The grid contains two or three objects, with the objects being upward and downward arrowheads.", 
            "An upward arrowhead object is defined as [[c, c], [0, c]] or [[c, c], [c, 0]] for a color c., and a downward arrowhead object is defined as [[0, c], [c, c]] or [[c, 0], [c, c]] for a color c.",
            "The upward and downward arrowhead objects should never be placed on the border of the grid.", 
            "They must be positioned so that if a continuous diagonal line were added to complete the arrow shape, it would not be interrupted by another arrowhead or its respective diagonal line, and only by the grid border.",
            "The color of the arrow heads should be the same within the grid and should change across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by identifying the upward and downward arrowheads.",
            "An upward arrowhead object is defined as [[c, c], [0, c]] or [[c, c], [c, 0]] for a color c., and a downward arrowhead object is defined as [[0, c], [c, c]] or [[c, 0], [c, c]] for a color c.",
            "Once identified, complete the arrow shape by adding a continuous diagonal line.",
            "For an upward arrowhead, extend a diagonal line starting from the cell after the bottom-left or bottom-right corner, going towards bottom-left or  bottom-right respectively.", 
            "For a downward arrowhead, extend the diagonal from the cell after the top-right or top-left corner, going toward the top-right or top-right respectively.", 
            "The bottom-left or bottom-right (for upward) and top-right or top-left(for downward) corners are left empty (0).",
            "The line should continue in that direction until it reaches the grid border."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Define task variables (grid size)
        grid_size = random.randint(10,30)  # Keep grid size reasonable
        taskvars = {'grid_size': grid_size}
        
        # Generate training examples (3-4)
        num_train_examples = random.randint(3, 4)
        train_examples = []
        
        # Use different colors for each example
        colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], num_train_examples + 1)
        
        for i in range(num_train_examples):
            gridvars = {'color': colors[i], 'is_test': False}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example with exactly 3 arrowheads
        test_gridvars = {'color': colors[-1], 'is_test': True}
        test_input_grid = self.create_input(taskvars, test_gridvars)
        test_output_grid = self.transform_input(test_input_grid, taskvars)
        test_examples = [{'input': test_input_grid, 'output': test_output_grid}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        arrow_color = gridvars['color']
        is_test = gridvars.get('is_test', False)
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Number of arrowheads: 2 for train examples, 3 for test
        num_arrowheads = 3 if is_test else 2
        
        # Define the arrowhead types
        arrowhead_types = [
            # Upward-left: [[c, c], [0, c]]
            (np.array([[arrow_color, arrow_color], [0, arrow_color]]), "upward-left"),
            # Upward-right: [[c, c], [c, 0]]
            (np.array([[arrow_color, arrow_color], [arrow_color, 0]]), "upward-right"),
            # Downward-left: [[0, c], [c, c]]
            (np.array([[0, arrow_color], [arrow_color, arrow_color]]), "downward-left"),
            # Downward-right: [[c, 0], [c, c]]
            (np.array([[arrow_color, 0], [arrow_color, arrow_color]]), "downward-right")
        ]
        
        # Define quadrant boundaries
        quadrants = [
            (grid_size//2, grid_size-3, 2, grid_size//2-2),           # Bottom-left for upward-left
            (grid_size//2, grid_size-3, grid_size//2, grid_size-3),   # Bottom-right for upward-right
            (2, grid_size//2-2, 2, grid_size//2-2),                   # Top-left for downward-left
            (2, grid_size//2-2, grid_size//2, grid_size-3)            # Top-right for downward-right
        ]
        
        # Keep track of placed arrowheads
        placed_count = 0
        placed_positions = set()
        
        # Shuffled list of arrowhead indices, prioritizing variety
        arrowhead_indices = list(range(4))
        random.shuffle(arrowhead_indices)
        
        # For test examples, always try to use 3 different types
        if is_test:
            arrowhead_indices = arrowhead_indices[:3]
        
        # Try to place arrowheads
        for idx in arrowhead_indices:
            if placed_count >= num_arrowheads:
                break
                
            arrowhead, arrow_type = arrowhead_types[idx]
            min_row, max_row, min_col, max_col = quadrants[idx]
            
            # Make quadrant smaller if grid is small
            if max_row - min_row < 4 or max_col - min_col < 4:
                continue  # Skip if region is too small
            
            # Multiple attempts for this arrowhead type
            for _ in range(20):  # Limit attempts per type
                row = random.randint(min_row, max_row)
                col = random.randint(min_col, max_col)
                
                # Check if this position or its diagonal would overlap with existing arrowheads
                position_valid = True
                
                # Check immediate area first
                for r in range(max(0, row-2), min(grid_size, row+4)):
                    for c in range(max(0, col-2), min(grid_size, col+4)):
                        if (r, c) in placed_positions:
                            position_valid = False
                            break
                    if not position_valid:
                        break
                
                if not position_valid:
                    continue
                
                # Calculate and check diagonal path
                diagonal_cells = set()
                if arrow_type == "upward-left":
                    r, c = row + 2, col - 1
                    while 0 <= r < grid_size and 0 <= c < grid_size:
                        diagonal_cells.add((r, c))
                        r += 1
                        c -= 1
                elif arrow_type == "upward-right":
                    r, c = row + 2, col + 2
                    while 0 <= r < grid_size and 0 <= c < grid_size:
                        diagonal_cells.add((r, c))
                        r += 1
                        c += 1
                elif arrow_type == "downward-left":
                    r, c = row - 1, col - 1
                    while 0 <= r < grid_size and 0 <= c < grid_size:
                        diagonal_cells.add((r, c))
                        r -= 1
                        c -= 1
                elif arrow_type == "downward-right":
                    r, c = row - 1, col + 2
                    while 0 <= r < grid_size and 0 <= c < grid_size:
                        diagonal_cells.add((r, c))
                        r -= 1
                        c += 1
                
                # Check diagonal against placed positions
                if any((r, c) in placed_positions for r, c in diagonal_cells):
                    continue
                
                # We've found a valid position - place the arrowhead
                grid[row:row+2, col:col+2] = arrowhead
                
                # Mark cells as placed
                for r in range(row, row+2):
                    for c in range(col, col+2):
                        placed_positions.add((r, c))
                for r, c in diagonal_cells:
                    placed_positions.add((r, c))
                
                placed_count += 1
                break  # Successfully placed this arrowhead
        
        # Check if we placed enough arrowheads
        if placed_count < num_arrowheads:
            # Create grid with predetermined positions for guaranteed placement
            grid = np.zeros((grid_size, grid_size), dtype=int)
            
            if not is_test:
                # For training examples, place 2 arrowheads
                pos1 = (grid_size//4, grid_size//4)
                pos2 = (3*grid_size//4, 3*grid_size//4)
                
                # Place upward-left and downward-right arrowheads
                grid[pos1[0]:pos1[0]+2, pos1[1]:pos1[1]+2] = arrowhead_types[0][0]  # upward-left
                grid[pos2[0]:pos2[0]+2, pos2[1]:pos2[1]+2] = arrowhead_types[3][0]  # downward-right
            else:
                # For test examples, place 3 arrowheads
                # Use three fixed positions spread across the grid
                positions = [
                    (grid_size//5, grid_size//5),                   # Top-left
                    (grid_size//2, 3*grid_size//4),                 # Middle-bottom
                    (grid_size//5, 3*grid_size//4)                  # Bottom-left
                ]
                
                # Place three different types of arrowheads
                grid[positions[0][0]:positions[0][0]+2, positions[0][1]:positions[0][1]+2] = arrowhead_types[0][0]  # upward-left
                grid[positions[1][0]:positions[1][0]+2, positions[1][1]:positions[1][1]+2] = arrowhead_types[1][0]  # upward-right
                grid[positions[2][0]:positions[2][0]+2, positions[2][1]:positions[2][1]+2] = arrowhead_types[2][0]  # downward-left
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        grid_size = grid.shape[0]
        output_grid = grid.copy()
        
        # Scan the grid to find all arrowheads
        for row in range(grid_size - 1):
            for col in range(grid_size - 1):
                # Get the 2x2 subgrid
                if row + 1 >= grid_size or col + 1 >= grid_size:
                    continue
                    
                subgrid = grid[row:row+2, col:col+2]
                
                # Skip if all zeros
                if np.all(subgrid == 0):
                    continue
                
                arrow_color = 0
                arrow_type = None
                
                # Check for upward-left arrowhead: [[c, c], [0, c]]
                if (subgrid[0, 0] > 0 and subgrid[0, 0] == subgrid[0, 1] == subgrid[1, 1] and subgrid[1, 0] == 0):
                    arrow_color = subgrid[0, 0]
                    arrow_type = "upward-left"
                
                # Check for upward-right arrowhead: [[c, c], [c, 0]]
                elif (subgrid[0, 0] > 0 and subgrid[0, 0] == subgrid[0, 1] == subgrid[1, 0] and subgrid[1, 1] == 0):
                    arrow_color = subgrid[0, 0]
                    arrow_type = "upward-right"
                
                # Check for downward-left arrowhead: [[0, c], [c, c]]
                elif (subgrid[0, 1] > 0 and subgrid[0, 1] == subgrid[1, 0] == subgrid[1, 1] and subgrid[0, 0] == 0):
                    arrow_color = subgrid[0, 1]
                    arrow_type = "downward-left"
                
                # Check for downward-right arrowhead: [[c, 0], [c, c]]
                elif (subgrid[0, 0] > 0 and subgrid[0, 0] == subgrid[1, 0] == subgrid[1, 1] and subgrid[0, 1] == 0):
                    arrow_color = subgrid[0, 0]
                    arrow_type = "downward-right"
                
                # Draw diagonal line based on arrow type
                if arrow_type == "upward-left":
                    # Diagonal goes bottom-left from [row+1, col]
                    r, c = row + 2, col - 1
                    while 0 <= r < grid_size and 0 <= c < grid_size:
                        output_grid[r, c] = arrow_color
                        r += 1
                        c -= 1
                
                elif arrow_type == "upward-right":
                    # Diagonal goes bottom-right from [row+1, col+1]
                    r, c = row + 2, col + 2
                    while 0 <= r < grid_size and 0 <= c < grid_size:
                        output_grid[r, c] = arrow_color
                        r += 1
                        c += 1
                
                elif arrow_type == "downward-left":
                    # Diagonal goes top-left from [row, col]
                    r, c = row - 1, col - 1
                    while 0 <= r < grid_size and 0 <= c < grid_size:
                        output_grid[r, c] = arrow_color
                        r -= 1
                        c -= 1
                
                elif arrow_type == "downward-right":
                    # Diagonal goes top-right from [row, col+1]
                    r, c = row - 1, col + 2
                    while 0 <= r < grid_size and 0 <= c < grid_size:
                        output_grid[r, c] = arrow_color
                        r -= 1
                        c += 1
        
        return output_grid

