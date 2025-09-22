from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple

class Task4odGuajvdxnCdJktv8MqgLGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['rows']} × {vars['cols']}.",
            "Each input grid contains some single multi-colored (1–9) cells, which are completely separated from each other by empty (0) cells."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by expanding each colored cell into a plus-shaped object of the same color.",
            "The plus shape is formed by adding one colored cell above, below, left, and right of the original cell.",
            "If there is no space to add a cell because of the grid boundary, the shape is left as is.",
            "If another colored cell is already present, overlaps are resolved based on priority.",
            "The priority rule is that: Left overrides right and top overrides bottom."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows, cols = taskvars['rows'], taskvars['cols']
        grid = np.zeros((rows, cols), dtype=int)
        
        # Generate 2-8 separated colored cells
        num_cells = random.randint(2, 8)
        colors = random.sample(range(1, 10), min(num_cells, 9))  # Ensure unique colors when possible
        
        placed_cells = set()
        
        for i in range(num_cells):
            # Try to place a cell that's separated from others
            attempts = 0
            while attempts < 100:  # Prevent infinite loop
                r = random.randint(0, rows - 1)
                c = random.randint(0, cols - 1)
                
                # Check if this position and its neighbors are free
                valid = True
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if (nr, nc) in placed_cells:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    grid[r, c] = colors[i % len(colors)]
                    placed_cells.add((r, c))
                    break
                
                attempts += 1
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows, cols = grid.shape
        output_grid = grid.copy()
        
        # Find all colored cells in the input
        colored_cells = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0:
                    colored_cells.append((r, c, grid[r, c]))
        
        # For each colored cell, try to add plus shape
        for r, c, color in colored_cells:
            # Define the plus shape directions: up, down, left, right
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                # Check if the new position is within bounds
                if 0 <= nr < rows and 0 <= nc < cols:
                    # Apply priority rules for conflicts
                    current_color = output_grid[nr, nc]
                    
                    if current_color == 0:
                        # Empty cell, place the color
                        output_grid[nr, nc] = color
                    else:
                        # There's already a color, apply priority rules
                        # Left overrides right and top overrides bottom
                        
                        # Find the original position of the current color
                        original_pos = None
                        for orig_r, orig_c, orig_color in colored_cells:
                            if orig_color == current_color:
                                # Check if this cell is part of the plus from orig_color
                                if ((orig_r == nr and abs(orig_c - nc) == 1) or 
                                    (orig_c == nc and abs(orig_r - nr) == 1)):
                                    original_pos = (orig_r, orig_c)
                                    break
                        
                        if original_pos:
                            orig_r, orig_c = original_pos
                            
                            # Apply priority rules:
                            # Left overrides right: if new color comes from left, it wins
                            # Top overrides bottom: if new color comes from top, it wins
                            
                            new_wins = False
                            
                            # Check horizontal priority (left overrides right)
                            if r == orig_r:  # Same row, horizontal conflict
                                if c < orig_c:  # New color is from the left
                                    new_wins = True
                            
                            # Check vertical priority (top overrides bottom)
                            elif c == orig_c:  # Same column, vertical conflict
                                if r < orig_r:  # New color is from above
                                    new_wins = True
                            
                            # For diagonal conflicts, prioritize by position
                            else:
                                # If new color's original position is more top-left
                                if r < orig_r or (r == orig_r and c < orig_c):
                                    new_wins = True
                            
                            if new_wins:
                                output_grid[nr, nc] = color
        
        return output_grid
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables with random grid size between 5 and 15
        rows = random.randint(5, 15)
        cols = random.randint(5, 15)
        taskvars = {'rows': rows, 'cols': cols}
        
        # Generate 3-5 training examples and 1 test example
        num_train = random.randint(3, 5)
        train_test_data = self.create_grids_default(num_train, 1, taskvars)
        
        return taskvars, train_test_data

