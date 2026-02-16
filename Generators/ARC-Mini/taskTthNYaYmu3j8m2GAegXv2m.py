from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import create_object, retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class TaskTthNYaYmu3j8m2GAegXv2mGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['rows']} × {vars['cols']}.",
            "Each input grid contains several differently colored single-cells, completely separated from each other by empty cells.",
            "All other cells are empty."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and expanding each colored cell by adding the eight surrounding cells (top, bottom, left, right, and the four diagonals).",
            "New cells are only added if there is space; otherwise, they are not added. For example, if a colored cell is near the boundary, it can only expand into the available cells.",
            "When multiple expansions overlap, the conflict is resolved by color priority based on the relative position of the original cells; left cells have priority over right cells. Top cells have priority over bottom cells."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        
        grid = np.zeros((rows, cols), dtype=int)
        
        # Get colors and positions from gridvars if specified, otherwise generate randomly
        if 'positions' in gridvars and 'colors' in gridvars:
            positions = gridvars['positions']
            colors = gridvars['colors']
        else:
            # Generate random separated positions
            num_cells = random.randint(2, min(6, rows * cols // 9))  # Ensure enough space for expansion
            colors = random.sample(range(1, 10), num_cells)
            
            positions = []
            for color in colors:
                attempts = 0
                while attempts < 100:
                    r = random.randint(0, rows - 1)
                    c = random.randint(0, cols - 1)
                    
                    # Check if position is valid (not occupied and sufficiently separated)
                    if grid[r, c] == 0:
                        # Check if separated from other cells (at least 2 cells apart)
                        valid = True
                        for pr, pc in positions:
                            if abs(r - pr) < 2 or abs(c - pc) < 2:
                                valid = False
                                break
                        if valid:
                            positions.append((r, c))
                            grid[r, c] = color
                            break
                    attempts += 1
        
        # Place cells from gridvars if provided
        if 'positions' in gridvars:
            for (r, c), color in zip(positions, colors):
                if 0 <= r < rows and 0 <= c < cols:
                    grid[r, c] = color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output = grid.copy()
        rows, cols = grid.shape
        
        # Find all colored cells with their positions
        colored_cells = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0:
                    colored_cells.append((r, c, grid[r, c]))
        
        # Create a priority map for conflict resolution
        # Lower values = higher priority
        priority_map = np.full((rows, cols), float('inf'))
        for r, c, color in colored_cells:
            priority_map[r, c] = r * cols + c  # Top-left has lower values (higher priority)
        
        # Expansion directions: 8 neighbors
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        # Track which cells are claimed and by which original cell
        claims = {}
        
        # For each colored cell, try to expand in all 8 directions
        for r, c, color in colored_cells:
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                # Check bounds
                if 0 <= nr < rows and 0 <= nc < cols:
                    # Only expand into empty cells or resolve conflicts
                    if grid[nr, nc] == 0:  # Empty cell
                        if (nr, nc) not in claims:
                            # Unclaimed cell - claim it
                            claims[(nr, nc)] = (r, c, color)
                        else:
                            # Conflict - resolve by priority
                            existing_r, existing_c, existing_color = claims[(nr, nc)]
                            current_priority = priority_map[r, c]
                            existing_priority = priority_map[existing_r, existing_c]
                            
                            if current_priority < existing_priority:
                                # Current cell has higher priority
                                claims[(nr, nc)] = (r, c, color)
        
        # Apply all claims to the output grid
        for (nr, nc), (orig_r, orig_c, color) in claims.items():
            output[nr, nc] = color
        
        return output

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)
        taskvars = {'rows': rows, 'cols': cols}
        
        # Generate training examples
        num_train = random.randint(3, 5)
        train_pairs = []
        
        # Track diagonal arrangements: 0 = top-left, 1 = top-right
        diagonal_arrangements = [False, False]
        
        for i in range(num_train):
            gridvars = {}
            
            # Ensure we have both diagonal arrangements in the first examples
            if i < 4 and not all(diagonal_arrangements):
                # Choose which diagonal arrangement to create
                if i < 2:
                    diagonal_type = i  # First two: top-left (0), top-right (1)
                else:
                    # For remaining slots, pick missing arrangements
                    diagonal_type = 0 if not diagonal_arrangements[0] else 1
                
                attempts = 0
                while attempts < 50:
                    # Position A
                    if diagonal_type == 0:  # top-left arrangement
                        a_r = random.randint(2, rows - 3)
                        a_c = random.randint(2, cols - 3)
                        # Position for empty cell (top-left of A)
                        empty_r, empty_c = a_r - 1, a_c - 1
                        # Position B (top-left of empty cell)
                        b_r, b_c = empty_r - 1, empty_c - 1
                    else:  # top-right arrangement
                        a_r = random.randint(2, rows - 3)
                        a_c = random.randint(2, cols - 3)
                        # Position for empty cell (top-right of A)
                        empty_r, empty_c = a_r - 1, a_c + 1
                        # Position B (top-right of empty cell)
                        b_r, b_c = empty_r - 1, empty_c + 1
                    
                    # Check if all positions are valid
                    if (0 <= b_r < rows and 0 <= b_c < cols and
                        0 <= empty_r < rows and 0 <= empty_c < cols):
                        
                        # Add some random other cells
                        num_other = random.randint(1, 3)
                        positions = [(a_r, a_c), (b_r, b_c)]
                        colors = [random.randint(1, 9), random.randint(1, 9)]
                        
                        # Ensure colors are different
                        while colors[1] == colors[0]:
                            colors[1] = random.randint(1, 9)
                        
                        # Add other random cells that don't interfere
                        for _ in range(num_other):
                            attempts_inner = 0
                            while attempts_inner < 20:
                                r = random.randint(0, rows - 1)
                                c = random.randint(0, cols - 1)
                                
                                # Check separation from existing cells
                                valid = True
                                for pr, pc in positions:
                                    if abs(r - pr) < 2 or abs(c - pc) < 2:
                                        valid = False
                                        break
                                
                                if valid and (r, c) != (empty_r, empty_c):
                                    positions.append((r, c))
                                    color = random.randint(1, 9)
                                    while color in colors:
                                        color = random.randint(1, 9)
                                    colors.append(color)
                                    break
                                
                                attempts_inner += 1
                        
                        gridvars = {'positions': positions, 'colors': colors}
                        diagonal_arrangements[diagonal_type] = True
                        break
                    
                    attempts += 1
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_pairs = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {
            'train': train_pairs,
            'test': test_pairs
        }

