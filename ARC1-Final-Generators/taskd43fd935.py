from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskd43fd935Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}.",
            "The grid contains exactly one 2×2 square block filled with color {color('object_color')}. This block occupies the following four adjacent cells: (i,j), (i+1, j), (i, j+1), (i+1, j+1).",
            "In addition to the object block, there are a number of single-colored cells using one or both of two different colors (random_1 and random_2), which are distinct from each other and different from the object color.",
            "There is a placement constraint for single-colored cells aligned with the same rows or columns as the 2×2 object block: At most one single-colored cell may appear on either row i or i+1 in columns after j+1 (to the right of the block). At most one single-colored cell may appear on either row i or i+1 in columns before j (to the left of the block). At most one single-colored cell may appear on either column j or j+1 in rows before i (above the block).At most one single-colored cell may appear on either column j or j+1 in rows after i+1 (below the block). At least one single-colored cell must appear in one or more of the four directional regions defined above.",
            "Additional single-colored cells may be placed anywhere else in the grid without restriction.",
            "No two single-colored cells are horizontally or vertically adjacent (4-connectivity).",
            "All the remaining cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "All single-colored cells and the 2×2 object block are identified.",
            "For each single-colored cell that lies in the same row or column as any part of the 2×2 object block, the entire straight path (either along the row or column) from the single-colored cell up to the nearest edge of the block is filled with the same color as the cell—excluding the object block itself."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'n': random.randint(8, 30),  # Grid size
            'object_color': random.randint(1, 9),  # Color of 2x2 block
        }
        
        # Generate train examples
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        object_color = taskvars['object_color']
        
        def generate_valid_grid():
            # Initialize empty grid
            grid = np.zeros((n, n), dtype=int)
            
            # Place 2x2 block with 1-cell border constraint
            # Block position (i, j) must satisfy: 1 <= i <= n-3, 1 <= j <= n-3
            max_i = n - 3
            max_j = n - 3
            if max_i < 1 or max_j < 1:
                return None  # Grid too small
            
            i = random.randint(1, max_i)
            j = random.randint(1, max_j)
            
            # Place 2x2 block
            grid[i:i+2, j:j+2] = object_color
            
            # Choose random colors for single cells (different from object color)
            available_colors = [c for c in range(1, 10) if c != object_color]
            random_colors = random.sample(available_colors, min(2, len(available_colors)))
            
            # Track placed cells to avoid adjacency
            placed_cells = set()
            for r in range(i, i+2):
                for c in range(j, j+2):
                    placed_cells.add((r, c))
            
            # Define directional regions with proper constraints

            right_positions = [(r, c) for r in [i, i+1] for c in range(j+2, n)]

            left_positions = [(r, c) for r in [i, i+1] for c in range(0, j)]

            above_positions = [(r, c) for r in range(0, i) for c in [j, j+1]]

            below_positions = [(r, c) for r in range(i+2, n) for c in [j, j+1]]
            
            regions = {
                'right': right_positions,
                'left': left_positions,
                'above': above_positions,
                'below': below_positions
            }
            
            # Place at most one cell per region, ensuring at least one region has a cell
            constrained_placements = 0
            regions_to_fill = []
            
            # Decide which regions to fill (at least one)
            for region_name in regions.keys():
                if random.random() < 0.6:  # 60% chance per region
                    regions_to_fill.append(region_name)
            
            # Ensure at least one region is selected
            if not regions_to_fill:
                regions_to_fill = [random.choice(list(regions.keys()))]
            
            # Place exactly one cell in each selected region
            for region_name in regions_to_fill:
                region_positions = regions[region_name]
                if not region_positions:
                    continue
                    
                # Find valid positions (not adjacent to already placed cells)
                valid_positions = [pos for pos in region_positions 
                                 if not self._is_adjacent_to_any(pos, placed_cells)]
                
                if valid_positions:
                    pos = random.choice(valid_positions)
                    color = random.choice(random_colors)
                    grid[pos[0], pos[1]] = color
                    placed_cells.add(pos)
                    constrained_placements += 1
            
            # If no constrained placements were made, force at least one
            if constrained_placements == 0:
                for region_name, region_positions in regions.items():
                    if not region_positions:
                        continue
                    valid_positions = [pos for pos in region_positions 
                                     if not self._is_adjacent_to_any(pos, placed_cells)]
                    if valid_positions:
                        pos = random.choice(valid_positions)
                        color = random.choice(random_colors)
                        grid[pos[0], pos[1]] = color
                        placed_cells.add(pos)
                        break
            
            # Place additional random cells elsewhere (not in the constrained regions)
            num_additional = random.randint(0, 10)
            attempts = 0
            placed_additional = 0
            
            # Get all constrained positions to avoid
            all_constrained_positions = set()
            for region_positions in regions.values():
                all_constrained_positions.update(region_positions)
            
            while placed_additional < num_additional and attempts < 50:
                attempts += 1
                r = random.randint(0, n-1)
                c = random.randint(0, n-1)
                
                if ((r, c) not in placed_cells and 
                    (r, c) not in all_constrained_positions and 
                    not self._is_adjacent_to_any((r, c), placed_cells)):
                    color = random.choice(random_colors)
                    grid[r, c] = color
                    placed_cells.add((r, c))
                    placed_additional += 1
            
            return grid
        
        return retry(generate_valid_grid, lambda x: x is not None)

    def _is_adjacent_to_any(self, pos: Tuple[int, int], placed_cells: set) -> bool:
        """Check if position is adjacent to any placed cell (4-connectivity)"""
        r, c = pos
        neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        return any(neighbor in placed_cells for neighbor in neighbors)

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        object_color = taskvars['object_color']
        output = grid.copy()
        
        # Find the 2x2 block
        block_positions = set()
        block_rows = set()
        block_cols = set()
        
        for r in range(grid.shape[0] - 1):
            for c in range(grid.shape[1] - 1):
                if (grid[r, c] == object_color and 
                    grid[r+1, c] == object_color and 
                    grid[r, c+1] == object_color and 
                    grid[r+1, c+1] == object_color):
                    block_positions = {(r, c), (r+1, c), (r, c+1), (r+1, c+1)}
                    block_rows = {r, r+1}
                    block_cols = {c, c+1}
                    break
        
        if not block_positions:
            return output
        
        # Find all single-colored cells (non-zero, non-object-color)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                cell_color = grid[r, c]
                if cell_color != 0 and cell_color != object_color:
                    # Check if this cell is aligned with the block
                    if r in block_rows:
                        # Same row - draw horizontal line
                        if c < min(block_cols):

                            for fill_c in range(c + 1, min(block_cols)):
                                if output[r, fill_c] == 0:  
                                    output[r, fill_c] = cell_color
                        elif c > max(block_cols):

                            for fill_c in range(max(block_cols) + 1, c):
                                if output[r, fill_c] == 0:  
                                    output[r, fill_c] = cell_color
                    
                    if c in block_cols:
                        # Same column - draw vertical line
                        if r < min(block_rows):

                            for fill_r in range(r + 1, min(block_rows)):
                                if output[fill_r, c] == 0:  
                                    output[fill_r, c] = cell_color
                        elif r > max(block_rows):

                            for fill_r in range(max(block_rows) + 1, r):
                                if output[fill_r, c] == 0:  
                                    output[fill_r, c] = cell_color
        
        return output
