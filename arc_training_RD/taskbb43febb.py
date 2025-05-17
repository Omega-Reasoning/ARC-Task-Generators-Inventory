from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, List, Tuple, Set, Optional
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, GridObject

class FillBlocksInteriorTaskGenerator(ARCTaskGenerator):
    def __init__(self):
        self.input_reasoning_chain = [
            "The input grid consists of square grids of varying sizes.",
            "The grid contains square and rectangle blocks filled with {color(\"block_color\")} color.",
            "The square blocks are typically ≥ 3x3, and the rectangle blocks are ≥ 4x3 or 3x4 in size."
        ]
        
        self.transformation_reasoning_chain = [
            "The output grid is a copy of the input grid.",
            "All blocks (square and rectangular) are preserved in position and size.",
            "In the output, each block boundary remains {color(\"block_color\")} color, while the inner region is filled with {color(\"inner_color\")} color."
        ]
        
        taskvars_definitions = {
            "block_color": "The color of the blocks in the input grid",
            "inner_color": "The color used to fill the interior of blocks in the output grid"
        }
        
        super().__init__(self.input_reasoning_chain, self.transformation_reasoning_chain)
    
    def create_input(self, grid_size: Tuple[int, int], n_blocks: int, block_color: int) -> np.ndarray:
        """Create an input grid with blocks of the specified color"""
        rows, cols = grid_size
        grid = np.zeros((rows, cols), dtype=int)
        
        # Track occupied cells to avoid overlapping blocks
        occupied_cells = set()
        
        for _ in range(n_blocks):
            # Decide between square and rectangle
            is_square = random.choice([True, False])
            
            if is_square:
                # Create a square block (≥ 3x3)
                size = random.randint(3, min(6, rows-2, cols-2))
                block_width, block_height = size, size
            else:
                # Create a rectangle block (≥ 4x3 or ≥ 3x4)
                if random.choice([True, False]):
                    block_width = random.randint(4, min(8, cols-2))
                    block_height = random.randint(3, min(5, rows-2))
                else:
                    block_width = random.randint(3, min(5, cols-2))
                    block_height = random.randint(4, min(8, rows-2))
            
            # Try to find a valid position for the block
            max_attempts = 50
            for attempt in range(max_attempts):
                start_row = random.randint(1, rows - block_height - 1)
                start_col = random.randint(1, cols - block_width - 1)
                
                # Check if this placement overlaps with existing blocks
                block_cells = {(r, c) for r in range(start_row, start_row + block_height) 
                              for c in range(start_col, start_col + block_width)}
                
                # Need some padding around blocks to ensure they don't touch
                padding_cells = {(r, c) for r in range(start_row-1, start_row + block_height+1) 
                               for c in range(start_col-1, start_col + block_width+1)}
                
                if not padding_cells.intersection(occupied_cells):
                    # Valid placement - add block to grid
                    for r, c in block_cells:
                        grid[r, c] = block_color
                    occupied_cells.update(block_cells)
                    break
            
            # If all attempts failed, just continue (we'll have fewer blocks)
        
        return grid
    
    def transform_input(self, grid: np.ndarray, vars: Dict[str, int]) -> np.ndarray:
        """Transform the input grid by filling the interior of blocks"""
        block_color = vars["block_color"]
        inner_color = vars["inner_color"]
        
        # Create a copy of the input grid
        output_grid = grid.copy()
        
        # Find all blocks (connected components of block_color)
        blocks = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)
        
        # Process each block
        for block in blocks:
            # Verify this is a block of the right color
            if block_color not in block.colors:
                continue
                
            # Find interior cells (those that don't touch background)
            interior_cells = set()
            boundary_cells = set()
            
            for r, c, _ in block:
                is_boundary = False
                # Check 4-connected neighbors
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    # If neighbor is outside grid or is background, this is a boundary cell
                    if (nr < 0 or nr >= grid.shape[0] or 
                        nc < 0 or nc >= grid.shape[1] or
                        grid[nr, nc] == 0):
                        is_boundary = True
                        break
                
                if is_boundary:
                    boundary_cells.add((r, c))
                else:
                    interior_cells.add((r, c))
            
            # Color interior cells with inner_color
            for r, c in interior_cells:
                output_grid[r, c] = inner_color
        
        return output_grid
    
    def create_grids(self) -> Tuple[Dict[str, int], TrainTestData]:
        # Choose colors for blocks and inner filling
        valid_colors = list(range(1, 10))
        block_color = random.choice(valid_colors)
        valid_colors.remove(block_color)
        inner_color = random.choice(valid_colors)
        
        vars = {
            "block_color": block_color,
            "inner_color": inner_color
        }
        
        # Create training examples
        num_train_examples = random.randint(3, 5)
        train_data = []
        
        for _ in range(num_train_examples):
            # Randomize grid size and number of blocks
            grid_size = (random.randint(7, 15), random.randint(7, 15))
            n_blocks = random.randint(1, 5)
            
            input_grid = self.create_input(grid_size, n_blocks, block_color)
            output_grid = self.transform_input(input_grid, vars)
            train_data.append(GridPair(input=input_grid, output=output_grid))
        # Create test example - make it a bit more complex
        test_grid_size = (random.randint(10, 20), random.randint(10, 20))
        test_n_blocks = random.randint(2, 6)
        test_input = self.create_input(test_grid_size, test_n_blocks, block_color)
        test_output = self.transform_input(test_input, vars)
        
        test_data = [GridPair(input=test_input, output=test_output)]
        
        return vars, TrainTestData(train = train_data, test = test_data)

