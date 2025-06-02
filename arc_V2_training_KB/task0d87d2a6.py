from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects, GridObject
from input_library import retry, random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Task0d87d2a6Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "Each grid contains one or two pairs of {color('line_color')} cells and several {color('block')} rectangular objects, with all remaining cells being empty (0).",
            "Each pair of {color('line_color')} cells must be placed such that the two cells lie at the start and end of a row or a column, excluding the four grid corners — no {color('line_color')} cells may be placed in the corners.",
            "If there are two pairs, ensure that one pair is placed in a row and the other in a column.",
            "All {color('line_color')} cells and {color('block')} rectangular objects must be completely separated from each other by at least one layer of empty (0) cells on all sides.",
            "Each {color('block')} rectangular object must be at least 2 cells wide and 2 cells long.",
            "The positions of both the {color('block')} rectangular objects and the {color('line_color')} cells must vary across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying all pairs of {color('line_color')} cells.",
            "For each identified pair of {color('line_color')} cells, fill the entire row with {color('line_color')} if the pair is placed at the start and end of a row, or fill the entire column if the pair is placed at the start and end of a column.",
            "Apply this rule to all valid pairs of {color('line_color')} cells in the grid.",
            "After drawing all such lines, change the color of any {color('block')} rectangular object to {color('line_color')} if it is touched or intersected by any of the drawn lines."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        height = gridvars['height']
        width = gridvars['width']
        num_pairs = gridvars['num_pairs']
        line_color = taskvars['line_color']
        block_color = taskvars['block']
        
        def will_block_change_color(block_cells, line_positions, grid_shape):
            """Check if a block will change color when lines are drawn"""
            height, width = grid_shape
            
            # Simulate drawing lines
            lines_drawn = set()
            
            # Check for row pairs
            row_groups = {}
            for r, c in line_positions:
                if r not in row_groups:
                    row_groups[r] = []
                row_groups[r].append(c)
            
            for r, cols in row_groups.items():
                if len(cols) == 2 and 0 in cols and (width - 1) in cols:
                    # This row will be filled
                    for c in range(width):
                        lines_drawn.add((r, c))
            
            # Check for column pairs
            col_groups = {}
            for r, c in line_positions:
                if c not in col_groups:
                    col_groups[c] = []
                col_groups[c].append(r)
            
            for c, rows in col_groups.items():
                if len(rows) == 2 and 0 in rows and (height - 1) in rows:
                    # This column will be filled
                    for r in range(height):
                        lines_drawn.add((r, c))
            
            # Check if block touches or intersects any line
            for block_r, block_c in block_cells:
                # Check direct intersection
                if (block_r, block_c) in lines_drawn:
                    return True
                
                # Check touching (8-way adjacency)
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        adj_r, adj_c = block_r + dr, block_c + dc
                        if (adj_r, adj_c) in lines_drawn:
                            return True
            
            return False
        
        def generate_valid_grid():
            grid = np.zeros((height, width), dtype=int)
            forbidden_cells = set()
            line_positions = []
            
            # Place line pairs
            if num_pairs == 1:
                # Single pair - always place in row
                row = random.randint(1, height - 2)  # Avoid corners
                grid[row, 0] = line_color
                grid[row, width - 1] = line_color
                line_positions = [(row, 0), (row, width - 1)]
                
                # Mark forbidden cells (line cells + buffer)
                for lr, lc in line_positions:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = lr + dr, lc + dc
                            if 0 <= nr < height and 0 <= nc < width:
                                forbidden_cells.add((nr, nc))
                                
            elif num_pairs == 2:
                # One row pair, one column pair
                row = random.randint(1, height - 2)
                col = random.randint(1, width - 2)
                
                # Row pair
                grid[row, 0] = line_color
                grid[row, width - 1] = line_color
                
                # Column pair  
                grid[0, col] = line_color
                grid[height - 1, col] = line_color
                
                line_positions = [(row, 0), (row, width - 1), (0, col), (height - 1, col)]
                
                # Mark forbidden cells
                for lr, lc in line_positions:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = lr + dr, lc + dc
                            if 0 <= nr < height and 0 <= nc < width:
                                forbidden_cells.add((nr, nc))
                                
            else:  # num_pairs == 3 (for test case)
                # 2 row pairs and 1 column pair
                rows = random.sample(range(1, height - 2), 2)
                col = random.randint(1, width - 2)
                
                # Row pairs
                for r in rows:
                    grid[r, 0] = line_color
                    grid[r, width - 1] = line_color
                    
                # Column pair
                grid[0, col] = line_color
                grid[height - 1, col] = line_color
                
                line_positions = [(rows[0], 0), (rows[0], width - 1), 
                                (rows[1], 0), (rows[1], width - 1),
                                (0, col), (height - 1, col)]
                
                # Mark forbidden cells
                for lr, lc in line_positions:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = lr + dr, lc + dc
                            if 0 <= nr < height and 0 <= nc < width:
                                forbidden_cells.add((nr, nc))
            
            # Place blocks ensuring we have at least one that changes and one that doesn't
            blocks_placed = []
            blocks_that_change = []
            blocks_that_dont_change = []
            attempts = 0
            
            while (len(blocks_that_change) == 0 or len(blocks_that_dont_change) == 0 or len(blocks_placed) < 2) and attempts < 200:
                attempts += 1
                
                # Random block size (at least 2x2)
                block_height = random.randint(2, min(4, height // 3))
                block_width = random.randint(2, min(4, width // 3))
                
                # Random position
                start_r = random.randint(0, height - block_height)
                start_c = random.randint(0, width - block_width)
                
                # Get block cells
                block_cells = set()
                for r in range(start_r, start_r + block_height):
                    for c in range(start_c, start_c + block_width):
                        block_cells.add((r, c))
                
                # Check buffer zone around the block
                block_and_buffer_cells = set()
                for r in range(max(0, start_r - 1), min(height, start_r + block_height + 1)):
                    for c in range(max(0, start_c - 1), min(width, start_c + block_width + 1)):
                        block_and_buffer_cells.add((r, c))
                
                # Check if placement is valid (no overlap with forbidden cells)
                if not (block_and_buffer_cells & forbidden_cells):
                    # Check if this block will change color
                    will_change = will_block_change_color(block_cells, line_positions, (height, width))
                    
                    # Decide whether to place this block based on what we need
                    should_place = False
                    
                    if len(blocks_that_change) == 0 and will_change:
                        should_place = True
                    elif len(blocks_that_dont_change) == 0 and not will_change:
                        should_place = True
                    elif len(blocks_that_change) > 0 and len(blocks_that_dont_change) > 0 and len(blocks_placed) < 5:
                        # We have both types, can place more randomly
                        should_place = random.random() < 0.7
                    
                    if should_place:
                        # Place the block
                        for r, c in block_cells:
                            grid[r, c] = block_color
                        
                        # Add to appropriate tracking list
                        if will_change:
                            blocks_that_change.append(block_cells)
                        else:
                            blocks_that_dont_change.append(block_cells)
                        
                        blocks_placed.append(block_cells)
                        
                        # Add this block and its buffer to forbidden cells
                        forbidden_cells.update(block_and_buffer_cells)
            
            return grid
        
        # Ensure we have the right conditions
        def is_valid_grid(g):
            # Check that we have blocks
            if np.sum(g == block_color) < 4:
                return False
            
            # Check that we have at least one block that changes and one that doesn't
            line_positions = list(zip(*np.where(g == line_color)))
            block_objects = find_connected_objects(g, background=0, monochromatic=True)
            block_objects = block_objects.with_color(block_color)
            
            changes = 0
            no_changes = 0
            
            for block in block_objects:
                block_cells = {(r, c) for r, c, _ in block.cells}
                if will_block_change_color(block_cells, line_positions, g.shape):
                    changes += 1
                else:
                    no_changes += 1
            
            return changes >= 1 and no_changes >= 1
        
        return retry(generate_valid_grid, is_valid_grid, max_attempts=100)

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output = grid.copy()
        height, width = grid.shape
        line_color = taskvars['line_color']
        block_color = taskvars['block']
        
        # Find all line_color cells
        line_positions = np.where(grid == line_color)
        line_cells = list(zip(line_positions[0], line_positions[1]))
        
        # Group into pairs and draw lines
        lines_drawn = set()
        
        # Check for row pairs (cells at start and end of same row)
        for r in range(height):
            row_cells = [(row, col) for row, col in line_cells if row == r]
            if len(row_cells) == 2:
                # Check if they're at start and end
                cols = [col for _, col in row_cells]
                if 0 in cols and (width - 1) in cols:
                    # Draw line across entire row
                    for c in range(width):
                        output[r, c] = line_color
                        lines_drawn.add((r, c))
        
        # Check for column pairs (cells at start and end of same column)
        for c in range(width):
            col_cells = [(row, col) for row, col in line_cells if col == c]
            if len(col_cells) == 2:
                # Check if they're at start and end
                rows = [row for row, _ in col_cells]
                if 0 in rows and (height - 1) in rows:
                    # Draw line across entire column
                    for r in range(height):
                        output[r, c] = line_color
                        lines_drawn.add((r, c))
        
        # Find block objects and change color if touching any drawn line
        block_objects = find_connected_objects(grid, background=0, monochromatic=True)
        block_objects = block_objects.with_color(block_color)
        
        for block in block_objects:
            touching = False
            
            # Check if any cell of the block is touching any line cell
            for block_r, block_c, _ in block.cells:
                # Check all 8 adjacent cells (including diagonals)
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue  # Skip the cell itself
                        
                        adj_r, adj_c = block_r + dr, block_c + dc
                        if 0 <= adj_r < height and 0 <= adj_c < width:
                            if (adj_r, adj_c) in lines_drawn:
                                touching = True
                                break
                    if touching:
                        break
                if touching:
                    break
            
            # Also check if the block is directly intersected by a line
            for block_r, block_c, _ in block.cells:
                if (block_r, block_c) in lines_drawn:
                    touching = True
                    break
            
            if touching:
                # Change entire block color to line_color
                for block_r, block_c, _ in block.cells:
                    output[block_r, block_c] = line_color
        
        return output

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        colors = list(range(1, 10))
        random.shuffle(colors)
        line_color = colors[0]
        block_color = colors[1]
        
        taskvars = {
            'line_color': line_color,
            'block': block_color
        }
        
        # Generate training examples
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            gridvars = {
                'height': random.randint(8, 20),
                'width': random.randint(8, 20),
                'num_pairs': random.choice([1, 2])
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example with 3 pairs (2 rows, 1 column)
        test_gridvars = {
            'height': random.randint(12, 25),
            'width': random.randint(12, 25),
            'num_pairs': 3
        }
        
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

