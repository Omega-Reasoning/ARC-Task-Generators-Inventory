from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import GridObject, GridObjects, find_connected_objects
from input_library import create_object, retry, Contiguity
import numpy as np
import random

class Task150deff5Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "Each input grid contains exactly one {color('object_color')} object made of 8-way connected cells, with all other cells being empty (0).",
            "To construct the {color('object_color')} object; create 2 or 3 blocks of size 2×2 using the {color('object_color')} color.",
            "Then, 4-way connect each 2×2 block to a 3×1(vertical or horizontal) block made of the same color.",
            "This results in 2 or 3 pairs where each pair has a 2×2 and 3×1 block.",
            "Finally, 8-way connect all the pairs together to form a single {color('object_color')} object, but ensure no extra cells are added.",
            "Ensure that the {color('object_color')} object contains a total number of cells that is a multiple of 7."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the large {color('object_color')} object.",
            "The {color('object_color')} object is formed by creating 2 or 3 blocks of size 2×2 using the {color('object_color')} color, then 4-way connecting each 2×2 block to a 3×1 block (either vertical or horizontal) made of the same color. This results in 2 or 3 pairs, where each pair consists of one 2×2 block and one 3×1 block. Finally, all the pairs are 8-way connected together to form a single {color('object_color')} object, which is then placed in the grid.",
            "The important part is to determine which parts of the {color('object_color')} object were originally 2×2 blocks and which were 3×1 blocks.",
            "After identifying and reconstructing the structure, there must be exactly 2 or 3 2×2 blocks, each with its own connected 3×1 block.",
            "At the end, no other cells of the {color('object_color')} object should remain unaccounted for outside of these pairs.",
            "At this point, each identified 2×2 block and 3×1 block is to be colored {color('fill_color1')} and {color('fill_color2')} respectively.",
            
        ]
        
        taskvars_definitions = {}
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        grid_size = gridvars.get('grid_size', random.choice([10, 12, 14, 16]))
        height = width = grid_size
        object_color = taskvars['object_color']
        num_pairs = gridvars.get('num_pairs', random.choice([2, 3]))
        
        def generate_valid_object():
            grid = np.zeros((height, width), dtype=int)
            
            # First, create individual pairs (2x2 + 3x1 blocks)
            pairs = []
            
            for i in range(num_pairs):
                # Try to create a pair
                pair_created = False
                for attempt in range(100):
                    # Create a temporary grid for this pair
                    pair_grid = np.zeros((7, 7), dtype=int)
                    
                    # Place 2x2 block in center area
                    r_2x2 = random.randint(2, 3)
                    c_2x2 = random.randint(2, 3)
                    pair_grid[r_2x2:r_2x2+2, c_2x2:c_2x2+2] = 1
                    
                    # Choose direction for 3x1 block (must be 4-way adjacent)
                    directions = [
                        ('up', r_2x2-1, c_2x2, 1, 3),      # horizontal above
                        ('up', r_2x2-1, c_2x2+1, 1, 3),    # horizontal above (right aligned)
                        ('down', r_2x2+2, c_2x2, 1, 3),    # horizontal below
                        ('down', r_2x2+2, c_2x2+1, 1, 3),  # horizontal below (right aligned)
                        ('left', r_2x2, c_2x2-1, 3, 1),    # vertical left
                        ('left', r_2x2+1, c_2x2-1, 3, 1),  # vertical left (bottom aligned)
                        ('right', r_2x2, c_2x2+2, 3, 1),   # vertical right
                        ('right', r_2x2+1, c_2x2+2, 3, 1)  # vertical right (bottom aligned)
                    ]
                    
                    valid_directions = []
                    for dir_name, r, c, h, w in directions:
                        if 0 <= r < 7 and 0 <= c < 7 and r+h <= 7 and c+w <= 7:
                            valid_directions.append((dir_name, r, c, h, w))
                    
                    if not valid_directions:
                        continue
                        
                    dir_name, r_3x1, c_3x1, h_3x1, w_3x1 = random.choice(valid_directions)
                    pair_grid[r_3x1:r_3x1+h_3x1, c_3x1:c_3x1+w_3x1] = 1
                    
                    # Extract the pair as coordinates
                    pair_coords = [(r, c) for r in range(7) for c in range(7) if pair_grid[r, c] == 1]
                    
                    # Store pair info
                    pairs.append({
                        'coords': pair_coords,
                        'grid': pair_grid,
                        '2x2': [(r_2x2 + dr, c_2x2 + dc) for dr in range(2) for dc in range(2)],
                        '3x1': [(r_3x1 + dr, c_3x1 + dc) for dr in range(h_3x1) for dc in range(w_3x1)]
                    })
                    pair_created = True
                    break
                
                if not pair_created:
                    return None
            
            # Now place pairs on the main grid, ensuring 8-way connectivity
            placed_pairs = []
            
            # Place first pair randomly
            first_pair = pairs[0]
            r_offset = random.randint(1, height - 8)
            c_offset = random.randint(1, width - 8)
            
            for r, c in first_pair['coords']:
                grid[r + r_offset, c + c_offset] = object_color
            
            placed_pairs.append({
                'coords': [(r + r_offset, c + c_offset) for r, c in first_pair['coords']],
                '2x2': [(r + r_offset, c + c_offset) for r, c in first_pair['2x2']],
                '3x1': [(r + r_offset, c + c_offset) for r, c in first_pair['3x1']]
            })
            
            # Place remaining pairs ensuring 8-way connectivity
            for i in range(1, num_pairs):
                pair = pairs[i]
                placed = False
                
                # Try different positions
                for _ in range(200):
                    r_offset = random.randint(0, height - 7)
                    c_offset = random.randint(0, width - 7)
                    
                    # Check if this position would overlap
                    would_overlap = False
                    new_coords = [(r + r_offset, c + c_offset) for r, c in pair['coords']]
                    
                    for nr, nc in new_coords:
                        if nr >= height or nc >= width or grid[nr, nc] != 0:
                            would_overlap = True
                            break
                    
                    if would_overlap:
                        continue
                    
                    # Check if 8-way connected to existing structure
                    is_connected = False
                    for nr, nc in new_coords:
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                ar, ac = nr + dr, nc + dc
                                if 0 <= ar < height and 0 <= ac < width and grid[ar, ac] == object_color:
                                    is_connected = True
                                    break
                            if is_connected:
                                break
                        if is_connected:
                            break
                    
                    if is_connected:
                        # Place this pair
                        for r, c in pair['coords']:
                            grid[r + r_offset, c + c_offset] = object_color
                        
                        placed_pairs.append({
                            'coords': [(r + r_offset, c + c_offset) for r, c in pair['coords']],
                            '2x2': [(r + r_offset, c + c_offset) for r, c in pair['2x2']],
                            '3x1': [(r + r_offset, c + c_offset) for r, c in pair['3x1']]
                        })
                        placed = True
                        break
                
                if not placed:
                    return None
            
            # Verify it's a single connected component
            objects = find_connected_objects(grid, diagonal_connectivity=True)
            if len(objects) != 1:
                return None
            
            # Verify total cells (should be 7 * num_pairs)
            total_cells = np.sum(grid == object_color)
            if total_cells != 7 * num_pairs:
                return None
            
            # Store pair information for later use
            gridvars['placed_pairs'] = placed_pairs
            
            return grid
        
        # Generate valid object
        result = retry(generate_valid_object, lambda x: x is not None, max_attempts=200)
        if result is None:
            raise ValueError("Failed to generate valid object")
            
        return result

    def transform_input(self, grid, taskvars):
        object_color = taskvars['object_color']
        fill_color1 = taskvars['fill_color1']
        fill_color2 = taskvars['fill_color2']
        
        output = grid.copy()
        
        # Count total object cells
        total_cells = np.sum(grid == object_color)
        expected_pairs = total_cells // 7
        
        if total_cells % 7 != 0 or expected_pairs not in [2, 3]:
            return grid.copy()
        
        # Find all object cells
        object_cells = set()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == object_color:
                    object_cells.add((r, c))
        
        # Find all possible 2x2 blocks
        all_2x2_blocks = []
        for r in range(grid.shape[0] - 1):
            for c in range(grid.shape[1] - 1):
                block = {(r, c), (r, c+1), (r+1, c), (r+1, c+1)}
                if block.issubset(object_cells):
                    all_2x2_blocks.append(block)
        
        # Try different combinations of 2x2 blocks
        from itertools import combinations
        blocks_2x2 = None
        blocks_3x1 = None
        
        for combo in combinations(all_2x2_blocks, expected_pairs):
            # Check if blocks don't overlap
            all_2x2_cells = set()
            valid_combo = True
            for block in combo:
                if block.intersection(all_2x2_cells):
                    valid_combo = False
                    break
                all_2x2_cells.update(block)
            
            if not valid_combo:
                continue
            
            # Check remaining cells can form 3x1 blocks
            remaining = object_cells - all_2x2_cells
            
            # Find all possible 3x1 blocks in remaining cells
            all_3x1_blocks = []
            
            # Horizontal 1x3
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1] - 2):
                    block = {(r, c), (r, c+1), (r, c+2)}
                    if block.issubset(remaining):
                        all_3x1_blocks.append(block)
            
            # Vertical 3x1
            for r in range(grid.shape[0] - 2):
                for c in range(grid.shape[1]):
                    block = {(r, c), (r+1, c), (r+2, c)}
                    if block.issubset(remaining):
                        all_3x1_blocks.append(block)
            
            # Try to find exactly expected_pairs 3x1 blocks
            for combo_3x1 in combinations(all_3x1_blocks, expected_pairs):
                all_3x1_cells = set()
                valid_3x1_combo = True
                for block in combo_3x1:
                    if block.intersection(all_3x1_cells):
                        valid_3x1_combo = False
                        break
                    all_3x1_cells.update(block)
                
                if not valid_3x1_combo:
                    continue
                
                # Check if we've covered all cells
                if all_2x2_cells.union(all_3x1_cells) == object_cells:
                    blocks_2x2 = combo
                    blocks_3x1 = combo_3x1
                    break
            
            if blocks_2x2 is not None:
                break
        
        if blocks_2x2 is None or blocks_3x1 is None:
            return grid.copy()
        
        # Verify we have the right number of blocks
        if len(blocks_2x2) != expected_pairs or len(blocks_3x1) != expected_pairs:
            return grid.copy()
        
        # Color the blocks
        for block in blocks_2x2:
            for r, c in block:
                output[r, c] = fill_color1
                
        for block in blocks_3x1:
            for r, c in block:
                output[r, c] = fill_color2
        
        return output

    def create_grids(self):
        # Choose colors ensuring they're all different
        colors = list(range(1, 10))
        random.shuffle(colors)
        object_color = colors[0]
        fill_color1 = colors[1]
        fill_color2 = colors[2]
        
        taskvars = {
            'object_color': object_color,
            'fill_color1': fill_color1,
            'fill_color2': fill_color2
        }
        
        # Create training examples
        train_grids = []
        
        # Ensure we show both 2 and 3 pairs in training
        for i in range(3):
            gridvars = {
                'grid_size': random.choice(range(10, 31)),
                'num_pairs': 2 if i == 0 else (3 if i == 1 else random.choice([2, 3]))
            }
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_grids.append({'input': input_grid, 'output': output_grid})
        
        # Create test example
        test_gridvars = {
            'grid_size': random.choice(range(10, 31)),
            'num_pairs': random.choice([2, 3])
        }
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_grids = [{'input': test_input, 'output': test_output}]
        
        train_test_data = {
            'train': train_grids,
            'test': test_grids
        }
        
        return taskvars, train_test_data