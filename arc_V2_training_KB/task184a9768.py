from arc_task_generator import ARCTaskGenerator, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple
from transformation_library import find_connected_objects

class Task184a9768Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "They contain multiple rectangle-like objects, with the remaining cells being empty (0).",
            "They are constructed by first creating 2 or 3 rectangular blocks, then removing several small rectangular sections from inside the created blocks (except for boundary cells).",
            "Ensure that no two removed sections are of the same size.",
            "Once the small parts are removed, place them outside the grid and change their colors so they differ from the original block colors.",
            "All blocks, including the moved pieces, should have different colors.",
            "Next, add several 1x1 blocks of {color('block_color')} color.",
            "Ensure all objects are completely separated from each other.",
            "The shapes and sizes of objects vary across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying the large rectangular blocks with empty rectangular regions, as well as the small completely colored rectangular blocks located outside the large blocks.",
            "Move all small colored blocks so that the empty regions inside the large blocks are filled by placing each same-sized colored block into a matching same-sized empty region.",
            "Ensure that {color('block_color')} cells are never used to replace any missing 1x1 block."
            "Once all empty regions have been filled with colored blocks, remove all {color('block_color')} cells."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables (only block_color is consistent across examples)
        block_color = random.randint(1, 9)
        
        taskvars = {
            'block_color': block_color
        }
        
        # Create 3-5 train examples and 1 test example
        num_train_examples = random.randint(3, 5)
        
        train_examples = []
        
        for _ in range(num_train_examples):
            # Create grid variables for this specific example (different grid size each time)
            grid_size = random.randint(15, 25)  # Different size for each example
            gridvars = {
                'grid_size': grid_size,
                'num_big_blocks': random.randint(2, 3),
                'num_small_pieces': random.randint(3, 6),
                'num_block_color_pieces': random.randint(2, 4)
            }
            
            # Create input grid and transform it
            input_grid = self.create_input(taskvars, gridvars, 0)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example (also with different grid size)
        test_grid_size = random.randint(15, 25)
        test_gridvars = {
            'grid_size': test_grid_size,
            'num_big_blocks': random.randint(2, 3),
            'num_small_pieces': random.randint(3, 6),
            'num_block_color_pieces': random.randint(2, 4)
        }
        
        test_input = self.create_input(taskvars, test_gridvars, 0)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any], recursion_depth: int = 0) -> np.ndarray:
        # Prevent infinite recursion
        if recursion_depth > 10:
            # Fallback: reduce requirements if we can't generate after many attempts
            gridvars = gridvars.copy()
            gridvars['num_big_blocks'] = max(1, gridvars['num_big_blocks'] - 1)
            gridvars['num_block_color_pieces'] = max(1, gridvars['num_block_color_pieces'] - 1)
        grid_size = gridvars['grid_size']
        block_color = taskvars['block_color']
        num_big_blocks = gridvars['num_big_blocks']
        num_block_color_pieces = gridvars['num_block_color_pieces']
        
        # Initialize empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Available colors (excluding 0 and block_color)
        available_colors = [c for c in range(1, 10) if c != block_color]
        random.shuffle(available_colors)
        
        # Track holes and their corresponding pieces for later placement
        holes_and_pieces = []
        used_hole_sizes = set()
        
        # Create big rectangular blocks with holes
        big_block_color_idx = 0
        for _ in range(num_big_blocks):
            # Ensure we have enough colors
            if big_block_color_idx >= len(available_colors):
                break
                
            big_block_color = available_colors[big_block_color_idx]
            big_block_color_idx += 1
            
            # Try to place the big block
            attempts = 100
            placed = False
            
            for _ in range(attempts):
                # Random size for the big block (ensure it's big enough for holes)
                block_height = random.randint(5, 9)
                block_width = random.randint(5, 9)
                
                # Random position
                if grid_size - block_height <= 0 or grid_size - block_width <= 0:
                    continue
                    
                block_row = random.randint(0, grid_size - block_height)
                block_col = random.randint(0, grid_size - block_width)
                
                # Check if area is clear (including 1-cell border)
                check_row_start = max(0, block_row - 1)
                check_row_end = min(grid_size, block_row + block_height + 1)
                check_col_start = max(0, block_col - 1)
                check_col_end = min(grid_size, block_col + block_width + 1)
                
                if np.all(grid[check_row_start:check_row_end, check_col_start:check_col_end] == 0):
                    # Place the block
                    grid[block_row:block_row+block_height, block_col:block_col+block_width] = big_block_color
                    
                    # MUST create holes inside the block (all big blocks must have holes)
                    num_holes = random.randint(1, 3)
                    holes_created = 0
                    
                    for _ in range(num_holes):
                        hole_attempts = 50
                        hole_placed = False
                        for _ in range(hole_attempts):
                            # Generate unique hole size
                            hole_height = random.randint(1, min(3, block_height - 2))
                            hole_width = random.randint(1, min(3, block_width - 2))
                            hole_size = (hole_height, hole_width)
                            
                            if hole_size in used_hole_sizes:
                                continue
                                
                            # Position inside the block (not on boundary)
                            if block_height - hole_height - 2 <= 0 or block_width - hole_width - 2 <= 0:
                                continue
                                
                            hole_row = random.randint(block_row + 1, block_row + block_height - hole_height - 1)
                            hole_col = random.randint(block_col + 1, block_col + block_width - hole_width - 1)
                            
                            # Check if hole area is currently the big block color
                            hole_area = grid[hole_row:hole_row+hole_height, hole_col:hole_col+hole_width]
                            if np.all(hole_area == big_block_color):
                                # Check if this hole would connect to another hole (avoid connected holes)
                                # Check surrounding area to ensure separation
                                check_row_start = max(block_row, hole_row - 1)
                                check_row_end = min(block_row + block_height, hole_row + hole_height + 1)
                                check_col_start = max(block_col, hole_col - 1)
                                check_col_end = min(block_col + block_width, hole_col + hole_width + 1)
                                
                                surrounding_area = grid[check_row_start:check_row_end, check_col_start:check_col_end]
                                # Count zeros in surrounding area (should only be the hole we're about to create)
                                if np.sum(surrounding_area == 0) == 0:
                                    # Create the hole
                                    grid[hole_row:hole_row+hole_height, hole_col:hole_col+hole_width] = 0
                                    
                                    # Store hole info for piece creation
                                    holes_and_pieces.append({
                                        'size': hole_size,
                                        'original_color': big_block_color
                                    })
                                    used_hole_sizes.add(hole_size)
                                    hole_placed = True
                                    holes_created += 1
                                    break
                        
                        if not hole_placed:
                            break  # If we can't place this hole, stop trying more holes for this block
                    
                    # Only consider the block successfully placed if it has at least one hole
                    if holes_created > 0:
                        placed = True
                        break
                    else:
                        # Remove the block if no holes could be created
                        grid[block_row:block_row+block_height, block_col:block_col+block_width] = 0
            
            if not placed:
                # If we can't place a big block, continue with fewer blocks
                continue
        
        # CRITICAL: Place ALL pieces corresponding to holes - this is mandatory
        # If we can't place a piece, we need to regenerate the entire grid
        
        piece_color_idx = big_block_color_idx
        successfully_placed_pieces = []
        
        for piece_info in holes_and_pieces:
            if piece_color_idx >= len(available_colors):
                # Not enough colors - regenerate grid
                return self.create_input(taskvars, gridvars, recursion_depth + 1)
                
            piece_color = available_colors[piece_color_idx]
            piece_color_idx += 1
            piece_info['piece_color'] = piece_color
            
            hole_height, hole_width = piece_info['size']
            
            # Try to place the piece outside big blocks
            attempts = 200  # More attempts
            piece_placed = False
            for _ in range(attempts):
                piece_row = random.randint(0, grid_size - hole_height)
                piece_col = random.randint(0, grid_size - hole_width)
                
                # Check if area and border is clear
                check_row_start = max(0, piece_row - 1)
                check_row_end = min(grid_size, piece_row + hole_height + 1)
                check_col_start = max(0, piece_col - 1)
                check_col_end = min(grid_size, piece_col + hole_width + 1)
                
                if np.all(grid[check_row_start:check_row_end, check_col_start:check_col_end] == 0):
                    # Place the piece (solid rectangle with no holes)
                    grid[piece_row:piece_row+hole_height, piece_col:piece_col+hole_width] = piece_color
                    successfully_placed_pieces.append(piece_info)
                    piece_placed = True
                    break
            
            if not piece_placed:
                # CRITICAL: If we can't place this piece, regenerate the entire grid
                # This ensures every hole has a corresponding piece
                return self.create_input(taskvars, gridvars, recursion_depth + 1)
        
        # Add 1x1 blocks of block_color (but not too many to avoid interfering)
        for _ in range(min(num_block_color_pieces, 3)):
            attempts = 50
            for _ in range(attempts):
                row = random.randint(0, grid_size - 1)
                col = random.randint(0, grid_size - 1)
                
                # Check if cell and its neighbors are clear
                check_row_start = max(0, row - 1)
                check_row_end = min(grid_size, row + 2)
                check_col_start = max(0, col - 1)
                check_col_end = min(grid_size, col + 2)
                
                if np.all(grid[check_row_start:check_row_end, check_col_start:check_col_end] == 0):
                    grid[row, col] = block_color
                    break
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        block_color = taskvars['block_color']
        
        # Create a copy of the input grid
        output_grid = grid.copy()
        
        # Find all connected objects
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        
        # Separate objects into big blocks (with holes) and small pieces
        big_blocks = []
        small_pieces = []
        
        for obj in objects:
            if obj.has_color(block_color):
                continue  # Skip block_color objects for now
                
            obj_array = obj.to_array()
            
            # Check if object has holes (contains 0s inside)
            has_holes = np.any(obj_array == 0)
            
            if has_holes and obj.size > 10:  # Likely a big block
                big_blocks.append(obj)
            elif not has_holes:  # Solid small piece
                small_pieces.append(obj)
        
        # For each big block, find its holes and try to fill them with matching pieces
        for big_block in big_blocks:
            # Get the bounding box
            bbox = big_block.bounding_box
            row_start, row_end = bbox[0].start, bbox[0].stop
            col_start, col_end = bbox[1].start, bbox[1].stop
            
            # Extract the region
            region = grid[row_start:row_end, col_start:col_end]
            
            # Find holes (connected 0 regions within the big block area)
            hole_mask = (region == 0)
            if not np.any(hole_mask):
                continue
                
            # Find connected hole regions
            from scipy.ndimage import label
            structure = np.array([[0,1,0],[1,1,1],[0,1,0]])  # 4-way connectivity
            labeled_holes, num_holes = label(hole_mask, structure=structure)
            
            for hole_id in range(1, num_holes + 1):
                hole_coords = np.where(labeled_holes == hole_id)
                if len(hole_coords[0]) == 0:
                    continue
                
                # Get the exact rectangular bounds of the hole
                hole_min_row = hole_coords[0].min()
                hole_max_row = hole_coords[0].max()
                hole_min_col = hole_coords[1].min() 
                hole_max_col = hole_coords[1].max()
                
                hole_height = hole_max_row - hole_min_row + 1
                hole_width = hole_max_col - hole_min_col + 1
                hole_size = (hole_height, hole_width)
                
                # Verify the hole is actually rectangular by checking all cells in the bounding box are empty
                hole_region = region[hole_min_row:hole_max_row+1, hole_min_col:hole_max_col+1]
                if not np.all(hole_region == 0):
                    continue  # Skip non-rectangular holes
                
                # Find a matching small piece with exact dimensions
                matching_piece = None
                for piece in small_pieces:
                    piece_height = piece.height
                    piece_width = piece.width
                    
                    # Exact dimension matching (not just area)
                    if (piece_height, piece_width) == hole_size:
                        matching_piece = piece
                        break
                
                if matching_piece:
                    # Remove the piece from its current location
                    matching_piece.cut(output_grid)
                    
                    # Place it in the hole using the correct coordinates
                    hole_row_offset = hole_min_row
                    hole_col_offset = hole_min_col
                    
                    # Get piece array and place it directly in the hole location
                    piece_array = matching_piece.to_array()
                    for r in range(piece_array.shape[0]):
                        for c in range(piece_array.shape[1]):
                            if piece_array[r, c] != 0:
                                new_r = row_start + hole_row_offset + r
                                new_c = col_start + hole_col_offset + c
                                output_grid[new_r, new_c] = piece_array[r, c]
                    
                    # Remove this piece from the list so it can't be used again
                    small_pieces.remove(matching_piece)
        
        # Remove all block_color cells
        output_grid[output_grid == block_color] = 0
        
        return output_grid

