from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from input_library import create_object, random_cell_coloring, Contiguity
from transformation_library import find_connected_objects, GridObject, GridObjects, BorderBehavior

class Task3e6067c3Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each input grid has several {color('object_color')} square objects and multi-colored (1-9) cells, with all remaining cells being {color('background_color')}.",
            "The {color('object_color')} square objects have their central cell or cells differently colored: if the square is 4x4, then the interior 2x2 block is differently colored; otherwise, only the center cell is.", 
            "This different interior color should not be {color('background_color')} and must be unique for each block.",
            "Each square block can be 3x3, 4x4, or 5x5 in size, and all square blocks within one grid must have the same size.",
            "The first block starts from position (1,1), and all blocks must be vertically and horizontally aligned.",
            "There should be at least one completely filled row and column of {color('background_color')} between two consecutive blocks.",
            "The number of completely filled rows and columns of {color('background_color')} color between two consecutive blocks can vary within a single grid, but the vertical and horizontal alignment must remain preserved.",
            "No {color('object_color')} square should occupy the last three rows of the grid.",
            "The second-last row contains single-colored cells, using colors that appear in the center of the square blocks, each placed with one cell of {color('background_color')} color between them.",
            "The sequence of colors in the second-last row begins with the interior color of the first square block (starting at 1,1), followed by the interior color of the next block either to its right or directly below.",
            "The sequence of single-colored cells in the second last row should be such that, starting from the first block, one can follow a continuous path through the blocks—moving only to adjacent blocks, covering all blocks or leaving at most one out, and the path must pass only through blocks.",
            "The following cases help illustrate how the sequence of interior colors in the second last row should be constructed based on a valid continuous path through adjacent blocks.",
            "Suppose there is a block at the top-left with interior color a, another to its right with interior color b, and one directly below the second with interior color c. The color sequence in the second last row should be a b c, not a c b",
            "Suppose there is a block at the top-left with interior color a, another to its right with color b, a third to the right of the second with color c, and one directly below the third with color d. The correct color sequence should be a b c d.",
            "Suppose there is a block at the top-left with color a, one to its right with color b, a third to the right with color c, one directly below the first with color d, one below the second with color e, and one below the third with color f. Valid sequences for the second last row include a b c f e d, a d e b c f, a b e f c, or a d e f c b—any sequence that follows a valid continuous path through adjacent blocks."
        ]
        
        transformation_reasoning_chain = [
            "Output grids are constructed by copying the input grids and identifying all square objects and their differently colored interior.",
            "Once identified, follow the color sequence provided in the second-to-last row and create a path by recoloring certain background cells, starting from the first square block and continuing to the block with the interior color that matches the final color in the sequence.",
            "The path is created using the colors that appear in sequence. For example, if the sequence in the second-to-last row is color a, b, c, the path would start from the block with color a as the interior color and extend to the block with color b as the interior color. This is done by recoloring cells, which are vertically or horizontally aligned with the cells that have color a as their interior color.",
            "The path never overlaps any cells within the square objects and stops when it reaches the square block with the correct interior color, as indicated by the final color in the sequence given in second-to-last row."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
        
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables for training examples
        train_taskvars = {
            'object_color': random.randint(1, 9),
            'background_color': random.randint(1, 9)
        }
        
        # Ensure object_color and background_color are different
        while train_taskvars['object_color'] == train_taskvars['background_color']:
            train_taskvars['background_color'] = random.randint(1, 9)
        
        # Create separate variables for test grid with different colors
        test_taskvars = {
            'object_color': random.randint(1, 9),
            'background_color': random.randint(1, 9)
        }
        
        # Ensure test colors are different from each other
        while test_taskvars['object_color'] == test_taskvars['background_color']:
            test_taskvars['background_color'] = random.randint(1, 9)
        
        # Ensure test background color differs from training background color
        while test_taskvars['background_color'] == train_taskvars['background_color']:
            test_taskvars['background_color'] = random.randint(1, 9)
        
        # Ensure test object color differs from training object color
        while test_taskvars['object_color'] == train_taskvars['object_color']:
            test_taskvars['object_color'] = random.randint(1, 9)
            # Re-check that it's different from background
            if test_taskvars['object_color'] == test_taskvars['background_color']:
                test_taskvars['object_color'] = random.randint(1, 9)
        
        # Create 4 training examples
        train_examples = []
        
        # Create two examples with 6 blocks (3 in each row)
        for _ in range(2):
            gridvars = {'block_style': 'six_blocks'}
            input_grid = self.create_input(train_taskvars, gridvars)
            output_grid = self.transform_input(input_grid, train_taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create two examples with 7 or 8 blocks (different patterns)
        for _ in range(2):
            gridvars = {'block_style': random.choice(['style_one', 'style_two', 'style_three'])}
            input_grid = self.create_input(train_taskvars, gridvars)
            output_grid = self.transform_input(input_grid, train_taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create test example (random style)
        test_gridvars = {'block_style': random.choice(['six_blocks', 'style_one', 'style_two', 'style_three'])}
        test_input = self.create_input(test_taskvars, test_gridvars)
        test_output = self.transform_input(test_input, test_taskvars)
        
        return train_taskvars, {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """Create an input grid with square objects according to instructions."""
        object_color = taskvars['object_color']
        background_color = taskvars['background_color']
        
        # Grid size between 20 and 30 rows/columns
        height = random.randint(26, 30)
        width = random.randint(26, 30)
        
        # Initialize grid with background color
        grid = np.full((height, width), background_color, dtype=int)
        
        # Randomly choose block size (3x3, 4x4, or 5x5)
        block_size = random.choice([3, 4, 5])
        
        # Get block style from gridvars
        block_style = gridvars.get('block_style', 'six_blocks')
        
        # Create block positions and interior colors based on the style
        block_positions = []
        if block_style == 'six_blocks':
            # Create 6 blocks: 3 in top row, 3 in bottom row
            block_positions = self._generate_block_positions_six_blocks(height, width, block_size)
        elif block_style in ['style_one', 'style_two', 'style_three']:
            # Create 7 or 8 blocks with specific patterns
            block_positions = self._generate_block_positions_style(block_style, height, width, block_size)
        
        # Get unique interior colors for each block (different from object_color and background_color)
        interior_colors = self._get_unique_colors(len(block_positions), [object_color, background_color])
        
        # Place blocks and store block info for the path sequence
        blocks_info = []
        for i, (pos_row, pos_col) in enumerate(block_positions):
            interior_color = interior_colors[i]
            self._place_square_block(grid, pos_row, pos_col, block_size, object_color, interior_color)
            blocks_info.append({
                'position': (pos_row, pos_col),
                'interior_color': interior_color,
                'size': block_size
            })
        
        # Generate color sequence for the second-last row based on block arrangement
        color_sequence = self._generate_color_sequence(blocks_info, block_style)
        
        # Place color sequence in the second-last row
        self._place_color_sequence(grid, color_sequence, height - 2)
        
        # Store block information in grid attributes for transformation
        # We're using globals since arc_task_generator doesn't pass these to transform_input
        globals()['blocks_info'] = blocks_info
        globals()['color_sequence'] = color_sequence
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform the input grid by creating paths between blocks according to the color sequence."""
        # Create a copy of the input grid to modify
        output_grid = grid.copy()
        
        # Get object color and background color
        object_color = taskvars['object_color']
        background_color = taskvars['background_color']
        
        # Retrieve block information and color sequence from globals
        blocks_info = globals().get('blocks_info', [])
        color_sequence = globals().get('color_sequence', [])
        
        if not blocks_info or not color_sequence:
            # If we can't get block info from globals, try to detect them
            blocks_info, color_sequence = self._detect_blocks_and_sequence(grid, object_color, background_color)
        
        # Create a mapping from interior color to block info
        color_to_block = {block['interior_color']: block for block in blocks_info}
        
        # Create paths between blocks according to the color sequence
        for i in range(len(color_sequence) - 1):
            curr_color = color_sequence[i]
            next_color = color_sequence[i+1]
            
            curr_block = color_to_block[curr_color]
            next_block = color_to_block[next_color]
            
            # Create a path from current block to next block
            self._create_path(output_grid, curr_block, next_block, curr_color, object_color, background_color)
        
        return output_grid
    
    def _generate_block_positions_six_blocks(self, height: int, width: int, block_size: int) -> List[Tuple[int, int]]:
        """Generate positions for 6 blocks: 3 in top row, 3 in bottom row."""
        positions = []
        
        # Calculate possible positions
        max_blocks_per_row = 3
        min_gap = 3  # Minimum gap between blocks
        
        # First block always at (1,1)
        row1 = 1
        col1 = 1
        positions.append((row1, col1))
        
        # Calculate width of a block plus gap
        block_width = block_size + min_gap
        
        # Second and third blocks in the first row
        for i in range(1, max_blocks_per_row):
            col = col1 + i * block_width
            if col + block_size < width:
                positions.append((row1, col))
        
        # Calculate second row position (ensure enough space from the last rows)
        row2 = row1 + block_size + min_gap
        
        # Make sure there's enough space for the block and the last three rows
        if row2 + block_size + 3 < height:
            # Three blocks in the second row
            for i in range(max_blocks_per_row):
                col = col1 + i * block_width
                if col + block_size < width:
                    positions.append((row2, col))
        
        return positions
    
    def _generate_block_positions_style(self, style: str, height: int, width: int, block_size: int) -> List[Tuple[int, int]]:
        """Generate positions for 7 or 8 blocks based on the specified style."""
        positions = []
        
        # Calculate possible positions
        max_blocks_per_row = 3
        min_gap = 3  # Minimum gap between blocks
        
        # First block always at (1,1)
        row1 = 1
        col1 = 1
        positions.append((row1, col1))
        
        # Calculate width of a block plus gap
        block_width = block_size + min_gap
        
        # Second and third blocks in the first row
        for i in range(1, max_blocks_per_row):
            col = col1 + i * block_width
            if col + block_size < width:
                positions.append((row1, col))
        
        # Calculate second row position
        row2 = row1 + block_size + min_gap
        
        # Make sure there's enough space for the block and the last three rows
        if row2 + block_size + 3 < height:
            # Three blocks in the second row
            for i in range(max_blocks_per_row):
                col = col1 + i * block_width
                if col + block_size < width:
                    positions.append((row2, col))
        
        # Third row
        row3 = row2 + block_size + min_gap
        
        # Make sure there's enough space for the block and the last three rows
        if row3 + block_size + 3 < height:
            # Different patterns for the third row based on style
            if style == 'style_one':  # { , , g}
                col = col1 + 2 * block_width
                if col + block_size < width:
                    positions.append((row3, col))
            elif style == 'style_two':  # {g, , }
                col = col1
                if col + block_size < width:
                    positions.append((row3, col))
            elif style == 'style_three':  # { ,g, }
                col = col1 + block_width
                if col + block_size < width:
                    positions.append((row3, col))
        
        return positions
    
    def _get_unique_colors(self, count: int, exclude_colors: List[int]) -> List[int]:
        """Get a list of unique colors different from excluded colors."""
        available_colors = [c for c in range(1, 10) if c not in exclude_colors]
        
        # If we need more colors than available, we'll have to reuse some
        if count > len(available_colors):
            return random.sample(available_colors * 2, count)
        
        # Otherwise, return a random selection
        return random.sample(available_colors, count)
    
    def _place_square_block(self, grid: np.ndarray, row: int, col: int, size: int, 
                          outer_color: int, inner_color: int) -> None:
        """Place a square block with different interior coloring."""
        # Fill the whole block with the outer color
        for r in range(row, row + size):
            for c in range(col, col + size):
                grid[r, c] = outer_color
        
        # Set the interior differently
        if size % 2 == 1:  # Odd size (3x3, 5x5) - single center
            center_r = row + size // 2
            center_c = col + size // 2
            grid[center_r, center_c] = inner_color
        else:  # Even size (4x4) - 2x2 center
            center_r = row + size // 2 - 1
            center_c = col + size // 2 - 1
            grid[center_r:center_r+2, center_c:center_c+2] = inner_color
    
    def _generate_color_sequence(self, blocks_info: List[Dict], style: str) -> List[int]:
        """Generate a color sequence based on block positions and style."""
        if not blocks_info:
            return []
        
        # Sort blocks by position (row first, then column)
        sorted_blocks = sorted(blocks_info, key=lambda b: (b['position'][0], b['position'][1]))
        
        # Extract colors
        colors = [block['interior_color'] for block in sorted_blocks]
        
        # Generate sequence based on the style
        if style == 'six_blocks':
            # Pick a random pattern for 6 blocks
            pattern = random.choice([
                lambda c: [c[0], c[1], c[2], c[5], c[4], c[3]],  # {a b c f e d}
                lambda c: [c[0], c[3], c[4], c[1], c[2], c[5]],  # {a d e b c f}
                lambda c: [c[0], c[3], c[4], c[5], c[2], c[1]]   # {a d e f c b}
            ])
            return pattern(colors)
        
        elif style == 'style_one':  # { , , g}
            # Random patterns for style one
            pattern = random.choice([
                lambda c: [c[0], c[1], c[2], c[5], c[4], c[3]],        # {a b c f e d}
                lambda c: [c[0], c[3], c[4], c[1], c[2], c[5], c[6]],  # {a d e b c f g}
            ])
            return pattern(colors)
        
        elif style == 'style_two':  # {g, , }
            pattern = random.choice([
                lambda c: [c[0], c[1], c[2], c[5], c[4], c[3], c[6]],  # {a b c f e d g}
                lambda c: [c[0], c[3], c[4], c[1], c[2], c[5]],        # {a d e b c f}
                lambda c: [c[0], c[1], c[2], c[5], c[4], c[3], c[6]]   # {a b c f e d g}
            ])
            return pattern(colors)
        
        elif style == 'style_three':  # { ,g, }
            pattern = random.choice([
                lambda c: [c[0], c[1], c[2], c[5], c[4], c[6]],        # {a b c f e g}
                lambda c: [c[0], c[3], c[4], c[1], c[2], c[5]],        # {a d e b c f}
                lambda c: [c[0], c[3], c[4], c[1], c[2], c[5]]         # {a d e b c f}
            ])
            return pattern(colors)
        
        # Default sequence for other cases
        return [block['interior_color'] for block in sorted_blocks]
    
    def _place_color_sequence(self, grid: np.ndarray, color_sequence: List[int], row: int) -> None:
        """Place the color sequence in the specified row with background color between them."""
        background_color = grid[0, 0]  # Assuming the top-left is background color
        
        col = 1  # Start from the first column
        for color in color_sequence:
            grid[row, col] = color
            col += 2  # Skip one cell for background color
    
    def _detect_blocks_and_sequence(self, grid: np.ndarray, object_color: int, background_color: int) -> Tuple[List[Dict], List[int]]:
        """Detect blocks and color sequence from the grid if not provided in globals."""
        height, width = grid.shape
        blocks_info = []
        
        # Find all connected objects with the object_color
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=background_color)
        
        # Filter for square objects with object_color
        for obj in objects:
            obj_colors = obj.colors
            if object_color in obj_colors:
                # Check if it's a square block
                bbox = obj.bounding_box
                h = bbox[0].stop - bbox[0].start
                w = bbox[1].stop - bbox[1].start
                
                if h == w:  # It's a square
                    # Find interior color
                    interior_colors = [c for c in obj_colors if c != object_color]
                    
                    if interior_colors:
                        interior_color = interior_colors[0]
                        blocks_info.append({
                            'position': (bbox[0].start, bbox[1].start),
                            'interior_color': interior_color,
                            'size': h
                        })
        
        # Sort blocks by position (row first, then column)
        blocks_info = sorted(blocks_info, key=lambda b: (b['position'][0], b['position'][1]))
        
        # Extract color sequence from the second-last row
        color_sequence = []
        second_last_row = grid[height-2, :]
        for col in range(width):
            if second_last_row[col] != background_color:
                color_sequence.append(second_last_row[col])
        
        return blocks_info, color_sequence
    
    def _create_path(self, grid: np.ndarray, curr_block: Dict, next_block: Dict, 
                    path_color: int, object_color: int, background_color: int) -> None:
        """Create a path from current block to the next block using the specified color."""
        # Get block positions and sizes
        curr_pos = curr_block['position']
        next_pos = next_block['position']
        block_size = curr_block['size']
        
        path_width = 1
        if block_size % 2 == 1:  # Odd size block (3x3, 5x5)
            # Center of the single center cell
            curr_center = (curr_pos[0] + block_size // 2, curr_pos[1] + block_size // 2)
            next_center = (next_pos[0] + block_size // 2, next_pos[1] + block_size // 2)
        else:  # Even size block (4x4)
            # Center of the 2x2 interior (between the 4 center cells)
            path_width = 2
            curr_center = (curr_pos[0] + (block_size // 2) - 1, curr_pos[1] + (block_size // 2) - 1)
            next_center = (next_pos[0] + (block_size // 2) - 1, next_pos[1] + (block_size // 2) - 1)
        
        # Determine if the path should be horizontal or vertical first
        if curr_pos[0] == next_pos[0]:  # Same row, horizontal path
            self._draw_horizontal_path(grid, curr_center, next_center, path_color, path_width, object_color, background_color)
        
        elif curr_pos[1] == next_pos[1]:  # Same column, vertical path
            self._draw_vertical_path(grid, curr_center, next_center, path_color, path_width, object_color, background_color)
        
        else:  # Need both horizontal and vertical path
            # Determine if we go horizontal first or vertical first
            if random.choice([True, False]):
                # Horizontal first, then vertical
                midpoint = (curr_center[0], next_center[1])
                self._draw_horizontal_path(grid, curr_center, midpoint, path_color, path_width, object_color, background_color)
                self._draw_vertical_path(grid, midpoint, next_center, path_color, path_width, object_color, background_color)
            else:
                # Vertical first, then horizontal
                midpoint = (next_center[0], curr_center[1])
                self._draw_vertical_path(grid, curr_center, midpoint, path_color, path_width, object_color, background_color)
                self._draw_horizontal_path(grid, midpoint, next_center, path_color, path_width, object_color, background_color)
    
    def _draw_horizontal_path(self, grid: np.ndarray, start: Tuple[float, float], end: Tuple[float, float], 
                            color: int, width: int, object_color: int, background_color: int) -> None:
        """Draw a horizontal path between two points."""
        start_row, start_col = int(start[0]), int(start[1])
        end_row, end_col = int(end[0]), int(end[1])
        
        # Ensure start is to the left of end
        if start_col > end_col:
            start_col, end_col = end_col, start_col
        
        # Draw horizontal line
        for col in range(start_col, end_col + 1):
            if width == 1:  # For 3x3 or 5x5 blocks
                if 0 <= start_row < grid.shape[0] and grid[start_row, col] == background_color:
                    grid[start_row, col] = color
            else:  # For 4x4 blocks with 2x2 interior
                for w in range(width):
                    row = start_row + w
                    if 0 <= row < grid.shape[0] and grid[row, col] == background_color:
                        grid[row, col] = color
    
    def _draw_vertical_path(self, grid: np.ndarray, start: Tuple[float, float], end: Tuple[float, float], 
                          color: int, width: int, object_color: int, background_color: int) -> None:
        """Draw a vertical path between two points."""
        start_row, start_col = int(start[0]), int(start[1])
        end_row, end_col = int(end[0]), int(end[1])
        
        # Ensure start is above end
        if start_row > end_row:
            start_row, end_row = end_row, start_row
        
        # Draw vertical line
        for row in range(start_row, end_row + 1):
            if width == 1:  # For 3x3 or 5x5 blocks
                if 0 <= start_col < grid.shape[1] and grid[row, start_col] == background_color:
                    grid[row, start_col] = color
            else:  # For 4x4 blocks with 2x2 interior
                for w in range(width):
                    col = start_col + w
                    if 0 <= col < grid.shape[1] and grid[row, col] == background_color:
                        grid[row, col] = color