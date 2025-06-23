from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import create_object, retry, random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class PuzzleReconstructionGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}×{vars['grid_size']}.",
            "Each grid contains a horizontal structure that spans the grid, starting from the first column and extending to the last column, several colored objects, with the remaining cells being empty (0).",
            "This structure consists of 5, 6, or 7 consecutive rows, all filled with the same color.",
            "These rows are located around the middle of the grid, specifically between {vars['grid_size']//2 - 4} and {vars['grid_size']//2 + 4}.",
            "The horizontal band includes small gaps (0s) at positions located within its first or last 1 to 3 rows.",
            "The colored objects appear above and below the horizontal structure and are not random.",
            "They are irregular fragments of a larger, vertically stacked rectangular block made up of two or three color segments (i.e., top, middle, and bottom parts).",
            "This block is split into smaller, puzzle-like pieces that are distributed around the horizontal bar.",
            "The puzzle-like pieces or fragments do not follow regular or uniform shapes (such as [[c,0,0,0,c],[c,c,c,c,c],[0,c,0,c,0]]), the shapes vary in both structure and size.",
            "Importantly, the bottom-most piece (the one that should end up at the bottom of the reconstructed block) is designed to have a horizontal or flat base, enabling a stable bottom alignment in the output.",
            "There can be either one or two possible rectangular blocks that can be reconstructed in output; if there is only one block, then gaps appear on one side of the horizontal structure, else there are two blocks, then gaps appear on both sides of the horizontal structure.",
            "The horizontal structure contains several gaps, with exactly one designed gap per block, where a corresponding puzzle fragment is meant to fit perfectly.",
            "This ensures that, during the reconstruction process, the composite rectangular block overlaps with the horizontal structure at the intended location."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying a horizontal structure that spans the entire width (from the first to the last column), along with fragments that appear to be parts of one or two vertically stacked rectangular blocks.",
            "The horizontal band includes small gaps (0s) in its first or last 1 to 3 rows. If gaps appear on both sides, then two rectangular blocks can be reconstructed in the output. Else gaps appear on only one side, then only one block is reconstructed.",
            "The fragments are grouped by matching their edge shapes and layer colors to reassemble the full rectangular block(s).",
            "The fragment with a flat base is identified as the bottom-most part of the rectangular block.",
            "The first fragment that appears near the top of the grid and matches the shape and color layer should be selected and aligned so that it fits perfectly into the designed gap in the horizontal structure.",
            "The remaining fragments are stacked below it, following the correct vertical order of color segments (top to bottom), to complete the reconstructed block."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def _create_rectangular_block(self, width: int, height: int, colors: List[int]) -> np.ndarray:
        """Create a rectangular block with color segments."""
        block = np.zeros((height, width), dtype=int)
        
        if len(colors) == 2:
            # Two segments
            mid = height // 2
            block[:mid, :] = colors[0]
            block[mid:, :] = colors[1]
        else:  # len(colors) == 3
            # Three segments
            seg_height = height // 3
            block[:seg_height, :] = colors[0]
            block[seg_height:2*seg_height, :] = colors[1]
            block[2*seg_height:, :] = colors[2]
        
        return block
    
    def _create_puzzle_fragments(self, block: np.ndarray) -> List[np.ndarray]:
        """Create irregular puzzle fragments from a rectangular block."""
        height, width = block.shape
        fragments = []
        
        # Split block into 2-4 horizontal segments
        num_fragments = random.randint(2, 4)
        segment_heights = []
        remaining_height = height
        
        for i in range(num_fragments - 1):
            seg_h = random.randint(1, max(1, remaining_height - (num_fragments - i - 1)))
            segment_heights.append(seg_h)
            remaining_height -= seg_h
        segment_heights.append(remaining_height)
        
        current_row = 0
        for i, seg_height in enumerate(segment_heights):
            # Extract segment
            segment = block[current_row:current_row + seg_height, :].copy()
            
            # Make it irregular by adding holes and jagged edges
            for r in range(seg_height):
                for c in range(width):
                    # Add some randomness but preserve structure
                    if random.random() < 0.15:  # 15% chance to remove cell
                        # Don't remove from critical connection points
                        if not ((r == 0 or r == seg_height - 1) and c in [0, width-1]):
                            segment[r, c] = 0
            
            # Ensure the bottom piece has a flat base (for the last fragment)
            if i == num_fragments - 1:
                # Make sure bottom row is complete
                bottom_row = seg_height - 1
                for c in range(width):
                    if block[current_row + bottom_row, c] != 0:
                        segment[bottom_row, c] = block[current_row + bottom_row, c]
            
            fragments.append(segment)
            current_row += seg_height
        
        return fragments
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        is_test = gridvars.get('is_test', False)
        num_blocks = gridvars.get('num_blocks', 1)
        
        if is_test:
            return self._create_test_input(grid_size, num_blocks)
        else:
            return self._create_train_input(grid_size, num_blocks)
    
    def _create_train_input(self, grid_size: int, num_blocks: int) -> np.ndarray:
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Create horizontal band structure
        band_height = random.randint(5, 7)
        band_start = random.randint(grid_size//2 - 4, grid_size//2 + 4 - band_height)
        band_color = random.randint(1, 9)
        
        # Fill the band
        grid[band_start:band_start + band_height, :] = band_color
        
        # Store block information for reconstruction
        self.blocks_info = []
        
        available_colors = [c for c in range(1, 10) if c != band_color]
        
        if num_blocks == 1:
            # Single block with gap on one side
            gap_side = random.choice(['left', 'right'])
            block_info = self._create_single_block(grid, band_start, band_height, gap_side, available_colors, grid_size)
            self.blocks_info.append(block_info)
        else:
            # Two blocks with gaps on both sides
            block_info1 = self._create_single_block(grid, band_start, band_height, 'left', available_colors[:3], grid_size)
            block_info2 = self._create_single_block(grid, band_start, band_height, 'right', available_colors[3:6], grid_size)
            self.blocks_info.extend([block_info1, block_info2])
        
        return grid
    
    def _create_single_block(self, grid: np.ndarray, band_start: int, band_height: int, gap_side: str, colors: List[int], grid_size: int) -> Dict:
        """Create a single block with fragments and gap."""
        block_width = random.randint(4, 6)
        block_height = random.randint(6, 9)
        
        # Choose colors for the block
        block_colors = random.sample(colors, random.randint(2, 3))
        
        # Create the complete rectangular block
        complete_block = self._create_rectangular_block(block_width, block_height, block_colors)
        
        # Create puzzle fragments
        fragments = self._create_puzzle_fragments(complete_block)
        
        # Determine gap position in band
        if gap_side == 'left':
            gap_col = random.randint(2, grid_size // 3)
        else:  # right
            gap_col = random.randint(2 * grid_size // 3, grid_size - block_width - 2)
        
        # Create gap in band (should accommodate one fragment that bridges)
        gap_height = random.randint(2, min(3, band_height - 1))
        gap_row = band_start + random.randint(0, band_height - gap_height)
        
        # Make the gap
        grid[gap_row:gap_row + gap_height, gap_col:gap_col + block_width] = 0
        
        # Place fragments around the band
        fragment_positions = []
        for i, fragment in enumerate(fragments):
            fh, fw = fragment.shape
            
            # Place some fragments above, some below the band
            if i % 2 == 0 and band_start - fh > 1:
                # Place above
                row = random.randint(1, band_start - fh)
            else:
                # Place below
                row = random.randint(band_start + band_height + 1, grid_size - fh - 1)
            
            col = random.randint(1, grid_size - fw - 1)
            
            # Place fragment
            for r in range(fh):
                for c in range(fw):
                    if fragment[r, c] != 0:
                        grid[row + r, col + c] = fragment[r, c]
            
            fragment_positions.append((row, col))
        
        return {
            'complete_block': complete_block,
            'fragments': fragments,
            'fragment_positions': fragment_positions,
            'gap_position': (gap_row, gap_col),
            'gap_size': (gap_height, block_width),
            'colors': block_colors
        }
    
    def _create_test_input(self, grid_size: int, num_blocks: int) -> np.ndarray:
        """Create test input with vertical structure (rotated 90 degrees)."""
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Create vertical band structure
        band_width = random.randint(5, 7)
        band_start = random.randint(grid_size//2 - 4, grid_size//2 + 4 - band_width)
        band_color = random.randint(1, 9)
        
        # Fill the vertical band
        grid[:, band_start:band_start + band_width] = band_color
        
        # Store block information for reconstruction
        self.blocks_info = []
        
        available_colors = [c for c in range(1, 10) if c != band_color]
        
        # Create horizontal blocks (for vertical reconstruction)
        block_height = random.randint(4, 6)  # Now width in horizontal direction
        block_width = random.randint(6, 9)   # Now height in vertical direction
        
        block_colors = random.sample(available_colors, random.randint(2, 3))
        
        # Create complete block (horizontal orientation)
        complete_block = self._create_rectangular_block(block_height, block_width, block_colors)
        
        # Create fragments (rotated logic)
        fragments = self._create_puzzle_fragments_horizontal(complete_block)
        
        # Create gap in vertical band
        gap_row = random.randint(grid_size//4, 3*grid_size//4 - block_width)
        gap_col = band_start + random.randint(0, band_width - 2)
        gap_width = 2
        
        grid[gap_row:gap_row + block_width, gap_col:gap_col + gap_width] = 0
        
        # Place fragments on left and right sides
        fragment_positions = []
        for i, fragment in enumerate(fragments):
            fh, fw = fragment.shape
            
            # Place fragments left or right of the band
            if i % 2 == 0 and band_start - fw > 1:
                # Place left
                col = random.randint(1, band_start - fw)
            else:
                # Place right
                col = random.randint(band_start + band_width + 1, grid_size - fw - 1)
            
            row = random.randint(1, grid_size - fh - 1)
            
            # Place fragment
            for r in range(fh):
                for c in range(fw):
                    if fragment[r, c] != 0:
                        grid[row + r, col + c] = fragment[r, c]
            
            fragment_positions.append((row, col))
        
        block_info = {
            'complete_block': complete_block,
            'fragments': fragments,
            'fragment_positions': fragment_positions,
            'gap_position': (gap_row, gap_col),
            'gap_size': (block_width, gap_width),
            'colors': block_colors,
            'is_vertical': True
        }
        self.blocks_info = [block_info]
        
        return grid
    
    def _create_puzzle_fragments_horizontal(self, block: np.ndarray) -> List[np.ndarray]:
        """Create fragments for horizontal stacking (vertical band case)."""
        height, width = block.shape
        fragments = []
        
        # Split block into vertical segments for horizontal reconstruction
        num_fragments = random.randint(2, 4)
        segment_widths = []
        remaining_width = width
        
        for i in range(num_fragments - 1):
            seg_w = random.randint(1, max(1, remaining_width - (num_fragments - i - 1)))
            segment_widths.append(seg_w)
            remaining_width -= seg_w
        segment_widths.append(remaining_width)
        
        current_col = 0
        for i, seg_width in enumerate(segment_widths):
            # Extract segment
            segment = block[:, current_col:current_col + seg_width].copy()
            
            # Make it irregular
            for r in range(height):
                for c in range(seg_width):
                    if random.random() < 0.15:
                        if not ((c == 0 or c == seg_width - 1) and r in [0, height-1]):
                            segment[r, c] = 0
            
            fragments.append(segment)
            current_col += seg_width
        
        return fragments
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform input by reconstructing the puzzle blocks."""
        output_grid = grid.copy()
        
        if not hasattr(self, 'blocks_info'):
            return output_grid
        
        for block_info in self.blocks_info:
            is_vertical = block_info.get('is_vertical', False)
            if is_vertical:
                self._reconstruct_vertical_block(output_grid, block_info)
            else:
                self._reconstruct_horizontal_block(output_grid, block_info)
        
        return output_grid
    
    def _reconstruct_horizontal_block(self, grid: np.ndarray, block_info: Dict):
        """Reconstruct a horizontal block by stacking fragments vertically."""
        complete_block = block_info['complete_block']
        gap_row, gap_col = block_info['gap_position']
        gap_height, gap_width = block_info['gap_size']
        
        # Place the complete block so it overlaps with the gap
        block_height, block_width = complete_block.shape
        
        # Position block so it bridges the gap
        start_row = gap_row - (block_height - gap_height)
        start_col = gap_col
        
        # Place the reconstructed block
        for r in range(block_height):
            for c in range(block_width):
                if complete_block[r, c] != 0:
                    new_row = start_row + r
                    new_col = start_col + c
                    if 0 <= new_row < grid.shape[0] and 0 <= new_col < grid.shape[1]:
                        grid[new_row, new_col] = complete_block[r, c]
    
    def _reconstruct_vertical_block(self, grid: np.ndarray, block_info: Dict):
        """Reconstruct a vertical block by stacking fragments horizontally."""
        complete_block = block_info['complete_block']
        gap_row, gap_col = block_info['gap_position']
        gap_height, gap_width = block_info['gap_size']
        
        # Place the complete block so it overlaps with the gap
        block_height, block_width = complete_block.shape
        
        # Position block so it bridges the gap
        start_row = gap_row
        start_col = gap_col - (block_width - gap_width)
        
        # Place the reconstructed block
        for r in range(block_height):
            for c in range(block_width):
                if complete_block[r, c] != 0:
                    new_row = start_row + r
                    new_col = start_col + c
                    if 0 <= new_row < grid.shape[0] and 0 <= new_col < grid.shape[1]:
                        grid[new_row, new_col] = complete_block[r, c]
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        taskvars = {
            'grid_size': random.randint(20, 25)  # Smaller range to ensure visibility
        }
        
        # Create training examples
        train_examples = []
        
        # First example: single block
        gridvars1 = {'num_blocks': 1, 'is_test': False}
        input1 = self.create_input(taskvars, gridvars1)
        output1 = self.transform_input(input1, taskvars)
        train_examples.append({'input': input1, 'output': output1})
        
        # Second example: two blocks  
        gridvars2 = {'num_blocks': 2, 'is_test': False}
        input2 = self.create_input(taskvars, gridvars2)
        output2 = self.transform_input(input2, taskvars)
        train_examples.append({'input': input2, 'output': output2})
        
        # Third example: single block (different configuration)
        gridvars3 = {'num_blocks': 1, 'is_test': False}
        input3 = self.create_input(taskvars, gridvars3)
        output3 = self.transform_input(input3, taskvars)
        train_examples.append({'input': input3, 'output': output3})
        
        # Test example: vertical structure
        gridvars_test = {'num_blocks': 1, 'is_test': True}
        input_test = self.create_input(taskvars, gridvars_test)
        output_test = self.transform_input(input_test, taskvars)
        test_examples = [{'input': input_test, 'output': output_test}]
        
        train_test_data = TrainTestData(train=train_examples, test=test_examples)
        
        return taskvars, train_test_data

# Test the generator
if __name__ == "__main__":
    generator = PuzzleReconstructionGenerator()
    taskvars, train_test_data = generator.create_grids()
    
    print("Task Variables:", taskvars)
    ARCTaskGenerator.visualize_train_test_data(train_test_data)