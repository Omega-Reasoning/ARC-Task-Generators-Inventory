from arc_task_generator import ARCTaskGenerator, TrainTestData
from typing import Dict, Any, Tuple
import numpy as np
import random

class Task18286ef8Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "Each grid is divided into 9 rectangular blocks of varying sizes.",
            "Let the number of rows be a. Compute a - 5, and divide it into two parts: a - 5 = b + c.",
            "Let the number of columns be d. Compute d - 5, and divide it into two parts: d - 5 = e + f.",
            "Now create 9 blocks using these values: b×e, b×3, b×f, 3×e, 3×3, 3×f, c×e, c×3, c×f.",
            "The blocks are arranged with one empty row and column separation.",
            "Each block is completely filled with {color('object_color')}.",
            "The center block (3×3) is filled with {color('object_color2')} and its interior cells colored {color('interior_color')}.",
            "Add a single differently colored cell in most of the non-center blocks and one of the blocks should get {color('new_color')} colored cell.",
            "Blocks are grouped in sets of three aligned vertically or horizontally."
        ]
        
        transformation_reasoning_chain = [
            "Copy the input grid.",
            "Find the block containing {color('new_color')} cell.",
            "Change the color of this cell from {color('new_color')} to {color('interior_color')}.",
            "Move the {color('interior_color')} cell in the center block based on the position of the identifed block in step 2. For example, if the identified block is in the top-left corner, move the {color('interior_color')} cell in the center block from position (1,1) to (0,0).",
            "No other cells change."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        a = taskvars['a']
        d = taskvars['d']
        b = taskvars['b']
        c = taskvars['c']
        e = taskvars['e']
        f = taskvars['f']
        
        grid = np.zeros((a, d), dtype=int)
        
        block_sizes = [
            (b, e), (b, 3), (b, f),
            (3, e), (3, 3), (3, f),
            (c, e), (c, 3), (c, f)
        ]
        
        row_starts = [0, b + 1, b + 1 + 3 + 1]
        col_starts = [0, e + 1, e + 1 + 3 + 1]
        
        positions = []
        for i in range(3):
            for j in range(3):
                positions.append((row_starts[i], col_starts[j]))
        
        for idx, ((r_start, c_start), (height, width)) in enumerate(zip(positions, block_sizes)):
            if idx == 4:
                grid[r_start:r_start+height, c_start:c_start+width] = taskvars['object_color2']
                grid[r_start+1, c_start+1] = taskvars['interior_color']
            else:
                grid[r_start:r_start+height, c_start:c_start+width] = taskvars['object_color']
        
        new_color_block = gridvars.get('new_color_block', random.randint(0, 8))
        if new_color_block == 4:  # Skip center block
            new_color_block = random.choice([0, 1, 2, 3, 5, 6, 7, 8])
        
        for idx, ((r_start, c_start), (height, width)) in enumerate(zip(positions, block_sizes)):
            if idx == 4:
                continue
            if idx == new_color_block:
                r_pos = r_start + random.randint(0, height-1)
                c_pos = c_start + random.randint(0, width-1)
                grid[r_pos, c_pos] = taskvars['new_color']
            else:
                available_colors = [i for i in range(1, 10) if i not in [
                    taskvars['object_color'], taskvars['object_color2'], 
                    taskvars['interior_color'], taskvars['new_color']
                ]]
                if available_colors:
                    r_pos = r_start + random.randint(0, height-1)
                    c_pos = c_start + random.randint(0, width-1)
                    grid[r_pos, c_pos] = random.choice(available_colors)
        
        return grid
    
    

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output = grid.copy()
        
        # Derive grid parameters from grid dimensions (since grids can have different sizes)
        a, d = grid.shape
        
        # Find the 3x3 center block to derive parameters
        # Look for the center block mostly filled with object_color2 and containing interior_color
        center_found = False
        for potential_b in range(1, a-4):
            for potential_e in range(1, d-4):
                center_r = potential_b + 1
                center_c = potential_e + 1
                
                # Check if this could be the center block (should have mostly object_color2 and one interior_color)
                if (center_r + 3 <= a and center_c + 3 <= d):
                    center_block = output[center_r:center_r+3, center_c:center_c+3]
                    color2_count = np.sum(center_block == taskvars['object_color2'])
                    interior_count = np.sum(center_block == taskvars['interior_color'])
                    
                    # Center block should have 8 object_color2 cells and 1 interior_color cell
                    if color2_count == 8 and interior_count == 1:
                        # Found center block, derive parameters
                        b = potential_b
                        e = potential_e
                        c = a - 5 - b
                        f = d - 5 - e
                        center_found = True
                        break
            if center_found:
                break
        
        if not center_found:
            return output  # If we can't find center block, return unchanged
        
        # Define block positions and sizes
        block_sizes = [
            (b, e), (b, 3), (b, f),
            (3, e), (3, 3), (3, f),
            (c, e), (c, 3), (c, f)
        ]
        
        row_starts = [0, b + 1, b + 1 + 3 + 1]
        col_starts = [0, e + 1, e + 1 + 3 + 1]
        
        positions = []
        for i in range(3):
            for j in range(3):
                positions.append((row_starts[i], col_starts[j]))
        
        # Find the block containing new_color
        new_color_block_idx = None
        for block_idx, ((r_start, c_start), (height, width)) in enumerate(zip(positions, block_sizes)):
            if block_idx == 4:  # Skip center block
                continue
            block_region = output[r_start:r_start + height, c_start:c_start + width]
            if np.any(block_region == taskvars['new_color']):
                new_color_block_idx = block_idx
                break
        
        if new_color_block_idx is not None:
            # Replace new_color with interior_color in the target block
            r_start, c_start = positions[new_color_block_idx]
            height, width = block_sizes[new_color_block_idx]
            for r in range(r_start, r_start + height):
                for c in range(c_start, c_start + width):
                    if output[r, c] == taskvars['new_color']:
                        output[r, c] = taskvars['interior_color']
                        break
            
            # Move center block interior cell
            center_r_start, center_c_start = positions[4]
            
            # Find current interior color position and reset it to object_color2
            for r in range(center_r_start, center_r_start + 3):
                for c in range(center_c_start, center_c_start + 3):
                    if output[r, c] == taskvars['interior_color']:
                        output[r, c] = taskvars['object_color2']
                        break
            
            # Calculate new position based on block index
            block_row = new_color_block_idx // 3
            block_col = new_color_block_idx % 3
            
            new_interior_r = center_r_start + block_row
            new_interior_c = center_c_start + block_col
            
            output[new_interior_r, new_interior_c] = taskvars['interior_color']
        
        return output
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate different colors (shared across all grids)
        colors = random.sample(range(1, 10), 4)
        
        # Base task variables (colors only)
        base_taskvars = {
            'object_color': colors[0],
            'object_color2': colors[1],
            'interior_color': colors[2],
            'new_color': colors[3]
        }
        
        # Generate training examples with varying grid sizes
        train_examples = []
        num_train = random.randint(3, 6)
        
        for _ in range(num_train):
            # Generate random grid size for this example
            a = random.randint(10, 30)
            d = random.randint(10, 30)
            
            # Ensure valid divisions
            a_remainder = a - 5
            d_remainder = d - 5
            
            b = random.randint(1, max(1, a_remainder - 1))
            c = a_remainder - b
            
            e = random.randint(1, max(1, d_remainder - 1))
            f = d_remainder - e
            
            # Create complete taskvars for this grid
            taskvars = {
                **base_taskvars,
                'a': a,
                'd': d,
                'b': b,
                'c': c,
                'e': e,
                'f': f
            }
            
            new_color_block = random.choice([0, 1, 2, 3, 5, 6, 7, 8])  # Exclude center block (4)
            gridvars = {'new_color_block': new_color_block}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example with different grid size
        test_a = random.randint(12, 30)
        test_d = random.randint(12, 30)
        
        # Ensure valid divisions
        test_a_remainder = test_a - 5
        test_d_remainder = test_d - 5
        
        test_b = random.randint(1, max(1, test_a_remainder - 1))
        test_c = test_a_remainder - test_b
        
        test_e = random.randint(1, max(1, test_d_remainder - 1))
        test_f = test_d_remainder - test_e
        
        test_taskvars = {
            **base_taskvars,
            'a': test_a,
            'd': test_d,
            'b': test_b,
            'c': test_c,
            'e': test_e,
            'f': test_f
        }
        
        test_new_color_block = random.choice([0, 1, 2, 3, 5, 6, 7, 8])
        test_gridvars = {'new_color_block': test_new_color_block}
        
        test_input = self.create_input(test_taskvars, test_gridvars)
        test_output = self.transform_input(test_input, test_taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        # Return only color variables for template instantiation (grid dimensions are derived from grid)
        return base_taskvars, train_test_data



