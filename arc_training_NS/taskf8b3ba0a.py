from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import random_cell_coloring, retry
import numpy as np
import random
from collections import Counter
from typing import Dict, Any, Tuple, List

class Taskf8b3ba0a(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids vary in size, defined as m × n, where m is an odd number n % 3 = 1 and.",
            "In each input grid: all the cells on rows at indices 2i (0-based indexing) and columns at indices 3j (0-based indexing) are empty.",
            "The remaining non-empty cells form blocks, with each block consisting of two horizontally adjacent cells.",
            "The majority of blocks are colored {color('color_main')}.",
            "The remaining blocks are partitioned into three distinct subsets, each assigned a different color, with a different number of blocks in each subset."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is initialized as an empty 3x1 grid.",
            "All horizontal blocks of size 2 in the input grid are identified along with their corresponding colors.",
            "The blocks are counted by color to determine their frequencies.",
            "The output grid is then colored based on these frequencies: The top cell is colored with the color that has the second-highest number of blocks. The middle cell is colored with the color that has the third-highest number of blocks. The bottom cell is colored with the color that occurs least frequently."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        # Get grid dimensions from gridvars
        m = gridvars['m']
        n = gridvars['n']
        
        # Initialize empty grid
        grid = np.zeros((m, n), dtype=int)
        
        # Get available positions for blocks (avoiding restricted rows/columns)
        available_positions = []
        for r in range(m):
            if r % 2 != 0:  # Skip rows at indices 2i (0, 2, 4, ...)
                for c in range(n-1):  # Need space for horizontal 2-cell block
                    if c % 3 != 0 and (c+1) % 3 != 0:  # Skip columns at indices 3j (0, 3, 6, ...)
                        available_positions.append((r, c))
        
        # Get the main color from task variables
        color_main = taskvars['color_main']
        
        # Generate 3 other colors (different from main color)
        other_colors = [c for c in range(1, 10) if c != color_main]
        random.shuffle(other_colors)
        color_2nd = other_colors[0]
        color_3rd = other_colors[1] 
        color_4th = other_colors[2]
        
        # Calculate how many blocks we can place
        total_available = len(available_positions)
        
        # Set frequencies ensuring main color is most frequent
        # and the other three have different frequencies
        freq_4th = max(1, total_available // 15)  # Least frequent (at least 1)
        freq_3rd = max(freq_4th + 1, total_available // 10)  # 3rd most frequent
        freq_2nd = max(freq_3rd + 1, total_available // 5)  # 2nd most frequent
        freq_main = total_available - freq_2nd - freq_3rd - freq_4th  # Remaining blocks
        
        # Ensure main color has the most blocks
        if freq_main <= freq_2nd:
            # Redistribute to ensure proper ordering
            remaining = total_available
            freq_4th = max(1, remaining // 8)
            freq_3rd = max(freq_4th + 1, remaining // 6)
            freq_2nd = max(freq_3rd + 1, remaining // 4)
            freq_main = remaining - freq_2nd - freq_3rd - freq_4th
        
        # Store frequencies in gridvars for the transform function
        gridvars['freq_main'] = freq_main
        gridvars['freq_2nd'] = freq_2nd
        gridvars['freq_3rd'] = freq_3rd
        gridvars['freq_4th'] = freq_4th
        gridvars['color_2nd'] = color_2nd
        gridvars['color_3rd'] = color_3rd
        gridvars['color_4th'] = color_4th
        
        # Create list of blocks to place
        blocks_to_place = []
        blocks_to_place.extend([color_main] * freq_main)
        blocks_to_place.extend([color_2nd] * freq_2nd)
        blocks_to_place.extend([color_3rd] * freq_3rd)
        blocks_to_place.extend([color_4th] * freq_4th)
        
        # Shuffle block colors and positions
        random.shuffle(blocks_to_place)
        random.shuffle(available_positions)
        
        # Place ALL blocks (no empty blocks)
        for i, color in enumerate(blocks_to_place):
            r, c = available_positions[i]
            grid[r, c] = color
            grid[r, c+1] = color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Find all horizontal 2-cell blocks and count by color
        color_counts = {}
        
        m, n = grid.shape
        for r in range(m):
            for c in range(n-1):
                if grid[r, c] != 0 and grid[r, c] == grid[r, c+1]:
                    # Found a horizontal 2-cell block
                    color = grid[r, c]
                    if color in color_counts:
                        color_counts[color] += 1
                    else:
                        color_counts[color] = 1
        
        # Sort colors by frequency (descending)
        color_freq_pairs = []
        for color, freq in color_counts.items():
            color_freq_pairs.append((color, freq))
        
        # Sort by frequency in descending order
        for i in range(len(color_freq_pairs)):
            for j in range(i + 1, len(color_freq_pairs)):
                if color_freq_pairs[i][1] < color_freq_pairs[j][1]:
                    color_freq_pairs[i], color_freq_pairs[j] = color_freq_pairs[j], color_freq_pairs[i]
        
        # Create 3x1 output grid
        output = np.zeros((3, 1), dtype=int)
        
        # Fill output: 2nd most frequent, 3rd most frequent, least frequent
        if len(color_freq_pairs) >= 4:
            output[0, 0] = color_freq_pairs[1][0]  # 2nd highest
            output[1, 0] = color_freq_pairs[2][0]  # 3rd highest  
            output[2, 0] = color_freq_pairs[3][0]  # 4th highest (least frequent)
        
        return output

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables - only the main color
        color_main = random.randint(1, 9)
        
        taskvars = {
            'color_main': color_main
        }
        
        # Generate training and test examples
        num_train = random.randint(3, 6)
        examples = []
        
        def get_valid_dimensions():
            """Generate valid grid dimensions that can fit enough blocks."""
            min_blocks = 10  # Minimum number of blocks needed (1+2+3+4)
            
            for _ in range(100):  # Try up to 100 times
                m = random.choice([7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29])  # odd numbers
                n_candidates = [x for x in range(10, 31) if x % 3 == 1]  # n % 3 = 1, larger grids
                n = random.choice(n_candidates)
                
                # Count available positions for blocks
                available_count = 0
                for r in range(m):
                    if r % 2 != 0:  # Skip rows at indices 2i
                        for c in range(n-1):
                            if c % 3 != 0 and (c+1) % 3 != 0:  # Skip columns at indices 3j
                                available_count += 1
                
                if available_count >= min_blocks:
                    return m, n
            
            raise ValueError("Cannot find valid grid dimensions")
        
        for _ in range(num_train + 1):  # +1 for test case
            m, n = get_valid_dimensions()
            
            gridvars = {'m': m, 'n': n}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        train_test_data = {
            'train': examples[:-1],
            'test': examples[-1:]
        }
        
        return taskvars, train_test_data

