from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import random_cell_coloring, retry
import numpy as np
import random
from collections import Counter
from typing import Dict, Any, Tuple, List

class Taskf8b3ba0aGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids vary in size, defined as m × n, where m is an odd number and n % 3 = 1.",
            "In each input grid: all the cells on rows at indices 2i (0-based indexing) and columns at indices 3j (0-based indexing) are empty.",
            "The remaining non-empty cells form blocks, with each block consisting of two horizontally adjacent cells.",
            "The majority of blocks are colored {color('color_main')}.",
            "The remaining blocks are partitioned into {vars['num_subsets']} distinct subsets, each assigned a different color, with a different number of blocks in each subset."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is initialized as an empty {vars['num_subsets']} x 1 grid.",
            "All horizontal blocks of size 2 in the input grid are identified along with their corresponding colors.",
            "The blocks are counted by color to determine their frequencies.",
            "The color with the highest frequency is excluded, and the output grid is then filled from top to bottom using the remaining colors, ordered from the highest to the lowest frequency."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        # Get grid dimensions from gridvars
        m = gridvars['m']
        n = gridvars['n']
        num_subsets = taskvars['num_subsets']
        
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
        
        # Generate num_subsets other colors (different from main color)
        other_colors = [c for c in range(1, 10) if c != color_main]
        random.shuffle(other_colors)
        subset_colors = other_colors[:num_subsets]
        
        # Calculate how many blocks we can place
        total_available = len(available_positions)
        
        # Reserve a significant portion for main color (40-60% of total)
        main_percentage = random.uniform(0.4, 0.6)
        freq_main = max(num_subsets + 2, int(total_available * main_percentage))
        
        # Remaining blocks for subsets
        remaining_for_subsets = total_available - freq_main
        
        # Ensure we have enough blocks
        if remaining_for_subsets < num_subsets:
            freq_main = total_available - num_subsets
            remaining_for_subsets = num_subsets
        
        # Generate diverse frequencies for subsets
        # Use a more varied distribution instead of sequential 1, 2, 3, ...
        subset_frequencies = []
        
        if num_subsets == 2:
            # For 2 subsets: pick two different numbers with good gap
            available_freqs = list(range(1, remaining_for_subsets))
            random.shuffle(available_freqs)
            # Ensure they're different and have a decent gap
            freq1 = random.randint(max(1, remaining_for_subsets // 3), remaining_for_subsets - 1)
            freq2 = remaining_for_subsets - freq1
            if freq1 == freq2:
                if freq1 > 1:
                    freq1 -= 1
                    freq2 += 1
                else:
                    freq2 += 1
            subset_frequencies = sorted([freq1, freq2], reverse=True)
            
        elif num_subsets == 3:
            # For 3 subsets: more diverse distribution
            # Example patterns: [8, 5, 2], [10, 6, 3], [7, 4, 1], etc.
            min_val = max(1, remaining_for_subsets // 15)
            mid_val = max(min_val + 2, remaining_for_subsets // 5)
            max_val = remaining_for_subsets - min_val - mid_val
            
            # Ensure they're all different and positive
            if max_val <= mid_val:
                max_val = mid_val + random.randint(2, 5)
                mid_val = (remaining_for_subsets - max_val - min_val) if (remaining_for_subsets - max_val - min_val) > min_val else mid_val
                min_val = remaining_for_subsets - max_val - mid_val
            
            # Ensure min_val is at least 1
            if min_val < 1:
                min_val = 1
                mid_val = max(min_val + 2, (remaining_for_subsets - min_val) // 2)
                max_val = remaining_for_subsets - min_val - mid_val
            
            subset_frequencies = [max_val, mid_val, min_val]
            
        elif num_subsets == 4:
            # For 4 subsets: even more diverse
            # Example: [12, 7, 4, 2] instead of [4, 3, 2, 1]
            min_val = max(1, remaining_for_subsets // 20)
            quarter = max(min_val + 1, remaining_for_subsets // 8)
            mid_val = max(quarter + 2, remaining_for_subsets // 4)
            max_val = remaining_for_subsets - min_val - quarter - mid_val
            
            if max_val <= mid_val:
                max_val = mid_val + random.randint(2, 6)
                remaining = remaining_for_subsets - max_val
                mid_val = max(quarter + 2, remaining // 2)
                quarter = max(min_val + 1, (remaining - mid_val) // 2)
                min_val = remaining - mid_val - quarter
            
            # Ensure all positive and different
            if min_val < 1:
                min_val = 1
            if quarter <= min_val:
                quarter = min_val + 1
            if mid_val <= quarter:
                mid_val = quarter + 2
            if max_val <= mid_val:
                max_val = mid_val + 3
            
            # Adjust to match total
            total = min_val + quarter + mid_val + max_val
            if total != remaining_for_subsets:
                diff = remaining_for_subsets - total
                max_val += diff
            
            subset_frequencies = [max_val, mid_val, quarter, min_val]
            
        else:  # num_subsets == 5
            # For 5 subsets: highly diverse
            freqs = []
            temp_remaining = remaining_for_subsets
            
            # Generate descending frequencies with varying gaps
            for i in range(num_subsets - 1):
                # Each frequency is a fraction of remaining, with randomization
                frac = random.uniform(0.2, 0.4)
                freq = max(i + 1, int(temp_remaining * frac))
                freqs.append(freq)
                temp_remaining -= freq
            
            # Last frequency gets the remainder
            freqs.append(max(1, temp_remaining))
            
            # Ensure they're all different
            freqs = sorted(set(freqs), reverse=True)
            while len(freqs) < num_subsets:
                # Add missing frequencies
                for i in range(len(freqs) - 1):
                    if freqs[i] - freqs[i+1] > 1:
                        freqs.insert(i+1, freqs[i] - 1)
                        break
                else:
                    freqs.append(freqs[-1] - 1 if freqs[-1] > 1 else 1)
                freqs = sorted(set(freqs), reverse=True)
            
            # Adjust to match total exactly
            current_sum = sum(freqs[:num_subsets])
            diff = remaining_for_subsets - current_sum
            freqs[0] += diff
            
            subset_frequencies = freqs[:num_subsets]
        
        # Ensure all frequencies are positive and different
        subset_frequencies = [max(1, f) for f in subset_frequencies]
        subset_frequencies = sorted(set(subset_frequencies), reverse=True)
        
        # If we lost some due to duplicates, regenerate
        while len(subset_frequencies) < num_subsets:
            subset_frequencies.append(1)
            # Make them different
            for i in range(len(subset_frequencies)):
                for j in range(i + 1, len(subset_frequencies)):
                    if subset_frequencies[i] == subset_frequencies[j]:
                        subset_frequencies[i] += 1
            subset_frequencies = sorted(set(subset_frequencies), reverse=True)
        
        subset_frequencies = subset_frequencies[:num_subsets]
        
        # Adjust frequencies to match exactly the remaining space
        current_sum = sum(subset_frequencies)
        if current_sum != remaining_for_subsets:
            diff = remaining_for_subsets - current_sum
            subset_frequencies[0] += diff
        
        # Final verification: main color should have the most
        if freq_main <= max(subset_frequencies):
            freq_main = max(subset_frequencies) + random.randint(2, 5)
            # Recalculate subsets
            remaining_for_subsets = total_available - freq_main
            if remaining_for_subsets < num_subsets:
                raise ValueError("Not enough space for diverse distribution")
            
            # Scale down proportionally
            scale = remaining_for_subsets / sum(subset_frequencies)
            subset_frequencies = [max(1, int(f * scale)) for f in subset_frequencies]
            # Adjust to match exactly
            current_sum = sum(subset_frequencies)
            subset_frequencies[0] += remaining_for_subsets - current_sum
        
        # Store in gridvars
        gridvars['freq_main'] = freq_main
        gridvars['subset_colors'] = subset_colors
        gridvars['subset_frequencies'] = subset_frequencies
        
        # Create list of blocks to place
        blocks_to_place = []
        blocks_to_place.extend([color_main] * freq_main)
        for i in range(num_subsets):
            blocks_to_place.extend([subset_colors[i]] * subset_frequencies[i])
        
        # Verify we're not placing more blocks than available positions
        if len(blocks_to_place) > len(available_positions):
            blocks_to_place = blocks_to_place[:len(available_positions)]
        
        # Shuffle block colors and positions
        random.shuffle(blocks_to_place)
        random.shuffle(available_positions)
        
        # Place blocks
        for i, color in enumerate(blocks_to_place):
            r, c = available_positions[i]
            grid[r, c] = color
            grid[r, c+1] = color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Find all horizontal 2-cell blocks and count by color
        color_counts = {}
        
        m, n = grid.shape
        visited = set()
        
        for r in range(m):
            for c in range(n-1):
                if (r, c) not in visited and grid[r, c] != 0 and grid[r, c] == grid[r, c+1]:
                    # Found a horizontal 2-cell block
                    color = grid[r, c]
                    if color in color_counts:
                        color_counts[color] += 1
                    else:
                        color_counts[color] = 1
                    # Mark as visited to avoid double counting
                    visited.add((r, c))
                    visited.add((r, c+1))
        
        # Sort colors by frequency (descending)
        color_freq_pairs = sorted(color_counts.items(), key=lambda x: (-x[1], x[0]))
        
        # Create output grid with size num_subsets x 1
        num_subsets = taskvars['num_subsets']
        output = np.zeros((num_subsets, 1), dtype=int)
        
        # Fill output: exclude the highest frequency (main color), 
        # then fill from top to bottom with remaining colors ordered by frequency
        if len(color_freq_pairs) >= num_subsets + 1:
            for i in range(num_subsets):
                output[i, 0] = color_freq_pairs[i + 1][0]  # Skip index 0 (main color)
        
        return output

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        color_main = random.randint(1, 9)
        num_subsets = random.randint(2, 5)  # Number of subsets (2 to 5)
        
        taskvars = {
            'color_main': color_main,
            'num_subsets': num_subsets
        }
        
        # Generate training and test examples
        num_train = random.randint(3, 6)
        examples = []
        
        def get_valid_dimensions():
            """Generate valid grid dimensions that can fit enough blocks."""
            # Need more space for diverse distributions
            min_blocks = num_subsets * 5 + 10  # Much larger buffer
            
            for attempt in range(100):  # Try up to 100 times
                m = random.choice([11, 13, 15, 17, 19, 21, 23, 25, 27, 29])  # odd numbers, larger
                n_candidates = [x for x in range(16, 31) if x % 3 == 1]  # n % 3 = 1, larger grids
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
            max_attempts = 20
            for attempt in range(max_attempts):
                try:
                    m, n = get_valid_dimensions()
                    
                    gridvars = {'m': m, 'n': n}
                    input_grid = self.create_input(taskvars, gridvars)
                    output_grid = self.transform_input(input_grid, taskvars)
                    
                    # Check if output is valid (not all zeros)
                    if np.any(output_grid != 0):
                        examples.append({
                            'input': input_grid,
                            'output': output_grid
                        })
                        break
                except (ValueError, IndexError) as e:
                    if attempt == max_attempts - 1:
                        raise
                    continue
        
        train_test_data = {
            'train': examples[:-1],
            'test': examples[-1:]
        }
        
        return taskvars, train_test_data