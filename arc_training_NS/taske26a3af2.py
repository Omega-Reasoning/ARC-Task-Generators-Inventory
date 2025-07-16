from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import GridObject, GridObjects, find_connected_objects
from input_library import random_cell_coloring, retry

import numpy as np
import random
from typing import Dict, Any, Tuple, List
from collections import Counter

class RegionNoiseRemovalGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}.",
            "Each input grid is split into 3 or 4 rectangular regions, and all regions are aligned either vertically or horizontally.",
            "Each region is filled with a solid base color, chosen randomly.",
            "A random number of cells in each region are colored differently from the base color of that region, and these cells are referred to as noise.",
            "Noise colors vary within the same region and across different regions.",
            "In each region, the amount of noise is at most 33% of the total cell count of that region.",
            "Noise cells are sparsely and randomly scattered inside each region."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid",
            "The grid is split into 3 or 4 rectangular regions, each consisting of a base color and some randomly placed noise cells.",
            "Noise cells are defined as cells whose color is different from the base color of the region they belong to.",
            "Each region is identified along with its base color and its noise cells.",
            "In each region, the color of the noise cells is transformed to match the base color of that region."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        taskvars = {
            'n': random.randint(8, 25)  # Slightly larger minimum to ensure regions are meaningful
        }
        
        # Create training and test examples
        num_train = random.randint(3, 6)
        train_examples = []
        test_examples = []
        
        # Generate training examples
        for _ in range(num_train):
            # Store region info for transformation
            input_grid, region_info = self.create_input_with_regions(taskvars, {})
            output_grid = self.transform_input_with_regions(input_grid, region_info, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example
        input_grid, region_info = self.create_input_with_regions(taskvars, {})
        output_grid = self.transform_input_with_regions(input_grid, region_info, taskvars)
        test_examples.append({'input': input_grid, 'output': output_grid})
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data

    def create_input_with_regions(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, int]]]:
        """Create input grid and return region information for transformation."""
        n = taskvars['n']
        grid = np.zeros((n, n), dtype=int)
        
        # Decide number of regions (3 or 4)
        num_regions = random.choice([3, 4])
        
        # Decide orientation (vertical or horizontal alignment)
        horizontal = random.choice([True, False])
        
        # Generate available colors (excluding background 0)
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        
        # Create regions
        regions = []
        region_info = []  # Will store (r1, r2, c1, c2, base_color)
        
        if horizontal:
            # Split horizontally (regions stacked vertically)
            total_height = n
            region_heights = self._split_dimension(total_height, num_regions)
            current_row = 0
            
            for i, height in enumerate(region_heights):
                region_bounds = (current_row, current_row + height, 0, n)
                regions.append(region_bounds)
                current_row += height
        else:
            # Split vertically (regions side by side)
            total_width = n
            region_widths = self._split_dimension(total_width, num_regions)
            current_col = 0
            
            for i, width in enumerate(region_widths):
                region_bounds = (0, n, current_col, current_col + width)
                regions.append(region_bounds)
                current_col += width
        
        # Fill each region with base color and add noise
        for i, (r1, r2, c1, c2) in enumerate(regions):
            # Choose base color for this region
            base_color = available_colors[i % len(available_colors)]
            
            # Fill region with base color
            grid[r1:r2, c1:c2] = base_color
            
            # Store region info
            region_info.append((r1, r2, c1, c2, base_color))
            
            # Add noise
            region_size = (r2 - r1) * (c2 - c1)
            max_noise = int(region_size * 0.33)  # At most 33% noise
            num_noise = random.randint(1, max_noise)
            
            if num_noise > 0:
                # Get all positions in this region
                positions = [(r, c) for r in range(r1, r2) for c in range(c1, c2)]
                noise_positions = random.sample(positions, num_noise)
                
                # Choose noise colors (different from base color)
                noise_colors = [color for color in available_colors if color != base_color]
                
                # Add noise to selected positions
                for r, c in noise_positions:
                    grid[r, c] = random.choice(noise_colors)
        
        return grid, region_info

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """Legacy method for compatibility - just returns the grid."""
        grid, _ = self.create_input_with_regions(taskvars, gridvars)
        return grid

    def transform_input_with_regions(self, grid: np.ndarray, region_info: List[Tuple[int, int, int, int, int]], taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform input using stored region information."""
        output_grid = grid.copy()
        
        # For each region, convert all cells to the base color
        for r1, r2, c1, c2, base_color in region_info:
            output_grid[r1:r2, c1:c2] = base_color
        
        return output_grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform input by detecting regions (for compatibility)."""
        n = taskvars['n']
        output_grid = grid.copy()
        
        # Try to detect regions
        regions = self._identify_regions_improved(grid)
        
        # For each region, find the base color and convert noise to base color
        for region_bounds in regions:
            r1, r2, c1, c2 = region_bounds
            region_cells = grid[r1:r2, c1:c2]
            
            # Find the most common color in this region (base color)
            colors, counts = np.unique(region_cells, return_counts=True)
            base_color = colors[np.argmax(counts)]
            
            # Set all cells in this region to the base color
            output_grid[r1:r2, c1:c2] = base_color
        
        return output_grid

    def _split_dimension(self, total_size: int, num_regions: int) -> List[int]:
        """Split a dimension into num_regions parts, ensuring each part is at least 2."""
        if num_regions * 2 > total_size:
            # Fallback to equal distribution if we can't ensure minimum size
            base_size = total_size // num_regions
            remainder = total_size % num_regions
            sizes = [base_size] * num_regions
            for i in range(remainder):
                sizes[i] += 1
            return sizes
        
        # Start with minimum size of 2 for each region
        sizes = [2] * num_regions
        remaining = total_size - (2 * num_regions)
        
        # Distribute remaining size randomly
        for _ in range(remaining):
            region_idx = random.randint(0, num_regions - 1)
            sizes[region_idx] += 1
        
        return sizes

    def _identify_regions_improved(self, grid: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Improved region detection using sliding window analysis."""
        n = grid.shape[0]
        
        # Try horizontal splits first (regions stacked vertically)
        horizontal_boundaries = self._find_horizontal_boundaries(grid)
        if len(horizontal_boundaries) >= 2:  # Need at least 2 boundaries for 3+ regions
            boundaries = [0] + horizontal_boundaries + [n]
            boundaries = sorted(list(set(boundaries)))  # Remove duplicates and sort
            if len(boundaries) >= 3:  # At least 3 boundaries for 2+ regions
                regions = []
                for i in range(len(boundaries) - 1):
                    if boundaries[i+1] - boundaries[i] >= 2:  # Ensure meaningful region size
                        regions.append((boundaries[i], boundaries[i + 1], 0, n))
                if len(regions) >= 2:
                    return regions
        
        # Try vertical splits (regions side by side)
        vertical_boundaries = self._find_vertical_boundaries(grid)
        if len(vertical_boundaries) >= 2:
            boundaries = [0] + vertical_boundaries + [n]
            boundaries = sorted(list(set(boundaries)))
            if len(boundaries) >= 3:
                regions = []
                for i in range(len(boundaries) - 1):
                    if boundaries[i+1] - boundaries[i] >= 2:
                        regions.append((0, n, boundaries[i], boundaries[i + 1]))
                if len(regions) >= 2:
                    return regions
        
        # Fallback: treat as single region
        return [(0, n, 0, n)]

    def _find_horizontal_boundaries(self, grid: np.ndarray) -> List[int]:
        """Find horizontal boundaries by analyzing color consistency."""
        n = grid.shape[0]
        boundaries = []
        
        # Look for rows where the dominant color changes significantly
        for r in range(1, n - 1):
            # Analyze a window around this row
            window_size = min(3, r, n - r)
            
            upper_region = grid[r-window_size:r, :]
            lower_region = grid[r:r+window_size, :]
            
            # Get most common colors in each region
            upper_colors, upper_counts = np.unique(upper_region, return_counts=True)
            lower_colors, lower_counts = np.unique(lower_region, return_counts=True)
            
            upper_dominant = upper_colors[np.argmax(upper_counts)]
            lower_dominant = lower_colors[np.argmax(lower_counts)]
            
            # Check if there's a significant difference
            if upper_dominant != lower_dominant:
                # Additional check: ensure the boundary is consistent across the width
                consistency = 0
                for c in range(grid.shape[1]):
                    upper_col = grid[max(0, r-window_size):r, c]
                    lower_col = grid[r:min(n, r+window_size), c]
                    
                    if len(upper_col) > 0 and len(lower_col) > 0:
                        upper_mode = Counter(upper_col).most_common(1)[0][0]
                        lower_mode = Counter(lower_col).most_common(1)[0][0]
                        if upper_mode != lower_mode:
                            consistency += 1
                
                # If majority of columns show the boundary, it's likely a real boundary
                if consistency > grid.shape[1] * 0.6:
                    boundaries.append(r)
        
        return boundaries

    def _find_vertical_boundaries(self, grid: np.ndarray) -> List[int]:
        """Find vertical boundaries by analyzing color consistency."""
        n = grid.shape[1]
        boundaries = []
        
        # Look for columns where the dominant color changes significantly
        for c in range(1, n - 1):
            # Analyze a window around this column
            window_size = min(3, c, n - c)
            
            left_region = grid[:, c-window_size:c]
            right_region = grid[:, c:c+window_size]
            
            # Get most common colors in each region
            left_colors, left_counts = np.unique(left_region, return_counts=True)
            right_colors, right_counts = np.unique(right_region, return_counts=True)
            
            left_dominant = left_colors[np.argmax(left_counts)]
            right_dominant = right_colors[np.argmax(right_counts)]
            
            # Check if there's a significant difference
            if left_dominant != right_dominant:
                # Additional check: ensure the boundary is consistent across the height
                consistency = 0
                for r in range(grid.shape[0]):
                    left_row = grid[r, max(0, c-window_size):c]
                    right_row = grid[r, c:min(n, c+window_size)]
                    
                    if len(left_row) > 0 and len(right_row) > 0:
                        left_mode = Counter(left_row).most_common(1)[0][0]
                        right_mode = Counter(right_row).most_common(1)[0][0]
                        if left_mode != right_mode:
                            consistency += 1
                
                # If majority of rows show the boundary, it's likely a real boundary
                if consistency > grid.shape[0] * 0.6:
                    boundaries.append(c)
        
        return boundaries


