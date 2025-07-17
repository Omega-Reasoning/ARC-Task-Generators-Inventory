from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import GridObject, GridObjects, find_connected_objects
from input_library import random_cell_coloring, retry

import numpy as np
import random
from typing import Dict, Any, Tuple, List
from collections import Counter

class Taske26a3af2(ARCTaskGenerator):
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
        
        # Try to detect regions more aggressively
        regions = self._identify_regions_robust(grid)
        
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

    def _identify_regions_robust(self, grid: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """More robust region detection using row/column analysis."""
        n = grid.shape[0]
        
        # Try horizontal splits first (regions stacked vertically)
        horizontal_boundaries = self._find_horizontal_boundaries_robust(grid)
        if len(horizontal_boundaries) > 0:
            boundaries = [0] + horizontal_boundaries + [n]
            boundaries = sorted(list(set(boundaries)))
            if len(boundaries) >= 3:  # At least 3 boundaries for 2+ regions
                regions = []
                for i in range(len(boundaries) - 1):
                    if boundaries[i+1] - boundaries[i] >= 2:
                        regions.append((boundaries[i], boundaries[i + 1], 0, n))
                if len(regions) >= 2:
                    return regions
        
        # Try vertical splits (regions side by side)
        vertical_boundaries = self._find_vertical_boundaries_robust(grid)
        if len(vertical_boundaries) > 0:
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

    def _find_horizontal_boundaries_robust(self, grid: np.ndarray) -> List[int]:
        """Find horizontal boundaries by analyzing dominant colors in each row."""
        n = grid.shape[0]
        boundaries = []
        
        # Get the dominant color for each row
        row_dominant_colors = []
        for r in range(n):
            row_colors = grid[r, :]
            colors, counts = np.unique(row_colors, return_counts=True)
            dominant_color = colors[np.argmax(counts)]
            row_dominant_colors.append(dominant_color)
        
        # Find boundaries where dominant color changes
        for r in range(1, n):
            if row_dominant_colors[r] != row_dominant_colors[r-1]:
                # Check if this is a consistent boundary across multiple rows
                # Look ahead to see if the new color persists
                if r < n - 1:
                    persistence = 0
                    for check_r in range(r, min(r + 3, n)):
                        if row_dominant_colors[check_r] == row_dominant_colors[r]:
                            persistence += 1
                    
                    if persistence >= 2:  # New color persists for at least 2 rows
                        boundaries.append(r)
        
        return boundaries

    def _find_vertical_boundaries_robust(self, grid: np.ndarray) -> List[int]:
        """Find vertical boundaries by analyzing dominant colors in each column."""
        n = grid.shape[1]
        boundaries = []
        
        # Get the dominant color for each column
        col_dominant_colors = []
        for c in range(n):
            col_colors = grid[:, c]
            colors, counts = np.unique(col_colors, return_counts=True)
            dominant_color = colors[np.argmax(counts)]
            col_dominant_colors.append(dominant_color)
        
        # Find boundaries where dominant color changes
        for c in range(1, n):
            if col_dominant_colors[c] != col_dominant_colors[c-1]:
                # Check if this is a consistent boundary across multiple columns
                # Look ahead to see if the new color persists
                if c < n - 1:
                    persistence = 0
                    for check_c in range(c, min(c + 3, n)):
                        if col_dominant_colors[check_c] == col_dominant_colors[c]:
                            persistence += 1
                    
                    if persistence >= 2:  # New color persists for at least 2 columns
                        boundaries.append(c)
        
        return boundaries

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