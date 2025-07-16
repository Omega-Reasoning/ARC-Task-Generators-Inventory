from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskde1cd16c(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "Each input grid is split into 3 or 4 rectangular regions.",
            "Each region is a non-overlapping, contiguous rectangle within the grid.",
            "Each input grid is divided into 3 or 4 rectangular, non-overlapping regions.",
            "Each region is filled with a solid base color, chosen randomly, or left empty (with value 0). At most one region in an input grid may be empty.",
            "A small number of cells in each region may be marked with {color('noise_color')}.",
            "The number of {color('noise_color')} cells in each region is chosen randomly and can be zero.",
            "At least one region always contains some {color('noise_color')} cells (not all regions are empty of noise).",
            "In each input grid, no two regions have the same number of {color('noise_color')} cells.",
            "In each region, the number of {color('noise_color')} cells is at most 10% of the total cell count of that region.",
            "These {color('noise_color')} cells are sparsely and randomly scattered inside each region.",
            "These {color('noise_color')} cells are non-adjacent."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is of size 1x1.",
            "The base regions (3 or 4) in the input grid are identified, along with the base color assigned to each region.",
            "For each region, the number of cells marked with {color('noise_color')} is counted.",
            "The output cell takes the base color of the region that contains the highest number of {color('noise_color')} cells."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        taskvars = {
            'noise_color': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        }
        
        # Create training and test examples
        num_train = random.randint(3, 6)
        num_test = 1
        
        train_examples = []
        for _ in range(num_train):
            # Store region info for consistent transform
            input_grid, region_info = self.create_input_with_regions(taskvars, {})
            output_grid = self.transform_input_with_regions(input_grid, taskvars, region_info)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        test_examples = []
        for _ in range(num_test):
            input_grid, region_info = self.create_input_with_regions(taskvars, {})
            output_grid = self.transform_input_with_regions(input_grid, taskvars, region_info)
            test_examples.append({'input': input_grid, 'output': output_grid})
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        # This is called by the framework, so we need to implement it
        input_grid, _ = self.create_input_with_regions(taskvars, gridvars)
        return input_grid
    
    def create_input_with_regions(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        noise_color = taskvars['noise_color']
        
        # Random grid size
        height = random.randint(5, 30)
        width = random.randint(5, 30)
        
        # Number of regions (3 or 4)
        num_regions = random.choice([3, 4])
        
        # Create regions by splitting the grid
        regions, split_info = self._create_regions_with_info(height, width, num_regions)
        
        # Initialize grid with background
        grid = np.zeros((height, width), dtype=int)
        
        # Available colors (excluding noise color and background)
        available_colors = [c for c in range(1, 10) if c != noise_color]
        
        # Assign base colors to regions (at most one can be empty)
        region_colors = []
        empty_region_idx = random.randint(0, num_regions - 1) if random.random() < 0.3 else -1
        
        for i in range(num_regions):
            if i == empty_region_idx:
                region_colors.append(0)  # Empty region
            else:
                color = random.choice(available_colors)
                available_colors.remove(color)  # Ensure unique colors
                region_colors.append(color)
        
        # Fill regions with base colors
        for i, (region, base_color) in enumerate(zip(regions, region_colors)):
            if base_color != 0:
                for r, c in region:
                    grid[r, c] = base_color
        
        # Add noise cells ensuring constraints are met
        self._add_noise_cells(grid, regions, region_colors, noise_color)
        
        # Store region info for transform
        region_info = {
            'regions': regions,
            'region_colors': region_colors,
            'split_info': split_info
        }
        
        return grid, region_info
    
    def _create_regions_with_info(self, height: int, width: int, num_regions: int) -> Tuple[List[List[Tuple[int, int]]], Dict[str, Any]]:
        """Create non-overlapping rectangular regions and return split info."""
        regions = []
        split_info = {'num_regions': num_regions}
        
        if num_regions == 3:
            # Split into 3 regions - various configurations
            split_type = random.choice(['horizontal', 'vertical', 'mixed'])
            split_info['split_type'] = split_type
            
            if split_type == 'horizontal':
                # Three horizontal strips
                h1 = height // 3
                h2 = 2 * height // 3
                split_info['h1'] = h1
                split_info['h2'] = h2
                regions.append([(r, c) for r in range(h1) for c in range(width)])
                regions.append([(r, c) for r in range(h1, h2) for c in range(width)])
                regions.append([(r, c) for r in range(h2, height) for c in range(width)])
            elif split_type == 'vertical':
                # Three vertical strips
                w1 = width // 3
                w2 = 2 * width // 3
                split_info['w1'] = w1
                split_info['w2'] = w2
                regions.append([(r, c) for r in range(height) for c in range(w1)])
                regions.append([(r, c) for r in range(height) for c in range(w1, w2)])
                regions.append([(r, c) for r in range(height) for c in range(w2, width)])
            else:  # mixed
                # One large region and two smaller ones
                h_mid = height // 2
                w_mid = width // 2
                split_info['h_mid'] = h_mid
                split_info['w_mid'] = w_mid
                regions.append([(r, c) for r in range(h_mid) for c in range(width)])
                regions.append([(r, c) for r in range(h_mid, height) for c in range(w_mid)])
                regions.append([(r, c) for r in range(h_mid, height) for c in range(w_mid, width)])
        
        else:  # num_regions == 4
            # Split into 4 quadrants
            h_mid = height // 2
            w_mid = width // 2
            split_info['split_type'] = 'quadrants'
            split_info['h_mid'] = h_mid
            split_info['w_mid'] = w_mid
            regions.append([(r, c) for r in range(h_mid) for c in range(w_mid)])
            regions.append([(r, c) for r in range(h_mid) for c in range(w_mid, width)])
            regions.append([(r, c) for r in range(h_mid, height) for c in range(w_mid)])
            regions.append([(r, c) for r in range(h_mid, height) for c in range(w_mid, width)])
        
        return regions, split_info
    
    def _create_regions(self, height: int, width: int, num_regions: int) -> List[List[Tuple[int, int]]]:
        """Create non-overlapping rectangular regions."""
        regions, _ = self._create_regions_with_info(height, width, num_regions)
        return regions
    
    def _get_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Get 4-connected neighbors (cardinal directions only)."""
        return [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
    
    def _is_valid_noise_placement(self, region: List[Tuple[int, int]], selected_cells: List[Tuple[int, int]], 
                                 new_cell: Tuple[int, int]) -> bool:
        """Check if placing noise at new_cell would violate non-adjacency constraint."""
        for existing_cell in selected_cells:
            if new_cell in self._get_neighbors(existing_cell[0], existing_cell[1]):
                return False
        return True
    
    def _add_noise_cells(self, grid: np.ndarray, regions: List[List[Tuple[int, int]]], 
                        region_colors: List[int], noise_color: int):
        """Add noise cells to regions with required constraints including non-adjacency."""
        num_regions = len(regions)
        
        # Calculate max noise cells per region (10% constraint)
        max_noise_per_region = []
        for region in regions:
            max_allowed = max(1, len(region) * 10 // 100)  # At least 1, at most 10%
            max_noise_per_region.append(max_allowed)
        
        # Generate unique noise counts for each region
        noise_counts = []
        used_counts = set()
        
        # Ensure at least one region has noise
        for i in range(num_regions):
            if i == 0:
                # First region must have at least 1 noise cell
                count = random.randint(1, max_noise_per_region[i])
            else:
                # Other regions can have 0 or more, but must be unique
                max_count = max_noise_per_region[i]
                available_counts = [c for c in range(0, max_count + 1) if c not in used_counts]
                if available_counts:
                    count = random.choice(available_counts)
                else:
                    count = 0
            
            noise_counts.append(count)
            used_counts.add(count)
        
        # If no region has noise, force the first region to have noise
        if all(count == 0 for count in noise_counts):
            noise_counts[0] = 1
        
        # Add noise cells to each region ensuring non-adjacency
        for region, noise_count in zip(regions, noise_counts):
            if noise_count > 0:
                selected_cells = []
                available_cells = region.copy()
                
                # Try to place noise cells with non-adjacency constraint
                max_attempts = 100
                for _ in range(noise_count):
                    attempts = 0
                    placed = False
                    
                    while attempts < max_attempts and not placed and available_cells:
                        candidate = random.choice(available_cells)
                        
                        if self._is_valid_noise_placement(region, selected_cells, candidate):
                            selected_cells.append(candidate)
                            placed = True
                        
                        available_cells.remove(candidate)
                        attempts += 1
                    
                    if not placed:
                        # If we can't place more non-adjacent cells, stop trying
                        break
                
                # Place the selected noise cells
                for r, c in selected_cells:
                    grid[r, c] = noise_color
    
    def transform_input_with_regions(self, grid: np.ndarray, taskvars: Dict[str, Any], region_info: Dict[str, Any]) -> np.ndarray:
        """Transform input using the stored region information."""
        noise_color = taskvars['noise_color']
        regions = region_info['regions']
        
        # For each region, count noise cells and identify base color
        region_data = []
        for region in regions:
            noise_count = 0
            base_color = 0
            
            # First pass: identify the base color (most common non-noise color)
            color_counts = {}
            for r, c in region:
                cell_color = grid[r, c]
                if cell_color != noise_color:
                    color_counts[cell_color] = color_counts.get(cell_color, 0) + 1
            
            # Base color is the most frequent non-noise color
            if color_counts:
                base_color = max(color_counts, key=color_counts.get)
            
            # Second pass: count noise cells
            for r, c in region:
                if grid[r, c] == noise_color:
                    noise_count += 1
            
            region_data.append((noise_count, base_color))
        
        # Find the region with the highest number of noise cells
        max_noise_count = -1
        winning_base_color = 0
        
        for noise_count, base_color in region_data:
            if noise_count > max_noise_count:
                max_noise_count = noise_count
                winning_base_color = base_color
        
        # Return 1x1 grid with winning base color
        return np.array([[winning_base_color]])
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform input - this is called by the framework but we need region info."""
        # This method is called by the framework but doesn't have region info
        # We need to reconstruct the regions from the grid
        noise_color = taskvars['noise_color']
        height, width = grid.shape
        
        # Try different region configurations to find the best fit
        best_config = None
        best_score = -1
        
        for num_regions in [3, 4]:
            for split_type in (['horizontal', 'vertical', 'mixed'] if num_regions == 3 else ['quadrants']):
                regions = self._reconstruct_regions(height, width, num_regions, split_type)
                score = self._evaluate_region_fit(grid, regions, noise_color)
                
                if score > best_score:
                    best_score = score
                    best_config = regions
        
        if best_config is None:
            return np.array([[0]])
        
        # Use the best configuration
        regions = best_config
        
        # For each region, count noise cells and identify base color
        region_data = []
        for region in regions:
            noise_count = 0
            base_color = 0
            
            # First pass: identify the base color (most common non-noise color)
            color_counts = {}
            for r, c in region:
                cell_color = grid[r, c]
                if cell_color != noise_color:
                    color_counts[cell_color] = color_counts.get(cell_color, 0) + 1
            
            # Base color is the most frequent non-noise color
            if color_counts:
                base_color = max(color_counts, key=color_counts.get)
            
            # Second pass: count noise cells
            for r, c in region:
                if grid[r, c] == noise_color:
                    noise_count += 1
            
            region_data.append((noise_count, base_color))
        
        # Find the region with the highest number of noise cells
        max_noise_count = -1
        winning_base_color = 0
        
        for noise_count, base_color in region_data:
            if noise_count > max_noise_count:
                max_noise_count = noise_count
                winning_base_color = base_color
        
        # Return 1x1 grid with winning base color
        return np.array([[winning_base_color]])
    
    def _reconstruct_regions(self, height: int, width: int, num_regions: int, split_type: str) -> List[List[Tuple[int, int]]]:
        """Reconstruct regions based on grid dimensions and split type."""
        regions = []
        
        if num_regions == 3:
            if split_type == 'horizontal':
                h1 = height // 3
                h2 = 2 * height // 3
                regions.append([(r, c) for r in range(h1) for c in range(width)])
                regions.append([(r, c) for r in range(h1, h2) for c in range(width)])
                regions.append([(r, c) for r in range(h2, height) for c in range(width)])
            elif split_type == 'vertical':
                w1 = width // 3
                w2 = 2 * width // 3
                regions.append([(r, c) for r in range(height) for c in range(w1)])
                regions.append([(r, c) for r in range(height) for c in range(w1, w2)])
                regions.append([(r, c) for r in range(height) for c in range(w2, width)])
            else:  # mixed
                h_mid = height // 2
                w_mid = width // 2
                regions.append([(r, c) for r in range(h_mid) for c in range(width)])
                regions.append([(r, c) for r in range(h_mid, height) for c in range(w_mid)])
                regions.append([(r, c) for r in range(h_mid, height) for c in range(w_mid, width)])
        else:  # num_regions == 4
            h_mid = height // 2
            w_mid = width // 2
            regions.append([(r, c) for r in range(h_mid) for c in range(w_mid)])
            regions.append([(r, c) for r in range(h_mid) for c in range(w_mid, width)])
            regions.append([(r, c) for r in range(h_mid, height) for c in range(w_mid)])
            regions.append([(r, c) for r in range(h_mid, height) for c in range(w_mid, width)])
        
        return regions
    
    def _evaluate_region_fit(self, grid: np.ndarray, regions: List[List[Tuple[int, int]]], noise_color: int) -> float:
        """Evaluate how well a region configuration fits the grid."""
        score = 0
        
        for region in regions:
            # Count colors in this region
            color_counts = {}
            for r, c in region:
                cell_color = grid[r, c]
                color_counts[cell_color] = color_counts.get(cell_color, 0) + 1
            
            # Remove noise color from consideration
            if noise_color in color_counts:
                del color_counts[noise_color]
            
            # Good regions should have one dominant color
            if color_counts:
                total_cells = sum(color_counts.values())
                max_color_count = max(color_counts.values())
                # Score based on how dominant the main color is
                score += max_color_count / total_cells
        
        return score

