from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taske98196ab(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}, where {vars['n']} is an odd number.",
            "Each grid contains a central row (at index n // 2), consistently colored with {color('middle_color')}, effectively dividing the grid into top and bottom halves.",
            "The top half contains a random number of single-colored cells, all in a specific color that is different from the {color('middle_color')}.",
            "The bottom half contains a random number of single-colored cells, all in a color that is different from both the top half and the {color('middle_color')}.",
            "For each single-colored cell in one half, the same position in the other half is empty.",
            "Each half has at least one single-colored cell.",
            "The color of the both half and top half varies"
        ]
        
        transformation_reasoning_chain = [
            "Output grids are of size {vars['n']//2} x {vars['n']}.",
            "The central row in the input, which divides the grid into top and bottom halves, is identified. The two halves, along with their respective single-colored cells, are also identified.",
            "The output grid is the union of the top and bottom parts of the input grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        n = random.choice([5, 7, 9, 11, 13, 15, 17, 19, 21])  # odd numbers between 5-21
        
        # Choose middle color (consistent across all examples)
        available_colors = list(range(1, 10))  # exclude 0 (background)
        middle_color = random.choice(available_colors)
        
        taskvars = {
            'n': n,
            'middle_color': middle_color
        }
        
        # Generate train and test examples with varying colors
        num_train = random.randint(3, 6)
        train_examples = []
        test_examples = []
        
        # Keep track of used color combinations to ensure diversity
        used_combinations = set()
        
        for i in range(num_train):
            # Choose different colors for top and bottom half for each example
            remaining_colors = [c for c in available_colors if c != middle_color]
            top_color, bottom_color = random.sample(remaining_colors, 2)
            
            # Ensure we don't repeat the same combination
            combo = (top_color, bottom_color)
            if combo in used_combinations:
                # Try to find a different combination
                for _ in range(10):  # limited attempts
                    top_color, bottom_color = random.sample(remaining_colors, 2)
                    combo = (top_color, bottom_color)
                    if combo not in used_combinations:
                        break
            used_combinations.add(combo)
            
            gridvars = {'top_color': top_color, 'bottom_color': bottom_color}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example with different colors
        remaining_colors = [c for c in available_colors if c != middle_color]
        top_color, bottom_color = random.sample(remaining_colors, 2)
        # Try to use a new combination for test
        combo = (top_color, bottom_color)
        if combo in used_combinations:
            for _ in range(10):
                top_color, bottom_color = random.sample(remaining_colors, 2)
                combo = (top_color, bottom_color)
                if combo not in used_combinations:
                    break
                    
        gridvars = {'top_color': top_color, 'bottom_color': bottom_color}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        test_examples.append({'input': input_grid, 'output': output_grid})
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        middle_color = taskvars['middle_color']
        top_color = gridvars['top_color']
        bottom_color = gridvars['bottom_color']
        
        # Create grid filled with background (0)
        grid = np.zeros((n, n), dtype=int)
        
        # Fill central row with middle_color
        central_row = n // 2
        grid[central_row, :] = middle_color
        
        # Calculate minimum cells needed to satisfy 25% constraint
        total_colorable_cells = n * (n - 1)  # exclude central row
        min_colored_cells = int(0.25 * total_colorable_cells)
        
        # Determine how many cells to color in each half
        half_size = n // 2
        
        # Ensure at least one cell per half and meet 25% constraint
        min_per_half = max(1, min_colored_cells // 4)  # conservative estimate
        max_per_half = half_size * n // 2  # don't fill more than half of available positions
        
        top_cells_to_color = random.randint(min_per_half, min(max_per_half, half_size * n // 2))
        bottom_cells_to_color = random.randint(min_per_half, min(max_per_half, half_size * n // 2))
        
        # Make sure we meet the 25% constraint overall
        total_colored = top_cells_to_color + bottom_cells_to_color
        if total_colored < min_colored_cells:
            # Add more cells to reach minimum
            additional_needed = min_colored_cells - total_colored
            if random.choice([True, False]):
                top_cells_to_color += additional_needed
            else:
                bottom_cells_to_color += additional_needed
        
        # Create complementary pattern
        # First, select positions for top half
        top_positions = [(r, c) for r in range(half_size) for c in range(n)]
        top_selected = random.sample(top_positions, min(top_cells_to_color, len(top_positions)))
        
        # Color the selected top positions
        for r, c in top_selected:
            grid[r, c] = top_color
        
        # For bottom half, select positions that don't correspond to filled top positions
        bottom_positions = []
        for r in range(central_row + 1, n):
            for c in range(n):
                # Map this bottom position to corresponding top position
                corresponding_top_row = r - (central_row + 1)
                if corresponding_top_row < half_size:
                    # Check if corresponding top position is filled
                    if (corresponding_top_row, c) not in top_selected:
                        bottom_positions.append((r, c))
                else:
                    bottom_positions.append((r, c))
        
        # Select bottom positions to color
        bottom_selected = random.sample(
            bottom_positions, 
            min(bottom_cells_to_color, len(bottom_positions))
        )
        
        # Color the selected bottom positions
        for r, c in bottom_selected:
            grid[r, c] = bottom_color
            
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        central_row = n // 2
        
        # Extract top and bottom halves (excluding central row)
        top_half = grid[:central_row, :]
        bottom_half = grid[central_row + 1:, :]
        
        # Create output grid of size (n//2) x n
        output_height = n // 2
        output_grid = np.zeros((output_height, n), dtype=int)
        
        # Union of top and bottom halves
        for r in range(output_height):
            for c in range(n):
                top_val = top_half[r, c] if r < top_half.shape[0] else 0
                bottom_val = bottom_half[r, c] if r < bottom_half.shape[0] else 0
                
                # Take union - any non-zero value
                if top_val != 0:
                    output_grid[r, c] = top_val
                elif bottom_val != 0:
                    output_grid[r, c] = bottom_val
                # else remains 0 (background)
        
        return output_grid

