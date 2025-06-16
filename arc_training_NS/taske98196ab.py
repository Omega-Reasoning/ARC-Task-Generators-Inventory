from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import random_cell_coloring, retry
import numpy as np
import random
from typing import Dict, Any, Tuple

class Taske98196ab(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}, where {vars['n']} is an odd number.",
            "Each grid contains a central row (at index n // 2), consistently colored with {color('middle_color')}, effectively dividing the grid into top and bottom halves.",
            "The top half contains a random number of colored cells in {color('top_color')}.",
            "The bottom half contains a random number of colored cells in {color('bottom_color')}.",
            "For each colored cell in one half, the same position in the other half is empty.",
            "Each half has at least one colored cell."
        ]
        
        transformation_reasoning_chain = [
            "Input grids are of size {vars['n']//2} x {vars['n']}.",
            "The central row in the input, which divides the grid into top and bottom halves, is identified. The two halves, along with their respective colored cells, are also identified.",
            "The output grid is the union of the top and bottom halves of the input grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        middle_color = taskvars['middle_color']
        top_color = taskvars['top_color']
        bottom_color = taskvars['bottom_color']
        
        def generate_valid_grid():
            # Create grid with background (0)
            grid = np.zeros((n, n), dtype=int)
            
            # Fill central row
            central_row = n // 2
            grid[central_row, :] = middle_color
            
            # Calculate total cells in grid
            total_cells = n * n
            
            # Target approximately 25% of all cells to be colored
            # This includes the central row + cells in top/bottom halves
            target_colored = int(0.25 * total_cells)
            
            # Central row already has n colored cells
            central_colored = n
            
            # Remaining cells to color in top and bottom halves
            remaining_to_color = max(2, target_colored - central_colored)  # At least 2 (one per half)
            
            # Add some randomness around the target (±20%)
            variation = int(0.2 * remaining_to_color)
            min_colored = max(2, remaining_to_color - variation)
            max_colored = remaining_to_color + variation
            
            # Calculate half dimensions (excluding central row)
            half_height = central_row  # Height of each half
            half_size = half_height * n  # Total cells in each half
            
            # Make sure we don't exceed the available positions
            max_possible = min(half_size, max_colored)
            num_colored = random.randint(min_colored, max_possible)
            
            # Generate all possible positions within one half (we'll use these as templates)
            half_positions = [(r, c) for r in range(half_height) for c in range(n)]
            
            # Randomly select positions for coloring
            selected_positions = random.sample(half_positions, min(num_colored, len(half_positions)))
            
            # Randomly decide which positions go to top vs bottom half
            # Ensure at least one in each half
            random.shuffle(selected_positions)
            split_point = random.randint(1, len(selected_positions) - 1)
            
            top_positions = selected_positions[:split_point]
            bottom_positions = selected_positions[split_point:]
            
            # Color the top half positions
            for r, c in top_positions:
                grid[r, c] = top_color
                # Ensure corresponding position in bottom half is empty (it already is)
            
            # Color the bottom half positions  
            for r, c in bottom_positions:
                # Map to bottom half (after central row)
                bottom_r = central_row + 1 + r
                grid[bottom_r, c] = bottom_color
                # Ensure corresponding position in top half is empty (it already is)
            
            return grid
        
        def is_valid_grid(grid):
            central_row = n // 2
            
            # Check that each half has at least one colored cell
            top_half = grid[:central_row, :]
            bottom_half = grid[central_row + 1:, :]
            
            top_colored = np.sum((top_half == top_color).astype(int))
            bottom_colored = np.sum((bottom_half == bottom_color).astype(int))
            
            # Check approximately 25% coloring constraint
            total_colored = top_colored + bottom_colored + n  # Include central row
            total_cells = n * n
            colored_percentage = total_colored / total_cells
            
            # Allow range from 25% to 35% to be "approximately 30%"
            valid_percentage = 0.25 <= colored_percentage <= 0.35
            
            # Verify complementary positioning constraint
            valid_positioning = True
            for r in range(central_row):
                for c in range(n):
                    top_cell = top_half[r, c]
                    bottom_cell = bottom_half[r, c]  # Same relative position
                    
                    # If one position has color, the other must be empty (0)
                    if top_cell == top_color and bottom_cell != 0:
                        valid_positioning = False
                    if bottom_cell == bottom_color and top_cell != 0:
                        valid_positioning = False
            
            return (top_colored >= 1 and 
                   bottom_colored >= 1 and 
                   valid_percentage and
                   valid_positioning)
        
        return retry(generate_valid_grid, is_valid_grid)
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        central_row = n // 2
        
        # Extract top and bottom halves
        top_half = grid[:central_row, :]
        bottom_half = grid[central_row + 1:, :]
        
        # Create output grid (half the height)
        output_height = central_row
        output = np.zeros((output_height, n), dtype=int)
        
        # Union operation: combine top and bottom halves
        # Since the halves have complementary positioning, union just overlays them
        for r in range(output_height):
            for c in range(n):
                top_val = top_half[r, c]
                bottom_val = bottom_half[r, c] if r < bottom_half.shape[0] else 0
                
                # Union: take the non-zero value (only one should be non-zero due to constraint)
                if top_val != 0:
                    output[r, c] = top_val
                elif bottom_val != 0:
                    output[r, c] = bottom_val
                else:
                    output[r, c] = 0
        
        return output
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        n = random.choice([5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29])  # Odd numbers between 5 and 30
        
        # Choose different colors
        all_colors = list(range(1, 10))  # Colors 1-9 (excluding 0 which is background)
        selected_colors = random.sample(all_colors, 3)
        
        taskvars = {
            'n': n,
            'middle_color': selected_colors[0],
            'top_color': selected_colors[1], 
            'bottom_color': selected_colors[2]
        }
        
        # Generate training examples
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data
