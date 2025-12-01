from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple

class Taskba97ae07Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} × {vars['m']}.",
            "The grid contains an overall cross shape, formed by a vertical bar and a horizontal bar, positioned mostly in the central area of the grid.",
            "The vertical bar consists of a random number of adjacent columns that span the grid from the top row to the bottom row.",
            "The horizontal bar consists of a random number of adjacent rows that span the grid from the leftmost column to the rightmost column.",
            "These two bars intersect, with one bar visually passing over the other.",
            "In each input grid, the horizontal and vertical bars are colored with two distinct colors.",
            "All remaining cells are empty (0).",
            "The colors of the bars vary across input grids to maintain diversity."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "At the intersection, the relative order of the bars is inverted.",
            "If the horizontal bar passes over the vertical bar in the input, then in the output the vertical bar passes over the horizontal bar.",
            "Conversely, if the vertical bar passes over the horizontal bar in the input, then in the output the horizontal bar passes over the vertical bar."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n, m = taskvars['n'], taskvars['m']
        
        # Get colors for this specific grid
        horizontal_color = gridvars['horizontal_color']
        vertical_color = gridvars['vertical_color']
        horizontal_on_top = gridvars['horizontal_on_top']
        
        # Create empty grid
        grid = np.zeros((n, m), dtype=int)
        
        # Create vertical bar (random width, positioned in central area)
        vertical_width = random.randint(1, min(3, m // 3))
        # Center the vertical bar with some randomness
        center_col = m // 2
        max_offset = min(2, m // 4)
        offset = random.randint(-max_offset, max_offset)
        vertical_start = max(0, min(m - vertical_width, center_col - vertical_width // 2 + offset))
        
        # Create horizontal bar (random height, positioned in central area)
        horizontal_height = random.randint(1, min(3, n // 3))
        # Center the horizontal bar with some randomness
        center_row = n // 2
        max_offset = min(2, n // 4)
        offset = random.randint(-max_offset, max_offset)
        horizontal_start = max(0, min(n - horizontal_height, center_row - horizontal_height // 2 + offset))
        
        # Draw the bar that goes underneath first
        if horizontal_on_top:
            # Draw vertical bar first (underneath)
            for col in range(vertical_start, vertical_start + vertical_width):
                grid[:, col] = vertical_color
            
            # Draw horizontal bar on top
            for row in range(horizontal_start, horizontal_start + horizontal_height):
                grid[row, :] = horizontal_color
        else:
            # Draw horizontal bar first (underneath)
            for row in range(horizontal_start, horizontal_start + horizontal_height):
                grid[row, :] = horizontal_color
            
            # Draw vertical bar on top
            for col in range(vertical_start, vertical_start + vertical_width):
                grid[:, col] = vertical_color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Copy the input grid
        output = grid.copy()
        
        # Find all non-zero colors in the grid
        colors = set(np.unique(grid)) - {0}
        
        if len(colors) != 2:
            return output  # Should have exactly 2 colors
        
        color1, color2 = list(colors)
        
        # Find intersection region (where both colors would be present if we overlay)
        # We need to reconstruct what the bars look like
        
        # Find rows that span the full width - these are horizontal bar rows
        horizontal_rows = []
        for r in range(grid.shape[0]):
            if np.all(grid[r, :] != 0):  # Row is completely filled
                horizontal_rows.append(r)
        
        # Find columns that span the full height - these are vertical bar columns
        vertical_cols = []
        for c in range(grid.shape[1]):
            if np.all(grid[:, c] != 0):  # Column is completely filled
                vertical_cols.append(c)
        
        # At the intersection, swap the colors
        for r in horizontal_rows:
            for c in vertical_cols:
                current_color = grid[r, c]
                # Swap to the other color
                new_color = color2 if current_color == color1 else color1
                output[r, c] = new_color
        
        return output

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        n = random.randint(8, 20)  # Grid height (increased minimum to accommodate central positioning)
        m = random.randint(8, 20)  # Grid width (increased minimum to accommodate central positioning)
        
        taskvars = {
            'n': n,
            'm': m
        }
        
        # Generate training examples
        num_train = random.randint(3, 6)
        train_examples = []
        
        # Keep track of used color combinations to ensure diversity
        used_color_pairs = set()
        available_colors = list(range(1, 10))
        
        for _ in range(num_train):
            # Generate unique color pair
            while True:
                colors = random.sample(available_colors, 2)
                color_pair = tuple(sorted(colors))
                if color_pair not in used_color_pairs:
                    used_color_pairs.add(color_pair)
                    break
                if len(used_color_pairs) >= len(available_colors) * (len(available_colors) - 1) // 2:
                    # If we've used all combinations, allow reuse
                    break
            
            horizontal_color, vertical_color = colors
            horizontal_on_top = random.choice([True, False])
            
            gridvars = {
                'horizontal_color': horizontal_color,
                'vertical_color': vertical_color,
                'horizontal_on_top': horizontal_on_top
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        while True:
            colors = random.sample(available_colors, 2)
            color_pair = tuple(sorted(colors))
            if color_pair not in used_color_pairs or len(used_color_pairs) < 3:
                break
        
        horizontal_color, vertical_color = colors
        horizontal_on_top = random.choice([True, False])
        
        test_gridvars = {
            'horizontal_color': horizontal_color,
            'vertical_color': vertical_color,
            'horizontal_on_top': horizontal_on_top
        }
        
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data

