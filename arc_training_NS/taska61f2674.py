from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class taska61f2674(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Each input grid is of size {vars['n']} × {vars['n']}.",
            "Each input grid contains a random number of vertical bars filled with color {color('fill_color')}, and all bars are aligned to either odd-indexed or even-indexed columns.",
            "Each input grid contains at least 3 bars.",
            "In each input grid heights of the bars vary across the grid, and no two bars share the same height."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "The tallest bar is recolored with {color('color_1')}.",
            "The shortest bar is recolored with {color('color_2')}.",
            "All remaining bars are replaced with empty cells (0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        fill_color = taskvars['fill_color']
        
        grid = np.zeros((n, n), dtype=int)
        
        # Determine which columns to use (odd or even indexed)
        # We'll randomly choose alignment for each grid
        even_columns = random.choice([True, False])
        if even_columns:
            available_columns = [i for i in range(0, n, 2)]  # 0, 2, 4, ...
        else:
            available_columns = [i for i in range(1, n, 2)]  # 1, 3, 5, ...
        
        # Ensure we have at least 3 bars, but no more than the available columns or grid height
        max_bars = min(len(available_columns), n)  # Can't have more bars than available columns or grid height
        num_bars = random.randint(3, min(6, max_bars))
        selected_columns = random.sample(available_columns, num_bars)
        
        # Create unique heights for each bar (no two bars share the same height)
        available_heights = list(range(1, n + 1))  # Heights from 1 to n
        selected_heights = random.sample(available_heights, num_bars)
        
        # Create bars with unique heights
        for col, bar_height in zip(selected_columns, selected_heights):
            # Place bar at bottom of grid (aligned to bottom)
            start_row = n - bar_height
            for row in range(start_row, n):
                grid[row, col] = fill_color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output = grid.copy()
        fill_color = taskvars['fill_color']
        color_1 = taskvars['color_1']  # tallest bar color
        color_2 = taskvars['color_2']  # shortest bar color
        
        # Find all vertical bars
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        bars = objects.with_color(fill_color)
        
        if len(bars) == 0:
            return output
        
        # Calculate heights of bars by counting cells in each bar
        bar_heights = []
        for bar in bars:
            height = len(bar.cells)
            bar_heights.append((bar, height))
        
        # Since all bars have unique heights, we can directly find tallest and shortest
        tallest_bar = max(bar_heights, key=lambda x: x[1])[0]
        shortest_bar = min(bar_heights, key=lambda x: x[1])[0]
        
        # Clear all bars first
        for bar in bars:
            bar.cut(output, background=0)
        
        # Recolor tallest bar with color_1
        new_cells = set()
        for r, c, _ in tallest_bar.cells:
            new_cells.add((r, c, color_1))
        tallest_bar.cells = new_cells
        tallest_bar.paste(output)
        
        # Recolor shortest bar with color_2
        new_cells = set()
        for r, c, _ in shortest_bar.cells:
            new_cells.add((r, c, color_2))
        shortest_bar.cells = new_cells
        shortest_bar.paste(output)
        
        return output

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize only the task variables mentioned in the reasoning chains
        taskvars = {
            'n': random.randint(8, 30),  # Grid size (n x n)
            'fill_color': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            'color_1': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),  # tallest bar color
            'color_2': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),  # shortest bar color
        }
        
        # Ensure colors are different
        while taskvars['color_1'] == taskvars['fill_color']:
            taskvars['color_1'] = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        while taskvars['color_2'] == taskvars['fill_color'] or taskvars['color_2'] == taskvars['color_1']:
            taskvars['color_2'] = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        # Generate training examples
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            
            # Ensure the transformation is meaningful (at least 3 bars with unique heights)
            bars = find_connected_objects(input_grid, diagonal_connectivity=False, background=0).with_color(taskvars['fill_color'])
            if len(bars) >= 3:
                # Verify all bars have unique heights
                heights = [len(bar.cells) for bar in bars]
                if len(set(heights)) == len(heights):  # All heights are unique
                    train_examples.append({
                        'input': input_grid,
                        'output': output_grid
                    })
        
        # If we don't have enough good examples, generate more
        while len(train_examples) < num_train:
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            bars = find_connected_objects(input_grid, diagonal_connectivity=False, background=0).with_color(taskvars['fill_color'])
            if len(bars) >= 3:
                heights = [len(bar.cells) for bar in bars]
                if len(set(heights)) == len(heights):  # All heights are unique
                    train_examples.append({
                        'input': input_grid,
                        'output': output_grid
                    })
        
        # Generate test example - ensure it has at least 3 bars with unique heights
        while True:
            test_input = self.create_input(taskvars, {})
            bars = find_connected_objects(test_input, diagonal_connectivity=False, background=0).with_color(taskvars['fill_color'])
            if len(bars) >= 3:
                heights = [len(bar.cells) for bar in bars]
                if len(set(heights)) == len(heights):  # All heights are unique
                    break
            
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data

