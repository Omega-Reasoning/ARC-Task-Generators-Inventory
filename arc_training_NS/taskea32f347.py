from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskea32f347(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}.",
            "Each grid contains {vars['num_bars']} bars of the color {color('object_color')}, each with a distinct length of at least 2.",
            "The lengths of the bars vary between different input grids.",
            "The bars may have different positions and orientations within the grid.",
            "The bars are completely isolated—none of them touch or overlap with any other bar."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "Each bar is recolored based on its length relative to all other bars: the longest bar receives the first color, the second longest receives the second color, and so on, with the shortest bar receiving the last color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        num_bars = random.randint(2, 6)
        taskvars = {
            'n': random.randint(10, 30),
            'num_bars': num_bars,
            'object_color': random.randint(1, 9)
        }
        
        # Generate unique colors for each bar (ordered from longest to shortest)
        colors = random.sample(range(1, 10), num_bars)
        for i, color in enumerate(colors):
            taskvars[f'color_{i}'] = color
        
        # Ensure object_color is different from all bar colors
        while taskvars['object_color'] in colors:
            taskvars['object_color'] = random.randint(1, 9)
        
        # Create 3-6 training examples and 1 test example
        num_train = random.randint(3, 6)
        
        def generate_examples(count):
            examples = []
            for _ in range(count):
                input_grid = self.create_input(taskvars, {})
                output_grid = self.transform_input(input_grid, taskvars)
                examples.append({'input': input_grid, 'output': output_grid})
            return examples
        
        train_test_data = {
            'train': generate_examples(num_train),
            'test': generate_examples(1)
        }
        
        return taskvars, train_test_data
    
    def is_bar_isolated(self, grid: np.ndarray, bar_cells: List[Tuple[int, int]], object_color: int) -> bool:
        """Check if a bar doesn't touch any other bar."""
        n = grid.shape[0]
        for r, c in bar_cells:
            # Check all 8 neighbors (including diagonals)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n:
                        # If neighbor has object_color but is not part of our bar, bars are touching
                        if grid[nr, nc] == object_color and (nr, nc) not in bar_cells:
                            return False
        return True
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        num_bars = taskvars['num_bars']
        object_color = taskvars['object_color']
        
        def generate_grid():
            grid = np.zeros((n, n), dtype=int)
            
            # Generate bars with different lengths (minimum length of 2)
            max_bar_length = min(n - 4, max(3, n // 3))
            available_lengths = list(range(2, max_bar_length + 1))
            
            if len(available_lengths) < num_bars:
                available_lengths = list(range(2, num_bars + 2))
            
            lengths = random.sample(available_lengths, min(num_bars, len(available_lengths)))
            
            # If we couldn't sample enough distinct lengths, extend with duplicates
            while len(lengths) < num_bars:
                lengths.append(random.randint(2, max_bar_length))
            
            # Randomly decide orientation for each bar (horizontal or vertical)
            bars_placed = 0
            max_attempts = 500
            attempts = 0
            
            while bars_placed < num_bars and attempts < max_attempts:
                attempts += 1
                length = lengths[bars_placed]
                is_horizontal = random.choice([True, False])
                
                if is_horizontal:
                    # Place horizontal bar with margin
                    row = random.randint(1, n - 2)
                    max_col = n - length - 1
                    if max_col >= 1:
                        col = random.randint(1, max_col)
                        bar_cells = [(row, c) for c in range(col, col + length)]
                        
                        # Check if space is free and isolated
                        if (np.all(grid[row, col:col + length] == 0) and 
                            self.is_bar_isolated(grid, bar_cells, object_color)):
                            for r, c in bar_cells:
                                grid[r, c] = object_color
                            bars_placed += 1
                else:
                    # Place vertical bar with margin
                    col = random.randint(1, n - 2)
                    max_row = n - length - 1
                    if max_row >= 1:
                        row = random.randint(1, max_row)
                        bar_cells = [(r, col) for r in range(row, row + length)]
                        
                        # Check if space is free and isolated
                        if (np.all(grid[row:row + length, col] == 0) and 
                            self.is_bar_isolated(grid, bar_cells, object_color)):
                            for r, c in bar_cells:
                                grid[r, c] = object_color
                            bars_placed += 1
            
            if bars_placed < num_bars:
                raise ValueError(f"Could not place all {num_bars} bars in grid")
            
            return grid
        
        return generate_grid()
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        num_bars = taskvars['num_bars']
        
        # Find all connected objects (bars)
        objects = find_connected_objects(grid, background=0)
        
        if len(objects) > 0:
            # Sort objects by length (size) in descending order
            sorted_objects = sorted(objects.objects, key=lambda obj: len(obj), reverse=True)
            
            # Recolor bars based on their rank by length
            for rank, bar_obj in enumerate(sorted_objects[:num_bars]):
                color_key = f'color_{rank}'
                if color_key in taskvars:
                    bar_color = taskvars[color_key]
                    # Recolor all cells of this bar
                    for cell in bar_obj.cells:
                        r, c, _ = cell if len(cell) == 3 else (cell[0], cell[1], None)
                        output_grid[r, c] = bar_color
        
        return output_grid