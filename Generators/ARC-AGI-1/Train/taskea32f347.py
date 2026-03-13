from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObjects
from Framework.input_library import retry
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
        n = random.randint(10, 30)
        
        max_bar_length = min(n // 2, max(4, n // 3))
        available_lengths = list(range(2, max_bar_length + 1))
        max_possible_bars = len(available_lengths)
        
        if n <= 12:
            num_bars = random.randint(2, min(3, max_possible_bars))
        elif n <= 18:
            num_bars = random.randint(2, min(4, max_possible_bars))
        else:
            num_bars = random.randint(3, min(6, max_possible_bars))
        
        # Always sample 6 colors regardless of num_bars
        colors = random.sample(range(1, 10), 6)

        taskvars = {
            'n': n,
            'num_bars': num_bars,
            'object_color': random.randint(1, 9),
            'color_0': colors[0],
            'color_1': colors[1],
            'color_2': colors[2],
            'color_3': colors[3],
            'color_4': colors[4],
            'color_5': colors[5],
        }
        
        while taskvars['object_color'] in colors:
            taskvars['object_color'] = random.randint(1, 9)
        
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
        n = grid.shape[0]
        for r, c in bar_cells:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n:
                        if grid[nr, nc] == object_color and (nr, nc) not in bar_cells:
                            return False
        return True
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        num_bars = taskvars['num_bars']
        object_color = taskvars['object_color']
        
        for attempt in range(20):
            try:
                grid = np.zeros((n, n), dtype=int)
                
                max_bar_length = min(n // 2, max(4, n // 3))
                available_lengths = list(range(2, max_bar_length + 1))
                
                if len(available_lengths) < num_bars:
                    raise ValueError(f"Not enough distinct lengths ({len(available_lengths)}) for {num_bars} bars")
                
                lengths = random.sample(available_lengths, num_bars)
                bars_placed = 0
                max_attempts = 1000
                attempts = 0
                
                while bars_placed < num_bars and attempts < max_attempts:
                    attempts += 1
                    length = lengths[bars_placed]
                    is_horizontal = random.choice([True, False])
                    margin = max(1, n // 15)
                    
                    if is_horizontal:
                        if n - length - 2 * margin < 0:
                            continue
                        row = random.randint(margin, n - margin - 1)
                        col = random.randint(margin, n - length - margin)
                        bar_cells = [(row, c) for c in range(col, col + length)]
                        
                        if (np.all(grid[row, col:col + length] == 0) and 
                            self.is_bar_isolated(grid, bar_cells, object_color)):
                            for r, c in bar_cells:
                                grid[r, c] = object_color
                            bars_placed += 1
                    else:
                        if n - length - 2 * margin < 0:
                            continue
                        col = random.randint(margin, n - margin - 1)
                        row = random.randint(margin, n - length - margin)
                        bar_cells = [(r, col) for r in range(row, row + length)]
                        
                        if (np.all(grid[row:row + length, col] == 0) and 
                            self.is_bar_isolated(grid, bar_cells, object_color)):
                            for r, c in bar_cells:
                                grid[r, c] = object_color
                            bars_placed += 1
                
                if bars_placed < num_bars:
                    raise ValueError(f"Could not place all {num_bars} bars in grid")
                
                return grid
                
            except ValueError:
                if attempt == 19:
                    raise
                continue
        
        raise ValueError(f"Failed to generate grid after 20 attempts")
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        num_bars = taskvars['num_bars']

        colors = [
            taskvars['color_0'],
            taskvars['color_1'],
            taskvars['color_2'],
            taskvars['color_3'],
            taskvars['color_4'],
            taskvars['color_5'],
        ][:taskvars['num_bars']]

        objects = find_connected_objects(grid, background=0)

        if len(objects) > 0:
            sorted_objects = sorted(objects.objects, key=lambda obj: len(obj), reverse=True)

            for rank, bar_obj in enumerate(sorted_objects[:num_bars]):
                bar_color = colors[rank]
                for cell in bar_obj.cells:
                    r, c, _ = cell if len(cell) == 3 else (cell[0], cell[1], None)
                    output_grid[r, c] = bar_color

        return output_grid