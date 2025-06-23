from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskea32f347(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}",
            "Each grid contains three bars of the color {color('object_color')}, each with a distinct length.",
            "The lengths of the bars vary between different input grids.",
            "The bars may have different positions and orientations within the grid.",
            "The bars are spaced so that they neither overlap nor touch each other."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "The bar with the greatest length is colored with {color('color_1')}, the bar with the second greatest length is colored with {color('color_2')}, and the shortest bar is colored with {color('color_3')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'n': random.randint(5, 30),  # Grid size
            'object_color': random.randint(1, 9),
            'color_1': random.randint(1, 9),
            'color_2': random.randint(1, 9), 
            'color_3': random.randint(1, 9)
        }
        
        # Ensure all colors are different
        colors = [taskvars['color_1'], taskvars['color_2'], taskvars['color_3']]
        while len(set(colors)) != 3 or taskvars['object_color'] in colors:
            taskvars['color_1'] = random.randint(1, 9)
            taskvars['color_2'] = random.randint(1, 9)
            taskvars['color_3'] = random.randint(1, 9)
            colors = [taskvars['color_1'], taskvars['color_2'], taskvars['color_3']]
        
        # Create 3-6 training examples and 1 test example
        num_train = random.randint(3, 6)
        
        def generate_examples(n):
            examples = []
            for _ in range(n):
                input_grid = self.create_input(taskvars, {})
                output_grid = self.transform_input(input_grid, taskvars)
                examples.append({
                    'input': input_grid,
                    'output': output_grid
                })
            return examples
        
        train_test_data = {
            'train': generate_examples(num_train),
            'test': generate_examples(1)
        }
        
        return taskvars, train_test_data
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        object_color = taskvars['object_color']
        
        def generate_grid():
            grid = np.zeros((n, n), dtype=int)
            
            # Generate three bars with different lengths
            # Try to create bars that fit in the grid and don't overlap
            bars = []
            max_attempts = 100
            
            for attempt in range(max_attempts):
                bars = []
                grid.fill(0)
                
                # Generate three different lengths
                min_length = 2
                max_length = min(n - 1, 8)  # Ensure bars can fit
                
                lengths = []
                while len(lengths) < 3:
                    length = random.randint(min_length, max_length)
                    if length not in lengths:
                        lengths.append(length)
                
                # Try to place each bar
                success = True
                for length in lengths:
                    placed = False
                    for bar_attempt in range(50):
                        # Choose orientation (horizontal or vertical)
                        horizontal = random.choice([True, False])
                        
                        if horizontal:
                            # Horizontal bar
                            if length <= n:
                                row = random.randint(0, n - 1)
                                col = random.randint(0, n - length)
                                
                                # Check if this position is clear (with buffer)
                                clear = True
                                for check_r in range(max(0, row - 1), min(n, row + 2)):
                                    for check_c in range(max(0, col - 1), min(n, col + length + 1)):
                                        if grid[check_r, check_c] != 0:
                                            clear = False
                                            break
                                    if not clear:
                                        break
                                
                                if clear:
                                    # Place the bar
                                    for c in range(col, col + length):
                                        grid[row, c] = object_color
                                    bars.append(((row, col), (row, col + length - 1)))
                                    placed = True
                                    break
                        else:
                            # Vertical bar
                            if length <= n:
                                row = random.randint(0, n - length)
                                col = random.randint(0, n - 1)
                                
                                # Check if this position is clear (with buffer)
                                clear = True
                                for check_r in range(max(0, row - 1), min(n, row + length + 1)):
                                    for check_c in range(max(0, col - 1), min(n, col + 2)):
                                        if grid[check_r, check_c] != 0:
                                            clear = False
                                            break
                                    if not clear:
                                        break
                                
                                if clear:
                                    # Place the bar
                                    for r in range(row, row + length):
                                        grid[r, col] = object_color
                                    bars.append(((row, col), (row + length - 1, col)))
                                    placed = True
                                    break
                    
                    if not placed:
                        success = False
                        break
                
                if success and len(bars) == 3:
                    # Verify we have three different lengths
                    bar_lengths = []
                    objects = find_connected_objects(grid, background=0)
                    for obj in objects:
                        bar_lengths.append(len(obj))
                    
                    if len(set(bar_lengths)) == 3:  # All different lengths
                        return grid
            
            # If we couldn't place bars properly, create a simpler fallback
            grid.fill(0)
            lengths = [2, 3, 4]  # Simple different lengths
            
            # Place horizontally in different rows
            for i, length in enumerate(lengths):
                row = i * 2 + 1
                if row < n:
                    col = 1
                    if col + length <= n:
                        for c in range(col, col + length):
                            grid[row, c] = object_color
            
            return grid
        
        return generate_grid()
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        
        # Find all connected objects (bars)
        objects = find_connected_objects(grid, background=0)
        
        if len(objects) >= 3:
            # Sort objects by length (size)
            sorted_objects = sorted(objects.objects, key=lambda obj: len(obj), reverse=True)
            
            # Take the three largest objects (in case there are more than 3)
            longest = sorted_objects[0]
            middle = sorted_objects[1]  
            shortest = sorted_objects[2]
            
            # Recolor based on length
            # Longest gets color_1, middle gets color_2, shortest gets color_3
            for r, c, _ in longest.cells:
                output_grid[r, c] = taskvars['color_1']
            
            for r, c, _ in middle.cells:
                output_grid[r, c] = taskvars['color_2']
                
            for r, c, _ in shortest.cells:
                output_grid[r, c] = taskvars['color_3']
        
        return output_grid

