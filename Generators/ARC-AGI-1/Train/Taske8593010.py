from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class taske8593010(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}.",
            "Each grid is uniformly filled with the color {color('object_color')}, except for a few empty regions.",
            "Empty regions appear in three distinct shapes: A single isolated cell, two edge-adjacent cells (forming a straight line) and three edge-adjacent cells (either in a straight line or forming an L-shape).",
            "Each of the above shapes appears a random number of times, with at least two instances of each present in every input grid.",
            "No two empty regions are edge-adjacent to one another.",
            "Empty cells (0) make up approximately 25% of the entire grid."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "Empty regions in the input grid are identified and classified into one of the following three types: A single isolated cell, two edge-adjacent cells (forming a straight line) and three edge-adjacent cells (either in a straight line or forming an L-shape).",
            "These shapes are then filled in the output grid with the corresponding colors: {color('color_1')} for single isolated cells, {color('color_2')} for two edge-adjacent cells and {color('color_3')} for three edge-adjacent cells."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'n': random.randint(7, 30),
            'object_color': random.randint(1, 9),
            'color_1': random.randint(1, 9),  # single cell
            'color_2': random.randint(1, 9),  # two cells
            'color_3': random.randint(1, 9)   # three cells
        }
        
        # Ensure all colors are different
        used_colors = {taskvars['object_color']}
        for key in ['color_1', 'color_2', 'color_3']:
            while taskvars[key] in used_colors:
                taskvars[key] = random.randint(1, 9)
            used_colors.add(taskvars[key])
        
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
        
        train_test_data = {'train': train_examples, 'test': test_examples}
        
        return taskvars, train_test_data
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        object_color = taskvars['object_color']
        
        # Start with grid filled with object color
        grid = np.full((n, n), object_color, dtype=int)
        
        # Target approximately 25% empty cells
        target_empty = int(0.25 * n * n)
        
        # Keep track of empty regions to ensure they don't touch
        empty_positions = set()
        
        # Generate empty regions of each type (at least 2 of each)
        single_count = random.randint(2, max(2, target_empty // 6))
        double_count = random.randint(2, max(2, (target_empty - single_count) // 4))
        triple_count = random.randint(2, max(2, (target_empty - single_count - 2*double_count) // 3))
        
        def is_valid_position(positions):
            """Check if positions don't overlap with existing empty regions or touch them"""
            for r, c in positions:
                if (r, c) in empty_positions:
                    return False
                # Check if adjacent to any existing empty position
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    if (r+dr, c+dc) in empty_positions:
                        return False
            return True
        
        def add_empty_region(positions):
            """Add positions to empty regions and mark them in grid"""
            for r, c in positions:
                empty_positions.add((r, c))
                grid[r, c] = 0
        
        # Place single cells
        attempts = 0
        placed_singles = 0
        while placed_singles < single_count and attempts < 1000:
            r, c = random.randint(0, n-1), random.randint(0, n-1)
            if is_valid_position([(r, c)]):
                add_empty_region([(r, c)])
                placed_singles += 1
            attempts += 1
        
        # Place double cells (straight lines)
        attempts = 0
        placed_doubles = 0
        while placed_doubles < double_count and attempts < 1000:
            r, c = random.randint(0, n-1), random.randint(0, n-1)
            # Try horizontal or vertical
            if random.choice([True, False]) and c < n-1:  # horizontal
                positions = [(r, c), (r, c+1)]
            elif r < n-1:  # vertical
                positions = [(r, c), (r+1, c)]
            else:
                attempts += 1
                continue
                
            if is_valid_position(positions):
                add_empty_region(positions)
                placed_doubles += 1
            attempts += 1
        
        # Place triple cells (L-shapes or straight lines)
        attempts = 0
        placed_triples = 0
        while placed_triples < triple_count and attempts < 1000:
            r, c = random.randint(0, n-2), random.randint(0, n-2)
            
            # Different triple shapes
            shapes = []
            if r < n-2:  # vertical line
                shapes.append([(r, c), (r+1, c), (r+2, c)])
            if c < n-2:  # horizontal line  
                shapes.append([(r, c), (r, c+1), (r, c+2)])
            if r < n-1 and c < n-1:  # L-shapes
                shapes.extend([
                    [(r, c), (r+1, c), (r, c+1)],  # L bottom-left
                    [(r, c), (r, c+1), (r+1, c+1)],  # L top-left
                    [(r, c), (r+1, c), (r+1, c+1)],  # L top-right
                    [(r+1, c), (r+1, c+1), (r, c+1)]  # L bottom-right
                ])
            
            if shapes:
                positions = random.choice(shapes)
                if all(0 <= r < n and 0 <= c < n for r, c in positions) and is_valid_position(positions):
                    add_empty_region(positions)
                    placed_triples += 1
            attempts += 1
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        
        # Find all empty regions (connected components of 0s)
        empty_objects = find_connected_objects(grid, diagonal_connectivity=False, background=taskvars['object_color'], monochromatic=False)
        
        for obj in empty_objects:
            size = len(obj)
            
            if size == 1:
                # Single cell -> color_1
                fill_color = taskvars['color_1']
            elif size == 2:
                # Two cells -> color_2
                fill_color = taskvars['color_2']
            elif size == 3:
                # Three cells -> color_3
                fill_color = taskvars['color_3']
            else:
                # Shouldn't happen in our generation, but handle gracefully
                continue
            
            # Fill the region with the appropriate color
            for r, c, _ in obj.cells:
                output_grid[r, c] = fill_color
        
        return output_grid
