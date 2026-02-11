from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskd23f8c26Generator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size n x n, where n is a positive odd integer greater than 1 and may vary for each input grid.",
            "Within each grid, m cells are randomly selected and assigned one of k distinct colors, where both m and k are integers greater than 2. The remaining cells are empty (0).",
            "Among the k colors, one is {color('color_1')} and one is {color('color_2')}."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "The output grid is constructed by copying the input grid.",
            "All cells except those in column {vars['target_col']} are set to empty (0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate random task variables
        taskvars = {
            'color_1': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            'color_2': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        }
        
        # Ensure colors are different
        while taskvars['color_2'] == taskvars['color_1']:
            taskvars['color_2'] = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        # Generate 3-6 training examples and 1 test example
        num_train = random.randint(3, 6)
        
        # First pass: generate all grids to find minimum size
        all_grids = []
        min_grid_size = float('inf')
        
        for _ in range(num_train + 1):
            input_grid = self.create_input(taskvars, {})
            all_grids.append(input_grid)
            min_grid_size = min(min_grid_size, input_grid.shape[0])
        
        # Set target_col based on smallest grid size
        target_col = random.randint(0, min_grid_size - 1)
        taskvars['target_col'] = target_col
        taskvars['min_grid_size'] = min_grid_size
        
        # Second pass: transform all grids using the same target_col
        train_examples = []
        for i in range(num_train):
            output_grid = self.transform_input(all_grids[i], taskvars)
            train_examples.append({'input': all_grids[i], 'output': output_grid})
        
        test_examples = []
        output_grid = self.transform_input(all_grids[num_train], taskvars)
        test_examples.append({'input': all_grids[num_train], 'output': output_grid})
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        # Generate random odd grid size between 5 and 29 (to stay within 30x30 limit)
        possible_sizes = [i for i in range(5, 30, 2)]  # odd numbers from 5 to 29
        n = random.choice(possible_sizes)
        
        # Create empty grid
        grid = np.zeros((n, n), dtype=int)
        
        # Generate k distinct colors (k > 2), ensuring color_1 and color_2 are included
        available_colors = [i for i in range(1, 10) if i not in [taskvars['color_1'], taskvars['color_2']]]
        k = random.randint(3, min(7, len(available_colors) + 2))  # k > 2, but not too many
        
        # Start with required colors
        color_palette = [taskvars['color_1'], taskvars['color_2']]
        
        # Add additional random colors to reach k colors
        additional_colors = random.sample(available_colors, k - 2)
        color_palette.extend(additional_colors)
        
        # Calculate m (number of colored cells) - must be at least one-third of total cells
        total_cells = n * n
        min_m = (total_cells + 2) // 3  # At least one-third (using ceiling division)
        max_m = min(total_cells - 1, int(total_cells * 0.8))  # Leave some empty cells, but not all
        m = random.randint(min_m, max_m)
        
        # Randomly select m cells to color
        all_positions = [(r, c) for r in range(n) for c in range(n)]
        selected_positions = random.sample(all_positions, m)
        
        # Assign random colors to selected positions
        for r, c in selected_positions:
            grid[r, c] = random.choice(color_palette)
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Copy the input grid
        output_grid = grid.copy()
        
        # Get target column from taskvars
        target_col = taskvars.get('target_col', 0)
        
        # Set all cells except target column to empty (0)
        n = grid.shape[0]
        for r in range(n):
            for c in range(n):
                if c != target_col:
                    output_grid[r, c] = 0
        
        return output_grid
