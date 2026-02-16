from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple

class Task46doKrgAAegbVUKxffJdExGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['grid_size']} × {vars['grid_size']}.",
            "Each input grid is completely filled with horizontal strips.",
            "Some strips consist of a single color (which may vary across examples), while others consist of multiple colors.",
            "No cells are empty."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grid and identifying which horizontal rows are single-colored and which are multi-colored.",
            "All single-colored rows remain unchanged, while rows with multiple colors are replaced entirely with {color('strip')} colored cells."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        available_colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        if taskvars['strip'] in available_colors:
            available_colors.remove(taskvars['strip'])
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Track which rows should be single vs multi-colored
        num_rows = grid_size
        single_colored_rows = set()
        multi_colored_rows = set()
        
        # Ensure we have at least 2 single-colored and 2 multi-colored rows
        min_single = 2
        min_multi = 2
        
        # Randomly assign at least the minimum required rows
        all_rows = list(range(num_rows))
        random.shuffle(all_rows)
        
        # Assign minimum required rows
        single_colored_rows.update(all_rows[:min_single])
        multi_colored_rows.update(all_rows[min_single:min_single + min_multi])
        
        # Randomly assign remaining rows
        remaining_rows = all_rows[min_single + min_multi:]
        for row in remaining_rows:
            if random.choice([True, False]):
                single_colored_rows.add(row)
            else:
                multi_colored_rows.add(row)
        
        # Fill single-colored rows
        for row in single_colored_rows:
            color = random.choice(available_colors)
            grid[row, :] = color
        
        # Fill multi-colored rows (2-4 colors each)
        for row in multi_colored_rows:
            num_colors = random.randint(2, 4)
            row_colors = random.sample(available_colors, num_colors)
            
            # Randomly assign colors to cells in this row
            for col in range(grid_size):
                grid[row, col] = random.choice(row_colors)
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        strip_color = taskvars['strip']
        
        # Check each row
        for row_idx in range(grid.shape[0]):
            row = grid[row_idx, :]
            unique_colors = set(row[row != 0])  # Exclude background (though there shouldn't be any)
            
            # If row has multiple colors, replace with strip color
            if len(unique_colors) > 1:
                output_grid[row_idx, :] = strip_color
        
        return output_grid

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        grid_size = random.randint(5, 30)
        available_colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        strip_color = random.choice(available_colors)
        
        taskvars = {
            'grid_size': grid_size,
            'strip': strip_color
        }
        
        # Generate training examples
        num_train = random.randint(3, 5)
        train_examples = []
        
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }

