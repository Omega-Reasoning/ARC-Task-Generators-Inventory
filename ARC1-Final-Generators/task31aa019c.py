from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import random_cell_coloring
from transformation_library import find_connected_objects

class Task31aa019cGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain multi-colored (1-9) and empty (0) cells.",
            "Each example includes at least six different colors, where each color appears in multiple cells except for one color, which is used in only a single cell and must not be {color('cell_color')}.",
            "The single-occurence colored cell must not be positioned on the border of the grid."
        ]
        
        transformation_reasoning_chain = [
            "They are constructed by copying the input grid, removing all colored cells except for the single-occurrence cell, and then adding a one-cell-wide {color('cell_color')} frame around the single-occurrence cell."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Set task variables
        taskvars = {
            'rows': random.randint(5, 30),
            'cols': random.randint(5, 30),
            'cell_color': random.randint(1, 9)
        }
        
        # Generate 3-5 training examples and 1 test example
        num_train_examples = random.randint(3, 5)
        
        # Keep track of single-occurrence colors already used so each example
        # (train + test) has a different single-occurrence color.
        used_single_colors = set()

        train_pairs = []
        for _ in range(num_train_examples):
            gridvars = {'used_single_colors': used_single_colors}
            input_grid = self.create_input(taskvars, gridvars)
            # record the chosen single-occurrence color so it's not reused
            used_single_colors.add(gridvars['single_occurrence_color'])
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append({'input': input_grid, 'output': output_grid})

        # Create test example with a single-occurrence color different from all train examples
        gridvars = {'used_single_colors': used_single_colors}
        test_input = self.create_input(taskvars, gridvars)
        used_single_colors.add(gridvars['single_occurrence_color'])
        test_output = self.transform_input(test_input, taskvars)
        
        return taskvars, {
            'train': train_pairs,
            'test': [{'input': test_input, 'output': test_output}]
        }
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        cell_color = taskvars['cell_color']
        
        # Create an empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Generate a set of at least 6 different colors (excluding cell_color and 0)
        available_colors = [i for i in range(1, 10) if i != cell_color]
        num_colors = min(random.randint(6, 9), len(available_colors))
        colors_to_use = random.sample(available_colors, num_colors)
        
        # Choose a color from the set to be the single occurrence color.
        # If the caller provided a set of used single-occurrence colors (via
        # gridvars['used_single_colors']), avoid reusing them so each example
        # has a different single-occurrence color.
        used = set()
        if isinstance(gridvars, dict):
            used = set(gridvars.get('used_single_colors', set()))

        candidates = [c for c in colors_to_use if c not in used]
        if not candidates:
            # Fallback: try any available color (excluding the reserved cell_color)
            candidates = [c for c in available_colors if c not in used]
        if not candidates:
            raise ValueError("Not enough distinct colors available for unique single-occurrence assignments.")

        single_occurrence_color = random.choice(candidates)
        if single_occurrence_color in colors_to_use:
            colors_to_use.remove(single_occurrence_color)

        # Record the chosen single-occurrence color back into gridvars so the
        # caller can mark it as used.
        if isinstance(gridvars, dict):
            gridvars['single_occurrence_color'] = single_occurrence_color
        
        # Add the single occurrence color to a non-border position
        while True:
            r = random.randint(1, rows-2)  # Avoid border
            c = random.randint(1, cols-2)  # Avoid border
            if grid[r, c] == 0:
                grid[r, c] = single_occurrence_color
                break
        
        # Add multiple occurrences of other colors
        for color in colors_to_use:
            # Add at least 2 occurrences of each color
            occurrences = random.randint(2, 5)
            for _ in range(occurrences):
                attempts = 0
                while attempts < 100:  # Prevent infinite loop
                    r = random.randint(0, rows-1)
                    c = random.randint(0, cols-1)
                    if grid[r, c] == 0:
                        grid[r, c] = color
                        break
                    attempts += 1
        
        # Add more random colored cells to make the pattern less obvious
        random_cell_coloring(grid, colors_to_use, density=0.15, background=0)
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        cell_color = taskvars['cell_color']
        
        # Create a copy of the input grid
        result = grid.copy()
        
        # Find the single occurrence color
        color_counts = {}
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] != 0:
                    color_counts[grid[r, c]] = color_counts.get(grid[r, c], 0) + 1
        
        single_occurrence_color = None
        for color, count in color_counts.items():
            if count == 1:
                single_occurrence_color = color
                break
        
        # Clear all cells except the single occurrence cell
        result.fill(0)
        
        # Find the position of the single occurrence cell
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == single_occurrence_color:
                    # Place the single occurrence cell
                    result[r, c] = single_occurrence_color
                    
                    # Add a frame around the cell
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                                if not (dr == 0 and dc == 0):  # Skip the center
                                    result[nr, nc] = cell_color
                    
                    # No need to continue searching
                    break
        
        return result

