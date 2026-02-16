from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Taskac0a08a4Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} x {vars['cols']}.",
            "Each grid contains a varying number of colored cells, and each colored cell has a unique color.",
            "The positions of the colored cells within the input grid determine their placement in the output grid."
        ]
        
        transformation_reasoning_chain = [
            "The transformation follows these rules:",
            "For n colored cells (where n is the number of colors in the input), the output grid size is determined by n × {vars['rows']} by n × {vars['cols']}.",
            "Each colored cell expands to an n×n block in the output grid.",
            "Each block is placed in the output grid at the magnified position corresponding to its original cell position in the input grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        taskvars = {}
        
        # Generate variable grid size (these are consistent across all examples)
        rows = random.randint(3, 6)  # Input grid rows
        cols = random.randint(3, 6)  # Input grid columns
        
        taskvars['rows'] = rows
        taskvars['cols'] = cols
        
        nr_train_examples = random.randint(3, 5)
        nr_test_examples = 1
        
        # Create training examples with different numbers of colors
        train_examples = []
        used_color_counts = []
        
        for i in range(nr_train_examples):
            # Ensure each grid has a different number of colors
            available_counts = [n for n in range(2, 7) if n not in used_color_counts]
            if available_counts:
                current_num_colors = random.choice(available_counts)
                used_color_counts.append(current_num_colors)
            else:
                # If we've used all counts, pick randomly
                current_num_colors = random.randint(2, 6)
            
            # Create grid with specific number of colors
            input_grid = self.create_input(taskvars, {'num_colors': current_num_colors})
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example with a different number of colors
        available_test_counts = [n for n in range(2, 7) if n not in used_color_counts]
        test_num_colors = random.choice(available_test_counts) if available_test_counts else random.randint(2, 6)
        
        test_input = self.create_input(taskvars, {'num_colors': test_num_colors})
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create input grid with randomly placed colored cells."""
        rows = taskvars['rows']
        cols = taskvars['cols']
        n_objects = gridvars["num_colors"]  # Now from gridvars, not taskvars
        colors = random.sample(range(1, 10), n_objects)  # Select unique colors
        
        # Create empty grid of variable size
        grid = np.zeros((rows, cols), dtype=int)
        
        # Randomly place n colored cells
        positions = [(r, c) for r in range(rows) for c in range(cols)]
        
        # Ensure we don't try to place more colors than available positions
        max_colors = min(n_objects, len(positions))
        selected_positions = random.sample(positions, max_colors)
        
        for (r, c), color in zip(selected_positions, colors[:max_colors]):
            grid[r, c] = color
            
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """Transform input by expanding each colored cell to n×n blocks."""
        rows = taskvars['rows']
        cols = taskvars['cols']
        
        # Count the actual number of colors in this specific grid
        num_colors = len(np.unique(grid[grid > 0]))  # Count non-zero unique values
        
        block_size = num_colors  # Block size equals number of colors
        output_rows = rows * block_size  # Output rows
        output_cols = cols * block_size  # Output columns

        output_grid = np.zeros((output_rows, output_cols), dtype=int)

        for r in range(rows):
            for c in range(cols):
                color = grid[r, c]
                if color > 0:
                    row_start = r * block_size
                    col_start = c * block_size
                    output_grid[row_start:row_start+block_size, col_start:col_start+block_size] = color

        return output_grid