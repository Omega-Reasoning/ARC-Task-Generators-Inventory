# rectangle_width_ordering_generator.py
from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# Optional: You can import from the provided libraries if desired.
# Below we use only Python's built-in random, but you could also do:
# from Framework.input_library import create_object, retry, Contiguity
# from Framework.transformation_library import find_connected_objects, GridObject, GridObjects

class Taskc8748oGCfemZNKsbPPkagiGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) The input reasoning chain (exactly one statement assigning the list of strings):
        input_reasoning_chain = [
            "Input grids are of size (1x{vars['cols']}).",
            "Each input grid contains three rectangular objects of different colors (1-9), with no empty cells.",
            "The width of each rectangle is different from the others."
        ]
        
        # 2) The transformation reasoning chain (exactly one statement assigning the list of strings):
        transformation_reasoning_chain = [
            "To construct the output grid, arrange the three rectangular objects in ascending order of width, from the narrowest to the widest."
        ]
        
        # 3) Call to the parent class constructor:
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        """
        Creates a dictionary of task variables (here, only 'cols') and 
        the train/test data for the ARC task.
        """
        # Decide how many training examples to produce, either 2 or 3
        num_train = random.choice([2, 3])
        
        # Randomly choose the total number of columns for this entire task
        # (invariant: 10 <= cols <= 30)
        cols = random.randint(10, 30)
        
        # Store in taskvars so that the reasoning chain template can substitute {vars['cols']}
        taskvars = {'cols': cols}
        
        train_pairs = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))

        # Create one test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)

    def create_input(self, taskvars, gridvars):
        """
        Create a 1 x cols grid containing exactly three rectangular segments
        (each of constant color, distinct from each other, each >= 2 columns wide),
        in a random order that is NOT strictly ascending by width.
        """
        cols = taskvars['cols']
        
        # Randomly choose 3 distinct widths (>=2) that sum to cols.
        # Also ensure the final arrangement is not already sorted ascending by width.
        w1, w2, w3 = self._generate_three_widths_summing_to(cols)
        
        # Randomly choose 3 distinct colors in [1..9]
        colors = random.sample(range(1, 10), 3)
        
        # Build the 1 x cols grid
        grid = np.zeros((1, cols), dtype=int)
        
        # Fill each segment left to right in the chosen order
        idx = 0
        for width, color in zip([w1, w2, w3], colors):
            grid[0, idx: idx + width] = color
            idx += width
        
        return grid

    def transform_input(self, grid, taskvars):
        """
        Transform the input grid by arranging the three color blocks in ascending order of width.
        Since the grid is 1 x cols with exactly three distinct color blocks, we can simply:
          1. Identify each block and its width.
          2. Sort them by ascending width.
          3. Re-place them left to right in the new order.
        """
        # grid shape: (1, cols)
        row = grid[0]
        
        # Identify the segments (color blocks) and their widths.
        # Each block is contiguous in this 1D row (and each color is distinct).
        segments = []
        current_color = row[0]
        current_start = 0
        
        for c in range(1, len(row)):
            if row[c] != current_color:
                # we found the end of the current block
                segments.append((current_color, c - current_start))  # (color, width)
                current_color = row[c]
                current_start = c
        # don't forget the final segment
        segments.append((current_color, len(row) - current_start))
        
        # Sort segments by ascending width
        segments_sorted = sorted(segments, key=lambda x: x[1])  # sort by width
        
        # Create a new output grid
        out_grid = np.zeros_like(grid)
        
        # Place the blocks in ascending order of width
        idx = 0
        for (color, width) in segments_sorted:
            out_grid[0, idx: idx + width] = color
            idx += width
        
        return out_grid
    
    def _generate_three_widths_summing_to(self, total_cols):
        """
        Helper to generate three distinct widths >=2 that sum to total_cols,
        with a random permutation that is not strictly ascending.
        Raises ValueError if cannot generate within tries (very unlikely).
        """
        for _ in range(500):  # A generous number of attempts
            # Pick two widths w1, w2; the third is total_cols - (w1+w2)
            w1 = random.randint(2, total_cols - 4)  # leave room for w2 and w3
            w2 = random.randint(2, total_cols - w1 - 2)
            w3 = total_cols - (w1 + w2)
            
            # Check constraints: each >= 2 and all distinct
            if w3 >= 2 and len({w1, w2, w3}) == 3:
                widths = [w1, w2, w3]
                random.shuffle(widths)
                
                # Check if widths are strictly ascending
                if not (widths[0] < widths[1] < widths[2]):
                    return widths[0], widths[1], widths[2]
        raise ValueError("Could not generate valid (w1, w2, w3) that sum to cols under constraints.")


