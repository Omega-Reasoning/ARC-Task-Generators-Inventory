# my_arc_task_generator.py

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from collections import Counter

class TaskBmszN4FfYKKSiswJRzvBeQGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They are completely filled with multi-colored (1-9) cells.",
            "Each row must use only three distinct colors."
        ]

        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "Output grids are of size {vars['rows']}x1.",
            "They are constructed by identifying the most frequent color in each row and pasting it into the first column of the same row."
        ]

        # 3) Call the parent constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Creates a random number of training examples (3 or 4) plus one test example,
        all using the same (rows, cols). Returns the chosen task variables and
        the train/test data.
        """
        # Randomly choose dimensions for all examples
        rows = random.randint(5, 10)
        cols = random.randint(5, 10)

        taskvars = {
            "rows": rows,
            "cols": cols
        }

        # Randomly choose how many training examples to generate
        nr_train_examples = random.randint(3, 4)
        # Always one test example
        nr_test_examples = 1

        # Use the helper method to generate the data
        data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)

        return taskvars, data

    def create_input(self, taskvars, gridvars):
        """
        Create an input grid according to the specification:
          * Shape: (rows x cols)
          * Each row has exactly three distinct colors (from 1..9).
          * One of those colors appears strictly more often in that row than the other two.
        """
        rows = taskvars["rows"]
        cols = taskvars["cols"]

        grid = np.zeros((rows, cols), dtype=int)

        for r in range(rows):
            # Pick 3 distinct colors
            three_colors = random.sample(range(1, 10), 3)
            # Choose one to be the majority
            majority_color = random.choice(three_colors)
            other_colors = [c for c in three_colors if c != majority_color]

            # We want: majority_color count > each of the other two
            # We'll keep retrying until we get a valid distribution
            while True:
                majority_count = random.randint(1, cols - 2)
                c2_count = random.randint(1, cols - majority_count - 1)
                c3_count = cols - majority_count - c2_count
                if majority_count > c2_count and majority_count > c3_count:
                    break

            # Fill row array
            row_array = (
                [majority_color]*majority_count +
                [other_colors[0]]*c2_count +
                [other_colors[1]]*c3_count
            )

            random.shuffle(row_array)
            grid[r, :] = row_array

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Transform each input grid according to the transformation reasoning chain:
        * Output grid has shape (rows x 1).
        * For each row, identify the most frequent color and paste it into the first column.
        """
        import numpy as np
        from collections import Counter  # Ensure Counter is available

        rows = grid.shape[0]
        output_grid = np.zeros((rows, 1), dtype=int)

        for r in range(rows):
            row_data = grid[r, :]
            color_counts = Counter(row_data)
            # Find color with highest count
            most_frequent_color = max(color_counts, key=color_counts.get)
            output_grid[r, 0] = most_frequent_color

        return output_grid


