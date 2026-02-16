import random
import numpy as np
from Framework.arc_task_generator import ARCTaskGenerator, TrainTestData

class Task7039b2d7Generator(ARCTaskGenerator):
    def __init__(self):
        # Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} by {vars['cols']}",
            "Each input grid is partitioned into many sub-grids.",
            "This partition is made with lines of the same color in a grid."
        ]
        # Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid copies the background color of the input grid",
            "Counts the rows and columns after being partitioned",
            "And forms a grid of that respective size."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """
        Create an input grid with strictly one-cell-wide partition lines.
        Ensure no two lines are adjacent (minimum spacing of one cell).
        """
        rows = taskvars['rows']
        cols = taskvars['cols']
        # Choose a background color (allowing zero)
        bg_color = random.choice([0] + list(range(1, 10)))
        # Choose a line color different from background
        line_color = random.choice([c for c in range(1, 10) if c != bg_color])
        # Initialize grid with background color
        grid = np.full((rows, cols), bg_color, dtype=int)
        # Determine number of lines (at least one of each)
        num_h = random.randint(1, max(1, rows // 4))
        num_v = random.randint(1, max(1, cols // 4))
        # Select horizontal positions with no adjacent lines
        while True:
            h_positions = sorted(random.sample(range(1, rows - 1), num_h))
            if all(j - i >= 2 for i, j in zip(h_positions, h_positions[1:])):
                break
        # Select vertical positions with no adjacent lines
        while True:
            v_positions = sorted(random.sample(range(1, cols - 1), num_v))
            if all(j - i >= 2 for i, j in zip(v_positions, v_positions[1:])):
                break
        # Draw strictly one-cell-wide horizontal lines
        for r in h_positions:
            grid[r, :] = line_color
        # Draw strictly one-cell-wide vertical lines
        for c in v_positions:
            grid[:, c] = line_color
        return grid

    def transform_input(self, grid: np.ndarray) -> np.ndarray:
        """
        Transform the partitioned grid into a smaller grid that counts partitions.
        Copies the background color and forms a grid of zero placeholders.
        """
        # Determine background as the most frequent color
        vals, counts = np.unique(grid, return_counts=True)
        bg_color = vals[np.argmax(counts)]
        # Identify line colors
        line_colors = [v for v in vals if v != bg_color]
        rows, cols = grid.shape
        h_lines = set()
        v_lines = set()
        # Scan for horizontal and vertical one-cell lines
        for color in line_colors:
            for i in range(rows):
                if np.all(grid[i, :] == color):
                    h_lines.add(i)
            for j in range(cols):
                if np.all(grid[:, j] == color):
                    v_lines.add(j)
        # Compute partition counts: lines + 1
        part_rows = len(h_lines) + 1
        part_cols = len(v_lines) + 1
        # Create output grid filled with background color
        output = np.full((part_rows, part_cols), bg_color, dtype=int)
        return output

    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        """Create train and test grids with consistent variables."""
        # Fix grid dimensions across all examples
        rows = random.randint(8, 20)
        cols = random.randint(8, 20)
        taskvars = {'rows': rows, 'cols': cols}
        # Generate 3-5 training examples
        train_examples = []
        for _ in range(random.randint(3, 5)):
            inp = self.create_input(taskvars, {})
            out = self.transform_input(inp)
            train_examples.append({'input': inp, 'output': out})
        # Single test example
        test_inp = self.create_input(taskvars, {})
        test_out = self.transform_input(test_inp)
        test_examples = [{'input': test_inp, 'output': test_out}]
        return taskvars, {'train': train_examples, 'test': test_examples}

if __name__ == '__main__':
    # Visualize a sample task
    generator = PartitionCountTask()
    vars, data = generator.create_grids()
    generator.visualize_train_test_data(data)
