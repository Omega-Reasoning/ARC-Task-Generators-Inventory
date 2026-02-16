from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry
import numpy as np
import random
from collections import deque

class TaskNhQ2VK7K9bmD9sQ6LsBe7W_1Generator(ARCTaskGenerator):
    def __init__(self):
        # Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain one {color('cell_color1')}, one {color('cell_color2')}, and empty (0) cells.",
            "Each colored cell is completely separated by empty (0) cells."
        ]
        # Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and expanding each colored cell vertically (up and down) and horizontally (left and right) until it meets another colored cell or the grid edge.",
            "The expansion begins with the {color('cell_color1')} cell, followed by {color('cell_color2')} cell.",
            "Once {color('cell_color1')} and {color('cell_color2')} cells expand into vertical and horizontal lines, fill all empty (0) cells in the subgrid that is enclosed by two {color('cell_color1')} and two {color('cell_color2')} lines with {color('fill_color')} color."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Generate train/test data while ensuring variety and constraints.
        """
        rows = random.randint(6, 30)
        cols = random.randint(6, 30)
        
        # Ensure distinct colors
        colors = random.sample(range(1, 10), 3)  # Pick three distinct colors
        cell_color1, cell_color2, fill_color = colors

        taskvars = {
            'rows': rows,
            'cols': cols,
            'cell_color1': cell_color1,
            'cell_color2': cell_color2,
            'fill_color': fill_color
        }

        # Ensure diverse train/test cases
        nr_train_examples = random.randint(3, 4)
        nr_test_examples = 1  # As per specification

        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)

        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        """
        Create an empty grid and place two colored cells with valid separation.
        """
        rows, cols = taskvars['rows'], taskvars['cols']
        color1, color2 = taskvars['cell_color1'], taskvars['cell_color2']

        grid = np.zeros((rows, cols), dtype=int)

        def generate_valid_positions():
            r1, c1 = random.randint(0, rows - 1), random.randint(0, cols - 1)
            r2, c2 = random.randint(0, rows - 1), random.randint(0, cols - 1)

            # Ensure separation by at least 2 rows and columns
            if abs(r1 - r2) >= 2 and abs(c1 - c2) >= 2:
                return (r1, c1), (r2, c2)
            return None

        (r1, c1), (r2, c2) = retry(generate_valid_positions, lambda x: x is not None)

        grid[r1, c1] = color1
        grid[r2, c2] = color2

        return grid

    def transform_input(self, grid, taskvars):
        """
        Apply transformations as per the reasoning chain:
        1. Expand the two colored cells.
        2. Fill the enclosed area with the third color.
        """
        color1, color2, fill_color = taskvars['cell_color1'], taskvars['cell_color2'], taskvars['fill_color']
        output_grid = grid.copy()

        # Expand both colors
        self._expand_color(output_grid, color1)
        self._expand_color(output_grid, color2)

        # Fill enclosed area
        self._fill_enclosed_area(output_grid, color1, color2, fill_color)

        return output_grid

    def _expand_color(self, grid, color):
        """
        Expand a colored cell horizontally and vertically until it meets another color or a boundary.
        """
        positions = np.argwhere(grid == color)
        if len(positions) == 0:
            return  # No cell of this color found
        
        r, c = positions[0]

        # Expand left
        for cc in range(c - 1, -1, -1):
            if grid[r, cc] != 0:
                break
            grid[r, cc] = color

        # Expand right
        for cc in range(c + 1, grid.shape[1]):
            if grid[r, cc] != 0:
                break
            grid[r, cc] = color

        # Expand up
        for rr in range(r - 1, -1, -1):
            if grid[rr, c] != 0:
                break
            grid[rr, c] = color

        # Expand down
        for rr in range(r + 1, grid.shape[0]):
            if grid[rr, c] != 0:
                break
            grid[rr, c] = color

    def _fill_enclosed_area(self, grid, color1, color2, fill_color):
        """
        Identify and fill only the subgrid that is fully enclosed by `color1` and `color2`.
        Uses a flood-fill approach to ensure correctness.
        """
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)

        # Step 1: Identify non-enclosed (open) regions using a BFS/DFS flood-fill from the edges
        queue = deque()
        
        # Start from all edge positions
        for r in range(rows):
            if grid[r, 0] == 0: queue.append((r, 0))
            if grid[r, cols - 1] == 0: queue.append((r, cols - 1))
        for c in range(cols):
            if grid[0, c] == 0: queue.append((0, c))
            if grid[rows - 1, c] == 0: queue.append((rows - 1, c))

        while queue:
            r, c = queue.popleft()
            if visited[r, c] or grid[r, c] != 0:
                continue
            visited[r, c] = True  # Mark as visited (open area)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-directional movement
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                    queue.append((nr, nc))

        # Step 2: Fill only the enclosed areas (cells not marked as open)
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 0 and not visited[r, c]:  # Enclosed region found
                    grid[r, c] = fill_color
