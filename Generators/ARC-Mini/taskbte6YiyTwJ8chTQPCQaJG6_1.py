from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random


class Taskbte6YiyTwJ8chTQPCQaJG6_1Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain three {color('cell_color2')} cells arranged around one {color('cell_color1')} cell, positioned within the left half of the grid.",
            "Each {color('cell_color2')} cell must be 4-way connected to the {color('cell_color1')} cell."
        ]
        
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and rotating the three {color('cell_color2')} cells 90 degrees clockwise around the {color('cell_color1')} cell."
        ]
        
        # 3) Call superclass constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """
        Create an input grid of shape (rows x cols) containing:
          - One cell of color cell_color1
          - Three cells of color cell_color2 arranged in 4-way adjacency around the cell_color1 cell
          - All four colored cells are in the left half of the grid.
        """
        rows = taskvars["rows"]
        cols = taskvars["cols"]
        color1 = taskvars["cell_color1"]
        color2 = taskvars["cell_color2"]

        grid = np.zeros((rows, cols), dtype=int)

        # Pick a pivot (color1) in the left half, ensuring room for adjacent color2 cells
        def valid_positions():
            r = random.randint(1, rows - 2) if rows > 2 else 0
            c = random.randint(1, cols // 2 - 1)  # Ensure it's in the left half
            # Possible neighbors (up, down, left, right)
            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(neighbors)
            chosen = neighbors[:3]  # Pick 3 adjacent cells

            coords = [(r, c)]  # Pivot cell
            for (dr, dc) in chosen:
                rr = r + dr
                cc = c + dc
                coords.append((rr, cc))
            
            # Ensure all are in left half
            if all(cc < (cols // 2) for (_, cc) in coords):
                return coords
            return None

        coords = None
        attempts = 0
        while coords is None and attempts < 100:
            coords = valid_positions()
            attempts += 1

        # Fallback (unlikely)
        if coords is None:
            r = rows // 2
            c = cols // 4  # Left side
            coords = [(r, c), (r, c+1), (r-1, c), (r+1, c)]

        # Place colors
        pivot = coords[0]
        grid[pivot[0], pivot[1]] = color1
        for pos in coords[1:]:
            grid[pos[0], pos[1]] = color2

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """
        Apply the transformation:
        * Copy the input grid
        * Rotate the three cell_color2 cells 90° **clockwise** around the single cell_color1 pivot.
        """
        color1 = taskvars["cell_color1"]
        color2 = taskvars["cell_color2"]

        # Make a copy
        out_grid = grid.copy()

        # Find pivot cell (color1). There's exactly one.
        pivot_positions = np.argwhere(out_grid == color1)
        if len(pivot_positions) != 1:
            return out_grid  # Unexpected case

        pr, pc = pivot_positions[0]

        # Find color2 cells
        color2_positions = np.argwhere(out_grid == color2)

        # Remove existing color2 cells
        for (r, c) in color2_positions:
            out_grid[r, c] = 0

        # Apply **clockwise** 90-degree rotation
        for (r, c) in color2_positions:
            dr = r - pr
            dc = c - pc
            new_r = pr + dc
            new_c = pc - dr
            out_grid[new_r, new_c] = color2

        return out_grid

    def create_grids(self) -> (dict, TrainTestData):
        """
        Create 3-4 training pairs and 1 test pair.
        """
        rows_candidates = [r for r in range(3, 31) if r % 2 == 1]
        cols_candidates = [c for c in range(7, 31) if c % 2 == 0]

        rows = random.choice(rows_candidates)
        cols = random.choice(cols_candidates)

        color1 = random.randint(1, 9)
        color2 = random.randint(1, 9)
        while color2 == color1:
            color2 = random.randint(1, 9)

        taskvars = {
            "rows": rows,
            "cols": cols,
            "cell_color1": color1,
            "cell_color2": color2
        }

        nr_train_examples = random.choice([3, 4])
        nr_test_examples = 1

        train_test_data = self.create_grids_default(
            nr_train_examples=nr_train_examples,
            nr_test_examples=nr_test_examples,
            taskvars=taskvars
        )

        return taskvars, train_test_data


