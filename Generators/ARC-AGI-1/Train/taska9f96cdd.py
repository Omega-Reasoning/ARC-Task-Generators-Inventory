import numpy as np
import random
from typing import Dict, Any, Tuple

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObject, GridObjects
from Framework.input_library import retry


class Taska9f96cddGenerator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} × {vars['cols']}.",
            "Each grid contains a single {color('cell_color1')} cell, while all remaining cells are empty (0).",
            "The position of the {color('cell_color1')} cell must vary across examples.",
        ]

        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the {color('cell_color1')} cell.",
            "Once the {color('cell_color1')} cell is identified, four new cells are added diagonally adjacent to it.",
            "The diagonal cells have fixed colors based on their positions: top-left → {color('cell_color2')}, top-right → {color('cell_color3')}, bottom-left → {color('cell_color4')}, and bottom-right → {color('cell_color5')}.",
            "It is possible that not all four diagonal cells can be added due to space constraints (e.g., near grid boundaries), in which case only the valid diagonal neighbors are added.",
            "Finally, the original {color('cell_color1')} cell is removed.",
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']

        grid = np.zeros((rows, cols), dtype=int)

        # Place the single colored cell at a position specified by gridvars (if given)
        if 'cell_row' in gridvars and 'cell_col' in gridvars:
            r = gridvars['cell_row']
            c = gridvars['cell_col']
        else:
            r = random.randint(0, rows - 1)
            c = random.randint(0, cols - 1)

        grid[r, c] = taskvars['cell_color1']
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output = grid.copy()
        rows, cols = grid.shape

        cell_color1 = taskvars['cell_color1']
        cell_color2 = taskvars['cell_color2']  # top-left
        cell_color3 = taskvars['cell_color3']  # top-right
        cell_color4 = taskvars['cell_color4']  # bottom-left
        cell_color5 = taskvars['cell_color5']  # bottom-right

        # Find the single colored cell
        positions = np.argwhere(grid == cell_color1)
        if len(positions) == 0:
            return output

        r, c = positions[0]

        # Remove original cell
        output[r, c] = 0

        # Add diagonal neighbors if within bounds
        diagonals = [
            (r - 1, c - 1, cell_color2),  # top-left
            (r - 1, c + 1, cell_color3),  # top-right
            (r + 1, c - 1, cell_color4),  # bottom-left
            (r + 1, c + 1, cell_color5),  # bottom-right
        ]

        for nr, nc, color in diagonals:
            if 0 <= nr < rows and 0 <= nc < cols:
                output[nr, nc] = color

        return output

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        rows = random.randint(5, 15)
        cols = random.randint(5, 15)

        # Pick 5 distinct non-zero colors
        colors = random.sample(range(1, 10), 5)
        cell_color1, cell_color2, cell_color3, cell_color4, cell_color5 = colors

        taskvars = {
            'rows': rows,
            'cols': cols,
            'cell_color1': cell_color1,
            'cell_color2': cell_color2,
            'cell_color3': cell_color3,
            'cell_color4': cell_color4,
            'cell_color5': cell_color5,
        }

        nr_train = random.randint(3, 6)

        train_examples = []
        used_positions = set()

        # Ensure at least one example where the cell has a 1-cell-wide empty frame (interior cell)
        # Interior means: row in [1, rows-2], col in [1, cols-2]
        interior_rows = list(range(1, rows - 1))
        interior_cols = list(range(1, cols - 1))

        if interior_rows and interior_cols:
            interior_r = random.choice(interior_rows)
            interior_c = random.choice(interior_cols)
            gridvars_interior = {'cell_row': interior_r, 'cell_col': interior_c}
            input_grid = self.create_input(taskvars, gridvars_interior)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
            used_positions.add((interior_r, interior_c))

        # Generate remaining training examples with varied positions
        attempts = 0
        while len(train_examples) < nr_train and attempts < 200:
            attempts += 1
            r = random.randint(0, rows - 1)
            c = random.randint(0, cols - 1)
            if (r, c) not in used_positions:
                used_positions.add((r, c))
                gridvars = {'cell_row': r, 'cell_col': c}
                input_grid = self.create_input(taskvars, gridvars)
                output_grid = self.transform_input(input_grid, taskvars)
                train_examples.append({'input': input_grid, 'output': output_grid})

        # Test example: pick a position not used in training
        test_r, test_c = retry(
            lambda: (random.randint(0, rows - 1), random.randint(0, cols - 1)),
            lambda pos: pos not in used_positions
        )
        gridvars_test = {'cell_row': test_r, 'cell_col': test_c}
        test_input = self.create_input(taskvars, gridvars_test)
        test_output = self.transform_input(test_input, taskvars)

        train_test_data: TrainTestData = {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }

        return taskvars, train_test_data



