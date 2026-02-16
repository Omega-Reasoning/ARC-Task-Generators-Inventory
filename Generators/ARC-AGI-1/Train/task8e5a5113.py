from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import create_object, retry, Contiguity, random_cell_coloring
from Framework.transformation_library import find_connected_objects
import numpy as np
import random


class Task8e5a5113Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['rows']} x {(vars['rows']*3)+2}.",
            "Each input grid is divided into three equal parts by completely filling columns {vars['rows']+1} and {(vars['rows']*2)+2} with {color('col_colour')} color.",
            "This results in three subgrids of size {vars['rows']} x {vars['rows']}, separated by {color('col_colour')} columns.",
            "The first {vars['rows']} x {vars['rows']} subgrid, on the left side of the grid, is completely filled with 4-way connected objects, with no empty cells.",
            "Within each grid, the first subgrid uses exactly {vars['num_colors_first_subgrid']} different colors (excluding {color('col_colour')}); these colors vary across examples."
        ]

        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the {color('col_colour')} dividers and the {vars['rows']} x {vars['rows']} subgrids separated by them.",
            "The first {vars['rows']} x {vars['rows']} subgrid is completely filled.",
            "The output grid is created by rotating the first subgrid 90 degree and 180 degree clockwise and placing it in the empty middle and last empty subgrids respectively."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        col_colour = taskvars['col_colour']
        width = (rows * 3) + 2

        grid = np.zeros((rows, width), dtype=int)

        # Dividers
        grid[:, rows] = col_colour
        grid[:, (rows * 2) + 1] = col_colour

        # Filled first subgrid
        first_subgrid = self._create_filled_subgrid(rows, taskvars)
        grid[:rows, :rows] = first_subgrid

        return grid

    def _choose_num_colors_for_rows(self, rows: int) -> int:
        """
        Small grids: 3 or 4 colors
        Medium grids: 5 or 6 colors
        Large grids: 7 or 8 colors

        (Supports up to 29 rows as requested.)
        """
        if rows <= 6:
            return random.choice([3, 4])
        elif rows <= 10:
            return random.choice([5, 6])
       

    def _create_filled_subgrid(self, size, taskvars):
        # Choose colors (excluding background 0 and the divider color)
        available_colors = [c for c in range(1, 10) if c != taskvars['col_colour']]

        # Use task-controlled number of colors (NOT size-1 anymore)
        num_colors = taskvars.get('num_colors_first_subgrid', None)
        if num_colors is None:
            num_colors = self._choose_num_colors_for_rows(size)

        # Safety: cannot exceed available distinct colors
        if num_colors > len(available_colors):
            raise ValueError(
                f"Not enough colors available. Need {num_colors}, have {len(available_colors)} "
                f"(col_colour={taskvars['col_colour']})."
            )

        colors = random.sample(available_colors, num_colors)

        # Start with the first color filling everything
        subgrid = np.full((size, size), colors[0], dtype=int)

        # Distribute other colors randomly across the grid (roughly equal)
        total_cells = size * size
        cells_per_color = total_cells // num_colors
        remaining_cells = total_cells % num_colors

        all_positions = [(r, c) for r in range(size) for c in range(size)]
        random.shuffle(all_positions)

        pos_idx = 0
        for i, color in enumerate(colors):
            cells_for_this_color = cells_per_color + (1 if i < remaining_cells else 0)
            for _ in range(cells_for_this_color):
                r, c = all_positions[pos_idx]
                subgrid[r, c] = color
                pos_idx += 1

        return subgrid

    def transform_input(self, grid, taskvars):
        rows = taskvars['rows']
        output_grid = grid.copy()

        first_subgrid = grid[:rows, :rows]
        rotated_90 = np.rot90(first_subgrid, k=-1)
        rotated_180 = np.rot90(first_subgrid, k=-2)

        middle_start = rows + 1
        output_grid[:rows, middle_start:middle_start + rows] = rotated_90

        last_start = (rows * 2) + 2
        output_grid[:rows, last_start:last_start + rows] = rotated_180

        return output_grid

    def create_grids(self):
        # Allow up to 29 rows as requested
        rows = random.randint(3, 9)

        col_colour = random.randint(1, 9)

        # New task var: number of colors used in the FIRST subgrid (same within each grid)
        num_colors_first_subgrid = self._choose_num_colors_for_rows(rows)

        taskvars = {
            'rows': rows,
            'col_colour': col_colour,
            'num_colors_first_subgrid': num_colors_first_subgrid
        }

        num_train = random.randint(3, 5)
        return taskvars, self.create_grids_default(num_train, 1, taskvars)
