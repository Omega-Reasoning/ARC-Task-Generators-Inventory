from Framework.arc_task_generator import ARCTaskGenerator, TrainTestData
from Framework.input_library import retry
from Framework.transformation_library import GridObject
import numpy as np
import random


class Task9f236235Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. The input uses {color('background_color')} for empty cells and one-cell separator lines of varying non-background color.",
            "2. Complete separator rows and columns divide the canvas into equal square panels.",
            "3. Each panel is either empty or completely filled by one color.",
            "4. Panel count, panel side, separator color, and occupancy pattern vary by example.",
        ]
        transformation_reasoning_chain = [
            "1. Detect all separator rows and columns, each {vars['separator_thickness']} cell thick.",
            "2. Partition the input into the square panels between consecutive separators.",
            "3. Emit one output cell per panel: {color('background_color')} for an empty panel, otherwise its fill color.",
            "4. Reflect the smaller panel matrix left-to-right before returning it.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        taskvars = {'background_color': 0, 'separator_thickness': 1}
        specs = [
            {'panels': 3, 'block_size': 3, 'fill_probability': 0.45},
            {'panels': 3, 'block_size': 5, 'fill_probability': 0.65},
            {'panels': 4, 'block_size': 3, 'fill_probability': 0.4},
            {'panels': 4, 'block_size': 4, 'fill_probability': 0.7},
        ]
        random.shuffle(specs)
        train = []
        for spec in specs:
            input_grid = self.create_input(taskvars, spec)
            train.append({'input': input_grid, 'output': self.transform_input(input_grid, taskvars)})
        test_input = self.create_input(
            taskvars,
            {'panels': 5, 'block_size': random.choice([3, 4]), 'fill_probability': 0.55},
        )
        return taskvars, {'train': train, 'test': [{'input': test_input, 'output': self.transform_input(test_input, taskvars)}]}

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        background = taskvars['background_color']
        thickness = taskvars['separator_thickness']
        panels = gridvars['panels']
        block_size = gridvars['block_size']
        separator_color = gridvars.get(
            'separator_color',
            random.randint(1, 9),
        )
        palette_size = gridvars.get(
            'palette_size',
            random.randint(2, 4),
        )
        block_palette = list(gridvars.get(
            'block_palette',
            random.sample(
                [color for color in range(1, 10) if color != separator_color],
                palette_size,
            ),
        ))
        if (
            separator_color not in range(1, 10)
            or not 2 <= palette_size <= 4
            or len(block_palette) != palette_size
            or len(set(block_palette)) != len(block_palette)
            or any(
                color not in range(1, 10) or color == separator_color
                for color in block_palette
            )
        ):
            raise ValueError(
                'separator_color and block_palette must be distinct ARC colors'
            )

        def sample_panel_colors():
            values = np.full((panels, panels), background, dtype=int)
            for row in range(panels):
                for col in range(panels):
                    if random.random() < gridvars['fill_probability']:
                        values[row, col] = random.choice(block_palette)
            return values

        try:
            sampled_panel_colors = retry(
                sample_panel_colors,
                lambda values: (
                    np.any(values == background)
                    and np.count_nonzero(values != background) >= 2
                    and not np.array_equal(values, np.fliplr(values))
                ),
                max_attempts=60,
            )
        except ValueError:
            sampled_panel_colors = np.full((panels, panels), background, dtype=int)
            for index in range(panels):
                sampled_panel_colors[index, index] = block_palette[index % len(block_palette)]

        panel_colors = np.asarray(
            gridvars.get('panel_colors', sampled_panel_colors),
            dtype=int,
        )
        if (
            panel_colors.shape != (panels, panels)
            or not np.any(panel_colors == background)
            or np.count_nonzero(panel_colors != background) < 2
            or np.array_equal(panel_colors, np.fliplr(panel_colors))
            or any(
                int(color) != background and int(color) not in block_palette
                for color in np.unique(panel_colors)
            )
        ):
            raise ValueError('panel_colors do not satisfy the panel sampler contract')

        size = panels * block_size + (panels - 1) * thickness
        grid = np.full((size, size), background, dtype=int)
        for index in range(1, panels):
            separator_start = index * block_size + (index - 1) * thickness
            grid[separator_start:separator_start + thickness, :] = separator_color
            grid[:, separator_start:separator_start + thickness] = separator_color
        for panel_row in range(panels):
            for panel_col in range(panels):
                color = int(panel_colors[panel_row, panel_col])
                if color == background:
                    continue
                row_start = panel_row * (block_size + thickness)
                col_start = panel_col * (block_size + thickness)
                block = np.full((block_size, block_size), color, dtype=int)
                GridObject.from_array(block, offset=(row_start, col_start)).paste(grid)
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        background = taskvars['background_color']
        separator_rows = [
            row for row in range(grid.shape[0])
            if np.all(grid[row] == grid[row, 0]) and grid[row, 0] != background
        ]
        separator_cols = [
            col for col in range(grid.shape[1])
            if np.all(grid[:, col] == grid[0, col]) and grid[0, col] != background
        ]
        separator_color = int(grid[separator_rows[0], 0])
        row_edges = [-1] + separator_rows + [grid.shape[0]]
        col_edges = [-1] + separator_cols + [grid.shape[1]]
        output = np.full((len(row_edges) - 1, len(col_edges) - 1), background, dtype=int)
        for out_row in range(len(row_edges) - 1):
            for out_col in range(len(col_edges) - 1):
                panel = grid[
                    row_edges[out_row] + 1:row_edges[out_row + 1],
                    col_edges[out_col] + 1:col_edges[out_col + 1],
                ]
                values = panel[(panel != background) & (panel != separator_color)]
                if values.size:
                    colors, counts = np.unique(values, return_counts=True)
                    output[out_row, out_col] = colors[int(np.argmax(counts))]
        return np.fliplr(output).copy()
