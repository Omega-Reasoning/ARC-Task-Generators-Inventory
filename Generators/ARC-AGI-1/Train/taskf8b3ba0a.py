from Framework.arc_task_generator import ARCTaskGenerator, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObject
from Framework.input_library import create_object, retry, Contiguity
import numpy as np
import random


class Taskf8b3ba0aGenerator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a regular raster of isolated one-row by {vars['token_width']}-column colored domino tokens separated by {color('divider_color')} rows and columns.",
            "2. Every token is monochromatic and all tokens have the same dimensions.",
            "3. One non-divider color is the dominant token color and fills most raster positions.",
            "4. Exactly {vars['signal_count']} other colors occur less often, with frequencies that can include ties.",
            "5. Raster dimensions, palettes, token frequencies, and token positions vary between examples."
        ]
        transformation_reasoning_chain = [
            "1. Parse every isolated one-row by {vars['token_width']}-column non-{color('divider_color')} object as one token and count tokens by color.",
            "2. Identify and exclude the most frequent color, which is the dominant filler token.",
            "3. Sort the remaining {vars['signal_count']} colors by decreasing token frequency, breaking equal-frequency ties by increasing numeric color value.",
            "4. Write that ordering from top to bottom in a {vars['signal_count']}-row by one-column output grid."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        divider_color = taskvars['divider_color']
        token_width = taskvars['token_width']
        macro_rows = gridvars['macro_rows']
        macro_cols = gridvars['macro_cols']
        dominant_color = gridvars['dominant_color']
        signal_colors = gridvars['signal_colors']
        signal_counts = gridvars['signal_counts']
        total_tokens = macro_rows * macro_cols
        signal_total = sum(signal_counts)
        colors = [dominant_color] * (total_tokens - signal_total)
        for color, count in zip(signal_colors, signal_counts):
            colors.extend([color] * count)

        def sample_token_order():
            shuffled = list(colors)
            random.shuffle(shuffled)
            return shuffled

        colors = [
            int(color)
            for color in gridvars.get("token_order", sample_token_order())
        ]

        grid = np.full(
            (2 * macro_rows + 1, (token_width + 1) * macro_cols + 1),
            divider_color,
            dtype=int
        )
        dominoes = {}
        for color in [dominant_color] + list(signal_colors):
            def sample_domino(current_color=color):
                return create_object(
                    1,
                    token_width,
                    current_color,
                    contiguity=Contiguity.FOUR,
                    background=divider_color
                )
            try:
                sampled_domino = retry(
                    sample_domino,
                    lambda value, current_color=color: bool(np.all(value == current_color)),
                    max_attempts=30
                )
            except ValueError:
                sampled_domino = np.full((1, token_width), color, dtype=int)
            dominoes[color] = np.asarray(
                gridvars.get(f"domino_{color}", sampled_domino), dtype=int
            )

        for index, color in enumerate(colors):
            macro_row, macro_col = divmod(index, macro_cols)
            row = 1 + 2 * macro_row
            col = 1 + (token_width + 1) * macro_col
            GridObject.from_array(dominoes[color], offset=(row, col)).paste(grid)
        return grid

    def transform_input(self, grid, taskvars):
        divider_color = taskvars['divider_color']
        token_width = taskvars['token_width']
        signal_count = taskvars['signal_count']
        source = np.asarray(grid, dtype=int)
        objects = find_connected_objects(
            source,
            diagonal_connectivity=False,
            background=divider_color,
            monochromatic=True
        )

        color_counts = {}
        for obj in objects:
            if len(obj) == token_width:
                color = int(next(iter(obj.colors)))
                color_counts[color] = color_counts.get(color, 0) + 1

        dominant_color = sorted(color_counts, key=lambda color: (-color_counts[color], color))[0]
        signal_colors = [color for color in color_counts if color != dominant_color]
        signal_colors.sort(key=lambda color: (-color_counts[color], color))
        return np.asarray([[color] for color in signal_colors[:signal_count]], dtype=int)

    def create_grids(self):
        taskvars = {
            'divider_color': 0,
            'token_width': 2,
            'signal_count': 3
        }
        shapes = [(4, 6), (5, 5), (3, 7), (6, 4)]
        count_patterns = [(4, 3, 2), (5, 3, 1), (4, 2, 2), (3, 3, 1)]
        train_gridvars = []
        for shape, counts in zip(shapes, count_patterns):
            palette = random.sample(range(1, 10), 4)
            train_gridvars.append({
                'macro_rows': shape[0],
                'macro_cols': shape[1],
                'dominant_color': palette[0],
                'signal_colors': palette[1:],
                'signal_counts': list(counts)
            })
        test_palette = random.sample(range(1, 10), 4)
        test_gridvars = [{
            'macro_rows': 5,
            'macro_cols': 6,
            'dominant_color': test_palette[0],
            'signal_colors': test_palette[1:],
            'signal_counts': [5, 2, 2]
        }]

        train = []
        test = []
        for gridvars in train_gridvars:
            input_grid = self.create_input(taskvars, gridvars)
            train.append({'input': input_grid, 'output': self.transform_input(input_grid, taskvars)})
        for gridvars in test_gridvars:
            input_grid = self.create_input(taskvars, gridvars)
            test.append({'input': input_grid, 'output': self.transform_input(input_grid, taskvars)})
        return taskvars, TrainTestData(train=train, test=test)
