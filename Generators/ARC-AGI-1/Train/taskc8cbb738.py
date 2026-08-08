from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry
from Framework.transformation_library import GridObject, parse_objects_by_color
from typing import Dict, Any, Tuple
import numpy as np
import random


class Taskc8cbb738Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input has one dominant background color and several sparse patterns drawn in other colors.",
            "2. Cells of a given foreground color form one logical layer even when they are spatially disconnected.",
            "3. The colored layers are scattered into separate input regions and have different odd-sized bounding boxes.",
            "4. Every layer encodes cells of one hidden square composite under {vars['alignment_mode']} alignment.",
        ]
        transformation_reasoning_chain = [
            "1. Group all non-background cells by color and crop each color layer to its tight bounding box.",
            "2. Choose a square output side equal to the largest height or width among the cropped layers.",
            "3. Place every cropped layer at the {vars['alignment_mode']} of that square while preserving its internal geometry.",
            "4. Superimpose the aligned layers on the dominant background to recover the compact composite pattern.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        side = gridvars['side']
        num_layers = gridvars['num_layers']
        middle = side // 2

        def sample_colors():
            background = gridvars.get("background_color", random.randint(1, 9))
            available = [color for color in range(1, 10) if color != background]
            layer_colors = gridvars.get(
                "layer_colors", random.sample(available, num_layers)
            )
            return int(background), [int(color) for color in layer_colors]

        try:
            background_color, layer_colors = retry(
                sample_colors,
                lambda value: len(set(value[1])) == num_layers and value[0] not in value[1],
                max_attempts=20,
            )
        except ValueError:
            background_color = 9
            layer_colors = list(range(1, num_layers + 1))

        patterns = [
            {(0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1)},
            {(0, middle), (middle, 0), (middle, side - 1), (side - 1, middle)},
        ]
        if num_layers >= 3:
            patterns.append(
                {
                    (0, middle - 1),
                    (0, middle + 1),
                    (side - 1, middle - 1),
                    (side - 1, middle + 1),
                }
            )
        if num_layers >= 4:
            patterns.append(
                {
                    (middle - 1, 0),
                    (middle + 1, 0),
                    (middle - 1, side - 1),
                    (middle + 1, side - 1),
                }
            )

        def sample_pattern_additions():
            additions = []
            occupied = set().union(*patterns)
            for row in range(side):
                for col in range(side):
                    partner = (side - 1 - row, side - 1 - col)
                    pair = {(row, col), partner}
                    if (row, col) > partner or pair & occupied:
                        continue
                    if random.random() < 0.18:
                        target = random.randrange(num_layers)
                        additions.append((target, sorted(pair)))
                        occupied.update(pair)
            return additions

        pattern_additions = gridvars.get(
            "pattern_additions", sample_pattern_additions()
        )
        for target, pair in pattern_additions:
            patterns[int(target)].update(
                (int(row), int(col)) for row, col in pair
            )

        def sample_slot_origins():
            origins = [
                (1, 1),
                (1, side + 3),
                (side + 3, 1),
                (side + 3, side + 3),
            ]
            random.shuffle(origins)
            return origins

        slot_origins = [
            (int(row), int(col))
            for row, col in gridvars.get(
                "slot_origins", sample_slot_origins()
            )
        ]
        input_side = 2 * side + 4
        input_rows = int(gridvars.get("rows", input_side))
        input_cols = int(gridvars.get("cols", input_side))
        grid = np.full((input_rows, input_cols), background_color, dtype=int)
        for color, coords, origin in zip(layer_colors, patterns, slot_origins):
            min_row = min(row for row, _ in coords)
            max_row = max(row for row, _ in coords)
            min_col = min(col for _, col in coords)
            max_col = max(col for _, col in coords)
            local = np.zeros((max_row - min_row + 1, max_col - min_col + 1), dtype=int)
            for row, col in coords:
                local[row - min_row, col - min_col] = color
            GridObject.from_array(local, offset=origin).paste(
                grid,
                overwrite=True,
                background=background_color,
            )
        return grid

    def transform_input(self, grid, taskvars):
        alignment_mode = taskvars['alignment_mode']
        colors, counts = np.unique(grid, return_counts=True)
        background_color = int(colors[int(np.argmax(counts))])
        layers = parse_objects_by_color(grid, background=background_color)
        if len(layers) == 0:
            return np.array([[background_color]], dtype=int)

        side = max(max(layer.height, layer.width) for layer in layers)
        output = np.full((side, side), background_color, dtype=int)
        for layer in layers:
            box = layer.bounding_box
            if alignment_mode == 'center':
                row_offset = (side - layer.height) // 2
                col_offset = (side - layer.width) // 2
            else:
                row_offset = 0
                col_offset = 0
            for row, col, color in layer.cells:
                local_row = int(row) - int(box[0].start)
                local_col = int(col) - int(box[1].start)
                output[row_offset + local_row, col_offset + local_col] = int(color)
        return output

    def create_grids(self):
        taskvars = {'alignment_mode': 'center'}
        train_gridvars = [
            {'side': 3, 'num_layers': 2},
            {'side': 5, 'num_layers': 2},
            {'side': 5, 'num_layers': 3},
            {'side': 5, 'num_layers': 4},
        ]
        test_gridvars = [{'side': 7, 'num_layers': 4}]
        train = []
        test = []
        for gridvars in train_gridvars:
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train.append({'input': input_grid, 'output': output_grid})
        for gridvars in test_gridvars:
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            test.append({'input': input_grid, 'output': output_grid})
        return taskvars, {'train': train, 'test': test}
