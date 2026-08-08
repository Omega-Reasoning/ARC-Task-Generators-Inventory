from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from Framework.transformation_library import parse_objects_by_color


class Taskeb5a1d5dGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input contains two or more solid axis-aligned rectangular color regions nested inside one another.",
            "2. Every inner rectangle is strictly contained by the preceding outer color region.",
            "3. Layer colors, input dimensions, margins, thicknesses, and core size vary; a color may recur at a nonadjacent inner level.",
            "4. Only the outside-to-inside color order is significant; absolute sizes and offsets are incidental.",
            "5. The output convention uses canonical layer thickness {vars['canonical_thickness']}.",
        ]
        transformation_reasoning_chain = [
            "1. Starting at the outside, recursively crop to the bounding box of cells that differ from the current layer color.",
            "2. Create the smallest odd square containing one concentric layer per input color.",
            "3. Paint the colors in their original outside-to-inside order using thickness {vars['canonical_thickness']}.",
            "4. Reduce the innermost region to the single center cell.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {"canonical_thickness": 1}
        train_gridvars = [
            {"level_count": 2, "repeat_depth": None},
            {"level_count": 3, "repeat_depth": None},
            {"level_count": 4, "repeat_depth": 3},
            {"level_count": 5, "repeat_depth": 2},
        ]
        test_gridvars = {"level_count": 6, "repeat_depth": 5}

        def make_pair(gridvars):
            input_grid = self.create_input(taskvars, gridvars)
            return {
                "input": input_grid,
                "output": self.transform_input(input_grid, taskvars),
            }

        return taskvars, {
            "train": [make_pair(gridvars) for gridvars in train_gridvars],
            "test": [make_pair(test_gridvars)],
        }

    def create_input(self, taskvars, gridvars):
        level_count = gridvars["level_count"]
        colors = list(
            gridvars.get("colors", random.sample(range(10), level_count))
        )
        repeat_depth = gridvars["repeat_depth"]
        if repeat_depth is not None:
            colors[repeat_depth] = colors[0]

        margins = [
            (
                gridvars.get(
                    f"margin_{depth}_top", random.randint(1, 2)
                ),
                gridvars.get(
                    f"margin_{depth}_bottom", random.randint(1, 2)
                ),
                gridvars.get(
                    f"margin_{depth}_left", random.randint(1, 2)
                ),
                gridvars.get(
                    f"margin_{depth}_right", random.randint(1, 2)
                ),
            )
            for depth in range(level_count - 1)
        ]
        core_height = gridvars.get("core_height", random.randint(2, 4))
        core_width = gridvars.get("core_width", random.randint(2, 5))
        rows = core_height + sum(top + bottom for top, bottom, _, _ in margins)
        cols = core_width + sum(left + right for _, _, left, right in margins)
        grid = np.full((rows, cols), colors[0], dtype=int)
        top, bottom = 0, rows
        left, right = 0, cols
        for depth in range(1, level_count):
            top_margin, bottom_margin, left_margin, right_margin = margins[depth - 1]
            top += top_margin
            bottom -= bottom_margin
            left += left_margin
            right -= right_margin
            grid[top:bottom, left:right] = colors[depth]
        return grid

    def transform_input(self, grid, taskvars):
        thickness = taskvars["canonical_thickness"]
        color_objects = parse_objects_by_color(grid, background=-1)
        if len(color_objects) == 0:
            return grid.copy()
        ordered_colors = []
        region = grid.copy()
        while region.size > 0:
            layer_color = int(region[0, 0])
            ordered_colors.append(layer_color)
            different = np.argwhere(region != layer_color)
            if len(different) == 0:
                break
            top, left = different.min(axis=0)
            bottom, right = different.max(axis=0)
            region = region[top : bottom + 1, left : right + 1]
        size = 2 * thickness * (len(ordered_colors) - 1) + 1
        output = np.zeros((size, size), dtype=int)
        for index, color in enumerate(ordered_colors):
            inset = index * thickness
            output[inset : size - inset, inset : size - inset] = color
        return output
