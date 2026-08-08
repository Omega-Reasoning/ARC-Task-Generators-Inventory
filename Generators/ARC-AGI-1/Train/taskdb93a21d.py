from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry
from Framework.transformation_library import GridObject, find_connected_objects
import numpy as np
import random


class Taskdb93a21dGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is an empty grid containing one or more separated solid {color('block_color')} rectangular blocks.",
            "2. Every block has an even horizontal width, and its visible height and position vary between examples.",
            "3. A block may touch a grid boundary and can therefore be a boundary-clipped view of the same width-scaled construction.",
            "4. Block counts, widths, heights, and relative arrangements vary, while the three output color roles remain fixed within an episode.",
        ]
        transformation_reasoning_chain = [
            "1. Find every separated {color('block_color')} block and set its scale to half of its horizontal width.",
            "2. Surround the visible block with a {color('frame_color')} frame whose thickness equals that scale, clipping the frame at grid boundaries.",
            "3. From the lower edge of the completed frame, project a {color('beam_color')} beam with the original block width down to the bottom edge.",
            "4. Preserve each {color('block_color')} block and combine all generated frames and beams in the original grid dimensions.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        block_color, frame_color, beam_color = random.sample(range(1, 10), 3)
        taskvars = {
            "block_color": block_color,
            "frame_color": frame_color,
            "beam_color": beam_color,
        }

        def make_pair(sizes, clip_first=False):
            largest = max(sizes)
            count = len(sizes)
            gridvars = {
                "rows": max(16, 7 * count + largest + random.randint(1, 4)),
                "cols": max(16, 7 * count + largest + random.randint(1, 4)),
                "sizes": list(sizes),
                "clip_first": clip_first,
            }
            gridvars["rows"] = min(30, gridvars["rows"])
            gridvars["cols"] = min(30, gridvars["cols"])
            input_grid = self.create_input(taskvars, gridvars)
            return {
                "input": input_grid,
                "output": self.transform_input(input_grid, taskvars),
            }

        train = [
            make_pair([2]),
            make_pair([4]),
            make_pair([2, 6]),
            make_pair([2, 4, 6], clip_first=True),
        ]
        test = [make_pair([4, 6], clip_first=True)]
        return taskvars, {"train": train, "test": test}

    def create_input(self, taskvars, gridvars):
        rows = gridvars["rows"]
        cols = gridvars["cols"]
        block_color = taskvars["block_color"]
        sizes = gridvars["sizes"]

        def sample_scene():
            grid = np.zeros((rows, cols), dtype=int)
            for index, side in enumerate(sizes):
                visible_height = side // 2 if index == 0 and gridvars["clip_first"] else side
                row = gridvars.get(
                    f"row_{index}",
                    0
                    if index == 0 and gridvars["clip_first"]
                    else random.randint(1, rows - visible_height - 2),
                )
                col = gridvars.get(
                    f"col_{index}",
                    random.randint(1, cols - side - 2),
                )
                top, bottom = max(0, row - 1), min(rows, row + visible_height + 1)
                left, right = max(0, col - 1), min(cols, col + side + 1)
                if np.any(grid[top:bottom, left:right] != 0):
                    return None
                block = np.full((visible_height, side), block_color, dtype=int)
                GridObject.from_array(block, offset=(row, col)).paste(grid)
            return grid

        try:
            return retry(
                sample_scene,
                lambda grid: (
                    grid is not None
                    and len(
                        find_connected_objects(
                            grid,
                            diagonal_connectivity=False,
                            background=0,
                            monochromatic=True,
                        )
                    )
                    == len(sizes)
                ),
                max_attempts=100,
            )
        except ValueError:
            grid = np.zeros((rows, cols), dtype=int)
            for index, side in enumerate(sizes):
                visible_height = side // 2 if index == 0 and gridvars["clip_first"] else side
                row = 0 if index == 0 and gridvars["clip_first"] else 1 + index * 8
                col = 1 + index * 8
                block = np.full((visible_height, side), block_color, dtype=int)
                GridObject.from_array(block, offset=(row, col)).paste(grid)
            return grid

    def transform_input(self, grid, taskvars):
        block_color = taskvars["block_color"]
        frame_color = taskvars["frame_color"]
        beam_color = taskvars["beam_color"]
        objects = find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=0,
            monochromatic=True,
        ).with_color(block_color)
        output = grid.copy()
        descriptions = []
        for obj in objects:
            box = obj.bounding_box
            row_start, row_stop = box[0].start, box[0].stop
            col_start, col_stop = box[1].start, box[1].stop
            thickness = max(1, (col_stop - col_start) // 2)
            descriptions.append(
                (row_start, row_stop, col_start, col_stop, thickness)
            )

        for row_start, row_stop, col_start, col_stop, thickness in descriptions:
            beam_start = min(grid.shape[0], row_stop + thickness)
            output[beam_start:, col_start:col_stop] = beam_color

        for row_start, row_stop, col_start, col_stop, thickness in descriptions:
            outer_top = max(0, row_start - thickness)
            outer_bottom = min(grid.shape[0], row_stop + thickness)
            outer_left = max(0, col_start - thickness)
            outer_right = min(grid.shape[1], col_stop + thickness)
            output[outer_top:row_start, outer_left:outer_right] = frame_color
            output[row_stop:outer_bottom, outer_left:outer_right] = frame_color
            output[outer_top:outer_bottom, outer_left:col_start] = frame_color
            output[outer_top:outer_bottom, col_stop:outer_right] = frame_color

        output[grid == block_color] = block_color
        return output
