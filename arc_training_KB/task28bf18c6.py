import random
import numpy as np
from typing import Dict, Any, Tuple
from arc_task_generator import ARCTaskGenerator, TrainTestData
from input_library import create_object, retry, Contiguity
from transformation_library import find_connected_objects


class Task28bf18c6Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid is a square of size {vars['rows']} x {vars['rows']}.",
"Exactly one monochromatic object is present, and it is 8-way connected (diagonal connections are allowed).",
"The object is fully surrounded by empty space: it does not touch the outer border, so there is at least one completely empty row above and below it, and at least one completely empty column to its left and right.",
"All remaining cells are empty (0).",

        ]
        transformation_reasoning_chain = [
            "The output grid has a different size than the input grid.",
"Identify the single 8-way connected object and extract its tight bounding-box subgrid (the smallest rectangle that fully contains the object).",
"Construct the output grid by horizontally concatenating two identical copies of this subgrid, doubling its width while keeping the height unchanged.",

        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars["rows"]
        color = gridvars["color"]

        def generate_grid():
            # Generate the object inside an inner canvas so the outer border is guaranteed empty
            inner = create_object(
                height=rows - 2,
                width=rows - 2,
                color_palette=color,
                contiguity=Contiguity.EIGHT,
                background=0
            )

            grid = np.zeros((rows, rows), dtype=int)
            grid[1:rows-1, 1:rows-1] = inner
            return grid

        def check_grid(grid: np.ndarray) -> bool:
            # Only background or the chosen color
            if not np.all((grid == 0) | (grid == color)):
                return False

            # Ensure 1-cell empty border (row/col) around the entire grid
            if np.any(grid[0, :] != 0) or np.any(grid[-1, :] != 0) or np.any(grid[:, 0] != 0) or np.any(grid[:, -1] != 0):
                return False

            objects = find_connected_objects(grid, diagonal_connectivity=True, background=0, monochromatic=True)
            if len(objects) != 1:
                return False

            # Ensure object is not "fully occupying" / too large
            object_size = int(np.sum(grid != 0))
            total_size = grid.size

            # keep your threshold (0.5) or adjust; 0.5 is already quite safe
            if object_size >= total_size * 0.5:
                return False

            # Also ensure it's not empty (just in case create_object can produce all-zeros)
            if object_size == 0:
                return False

            return True

        return retry(generate_grid, check_grid)


    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0, monochromatic=True)
        assert len(objects) == 1, "Input grid must contain exactly one 8-connected object"

        obj = objects[0]
        subgrid = obj.to_array()  # tight bbox subgrid
        return np.hstack([subgrid, subgrid])

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        rows = random.choice([9, 10])
        taskvars = {"rows": rows}

        train_count = random.choice([3, 4])
        total_examples = train_count + 1

        available_colors = list(range(1, 10))
        selected_colors = random.sample(available_colors, total_examples)
        train_colors = selected_colors[:train_count]
        test_color = selected_colors[train_count]

        train = []
        for color in train_colors:
            input_grid = self.create_input(taskvars, {"color": color})
            output_grid = self.transform_input(input_grid, taskvars)
            train.append({"input": input_grid, "output": output_grid})

        test_input = self.create_input(taskvars, {"color": test_color})
        test_output = self.transform_input(test_input, taskvars)
        test = [{"input": test_input, "output": test_output}]

        return taskvars, {"train": train, "test": test}



