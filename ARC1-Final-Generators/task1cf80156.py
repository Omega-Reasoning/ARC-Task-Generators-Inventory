from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import create_object, Contiguity, retry
import numpy as np
import random

class Task1cf80156Generator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['cols']}.",
            "A single 8-way connected object is present in the input grid and the remaining cells are empty(0)"
        ]
        transformation_reasoning_chain = [
            "The output grid dimensions are different from the input grid dimensions.",
            "First identify the 8-way connected object in the input grid.",
            "The sub grid which contains the identified object is the output grid."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        rows, cols = taskvars['rows'], taskvars['cols']
        def generate_object():
            return create_object(
                height=random.randint(2, rows // 2),
                width=random.randint(2, cols // 2),
                # choose a random color for each generated object so colors vary across examples
                color_palette=random.randint(1, 9),
                contiguity=Contiguity.EIGHT,
                background=0
            )

        object_matrix = retry(
            generator=generate_object,
            predicate=lambda obj: 1 <= np.count_nonzero(obj) <= (rows * cols // 4)
        )

        # Place the object in a larger grid
        grid = np.zeros((rows, cols), dtype=int)
        obj_rows, obj_cols = object_matrix.shape
        start_row = random.randint(0, rows - obj_rows)
        start_col = random.randint(0, cols - obj_cols)
        grid[start_row:start_row + obj_rows, start_col:start_col + obj_cols] = object_matrix
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        # Identify the object in the grid (there will be a single connected object)
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0, monochromatic=True)
        # Pick the first detected object (no need to filter by color)
        target_object = objects[0]

        # Extract the object's bounding box and create the output grid
        bbox = target_object.bounding_box
        output_grid = grid[bbox[0], bbox[1]]
        return output_grid

    def create_grids(self) -> tuple:
        taskvars = {
            'rows': random.randint(8, 30),
            'cols': random.randint(8, 30),
        }
        nr_train = random.randint(3, 4)
        train_examples = []
        # Create multiple training examples; each example will get its own random color
        for _ in range(nr_train):  # Using self.num_train_examples from base class
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append(GridPair({"input": input_grid, "output": output_grid}))
        
        # Create test example (color chosen internally)
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [GridPair({"input": test_input, "output": test_output})]
        
        return taskvars, TrainTestData({"train": train_examples, "test": test_examples})


