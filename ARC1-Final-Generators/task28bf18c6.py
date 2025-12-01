import random
import numpy as np
from typing import Dict, Any, Tuple
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry, Contiguity
from transformation_library import find_connected_objects

class Task28bf18c6Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['rows']}.",
            "A single 8-way connected object is present in the input grid which has color (between 1-9).",
            "The remaining cells are empty(0)."
        ]
        transformation_reasoning_chain = [
            "The output grid has a different size than the input grid.",
            "First identify the 8-way connected object and the subgrid that contains it.",
            "The output grid is constructed by horizontally joining two identical copies of the subgrid that includes the object, doubling its width compared to the original subgrid."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        color = gridvars['color']

        def generate_grid():
            return create_object(
                height=rows,
                width=rows,
                color_palette=color,
                contiguity=Contiguity.EIGHT,
                background=0
            )

        def check_grid(grid: np.ndarray) -> bool:
            if not np.all((grid == 0) | (grid == color)):
                return False
            objects = find_connected_objects(grid, diagonal_connectivity=True, background=0, monochromatic=True)
            if len(objects) != 1:
                return False
            # Check that the object doesn't occupy the entire grid
            object_size = np.sum(grid != 0)
            total_size = grid.size
            return object_size < total_size * 0.5  # Object should occupy less than 90% of the grid
            
        return retry(generate_grid, check_grid)

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0, monochromatic=True)
        assert len(objects) == 1, "Input grid must contain exactly one 8-connected object"
        obj = objects[0]
        subgrid = obj.to_array()
        return np.hstack([subgrid, subgrid])

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        rows = random.choice([9, 10])
        taskvars = {'rows': rows}
        train_count = random.choice([3, 4])
        total_examples = train_count + 1
        available_colors = list(range(1, 10))
        selected_colors = random.sample(available_colors, total_examples)
        train_colors = selected_colors[:train_count]
        test_color = selected_colors[train_count]

        train = []
        for color in train_colors:
            input_grid = self.create_input(taskvars, {'color': color})
            output_grid = self.transform_input(input_grid, taskvars)
            train.append({'input': input_grid, 'output': output_grid})

        test_input = self.create_input(taskvars, {'color': test_color})
        test_output = self.transform_input(test_input, taskvars)
        test = [{'input': test_input, 'output': test_output}]

        return taskvars, {'train': train, 'test': test}

if __name__ == "__main__":
    generator = ARCTask28bf18c6Generator()
    taskvars, train_test_data = generator.create_grids()
    ARCTaskGenerator.visualize_train_test_data(train_test_data)