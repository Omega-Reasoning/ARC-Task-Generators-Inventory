from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.transformation_library import find_connected_objects, GridObject
from Framework.input_library import Contiguity, create_object, retry

class Task694f12f3Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain exactly two {color('block_color')} rectangular blocks, with the remaining cells being empty (0).",
            "The blocks must be completely separated from each other and occupy more than two rows and columns each.",
            "The sizes of the blocks must be different within each grid and should vary across examples."
        ]

        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the larger and smaller {color('block_color')} rectangular blocks.",
            "Once identified, fill all interior cells of the larger block with {color('fill_color1')}, leaving the {color('block_color')} border intact.",
            "Similarly, fill all interior cells of the smaller block with {color('fill_color2')}, also preserving its {color('block_color')} border."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        taskvars = {
            'grid_size': random.randint(12, 30),
            'block_color': random.randint(1, 9)
        }

        available_colors = [c for c in range(1, 10) if c != taskvars['block_color']]
        fill_colors = random.sample(available_colors, 2)
        taskvars['fill_color1'] = fill_colors[0]
        taskvars['fill_color2'] = fill_colors[1]

        num_train_examples = random.randint(3, 4)
        train_data = []

        for _ in range(num_train_examples):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})

        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)

        return taskvars, {
            'train': train_data,
            'test': [{'input': test_input, 'output': test_output}]
        }

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        grid_size = taskvars['grid_size']
        block_color = taskvars['block_color']
        
        def generate_valid_grid():
            grid = np.zeros((grid_size, grid_size), dtype=int)

            max_block_dim = max(3, grid_size // 2 - 1)

            block1_height = random.randint(3, max_block_dim)
            block1_width = random.randint(3, max_block_dim)

            for _ in range(10):
                block2_height = random.randint(3, max_block_dim)
                block2_width = random.randint(3, max_block_dim)
                if block1_height * block1_width != block2_height * block2_width:
                    break
            else:
                return None

            row1 = random.randint(0, grid_size - block1_height)
            col1 = random.randint(0, grid_size - block1_width)
            grid[row1:row1 + block1_height, col1:col1 + block1_width] = block_color

            for _ in range(50):
                row2 = random.randint(0, grid_size - block2_height)
                col2 = random.randint(0, grid_size - block2_width)

                if (row2 + block2_height <= row1 or row2 >= row1 + block1_height or
                    col2 + block2_width <= col1 or col2 >= col1 + block1_width):
                    grid[row2:row2 + block2_height, col2:col2 + block2_width] = block_color
                    return grid

            return None

        return retry(
            generate_valid_grid,
            lambda g: g is not None and len(find_connected_objects(g, diagonal_connectivity=False, background=0)) == 2,
            max_attempts=100
        )

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        output_grid = grid.copy()

        block_color = taskvars['block_color']
        fill_color1 = taskvars['fill_color1']
        fill_color2 = taskvars['fill_color2']

        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        objects = objects.sort_by_size(reverse=True)

        for i, obj in enumerate(objects):
            box = obj.bounding_box
            block_array = grid[box[0], box[1]]
            h, w = block_array.shape

            interior = np.zeros_like(block_array)
            interior[1:h-1, 1:w-1] = 1

            fill_color = fill_color1 if i == 0 else fill_color2

            for r in range(box[0].start, box[0].stop):
                for c in range(box[1].start, box[1].stop):
                    r_rel = r - box[0].start
                    c_rel = c - box[1].start

                    if (1 <= r_rel < h-1 and 1 <= c_rel < w-1 and 
                        output_grid[r, c] == block_color):
                        output_grid[r, c] = fill_color

        return output_grid
