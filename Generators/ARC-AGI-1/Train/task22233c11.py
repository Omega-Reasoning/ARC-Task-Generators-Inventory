from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry
from Framework.transformation_library import find_connected_objects
import numpy as np
import random


class Task22233c11Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. The input is an otherwise empty grid containing filled square objects of color {color('source_color')}.",
            "2. Equal-sized {color('source_color')} squares occur in pairs whose nearest corners touch diagonally.",
            "3. A pair can descend left or right, and its square size and location vary between examples.",
            "4. More than one separated pair may occur in the same grid.",
        ]
        transformation_reasoning_chain = [
            "1. Find the filled {color('source_color')} squares and pair equal squares that touch at one corner.",
            "2. For each pair, identify the diagonal through the two source squares and the perpendicular diagonal through their midpoint.",
            "3. Add two squares of the same size in color {color('added_color')} at the outer positions of that perpendicular diagonal.",
            "4. Preserve the source squares and clip any part of an added square that falls beyond the grid boundary.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows = gridvars['rows']
        cols = gridvars['cols']
        grid = np.zeros((rows, cols), dtype=int)
        for size, sign, top_row, top_col in gridvars['pairs']:
            bottom_row = top_row + size
            bottom_col = top_col + sign * size
            grid[top_row:top_row + size, top_col:top_col + size] = taskvars['source_color']
            grid[bottom_row:bottom_row + size, bottom_col:bottom_col + size] = taskvars['source_color']
        return grid

    def transform_input(self, grid, taskvars):
        source_color = taskvars['source_color']
        added_color = taskvars['added_color']
        objects = find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=0,
            monochromatic=True,
        ).with_color(source_color)
        squares = [
            obj for obj in objects
            if obj.height == obj.width and obj.size == obj.height * obj.width
        ]
        squares = sorted(
            squares,
            key=lambda obj: (obj.bounding_box[0].start, obj.bounding_box[1].start),
        )
        output = np.array(grid, copy=True)
        used = set()
        rows, cols = grid.shape
        for index, first in enumerate(squares):
            if index in used:
                continue
            size = first.height
            first_row = first.bounding_box[0].start
            first_col = first.bounding_box[1].start
            partner_index = None
            for candidate_index in range(index + 1, len(squares)):
                if candidate_index in used or squares[candidate_index].height != size:
                    continue
                candidate_row = squares[candidate_index].bounding_box[0].start
                candidate_col = squares[candidate_index].bounding_box[1].start
                if abs(candidate_row - first_row) == size and abs(candidate_col - first_col) == size:
                    partner_index = candidate_index
                    break
            if partner_index is None:
                continue
            used.add(index)
            used.add(partner_index)
            second = squares[partner_index]
            second_row = second.bounding_box[0].start
            second_col = second.bounding_box[1].start
            if second_row < first_row:
                first_row, second_row = second_row, first_row
                first_col, second_col = second_col, first_col
            sign = 1 if second_col > first_col else -1
            target_tops = [
                (first_row - size, first_col + 2 * sign * size),
                (second_row + size, second_col - 2 * sign * size),
            ]
            for top_row, top_col in target_tops:
                for row in range(top_row, top_row + size):
                    for col in range(top_col, top_col + size):
                        if 0 <= row < rows and 0 <= col < cols:
                            output[row, col] = added_color
        return output

    def create_grids(self):
        source_color, added_color = random.sample(range(1, 10), 2)
        taskvars = {'source_color': source_color, 'added_color': added_color}
        base_sign = random.choice([-1, 1])
        examples = [
            {'rows': 12, 'cols': 12, 'pairs': [(1, base_sign, 4, 4)]},
            {'rows': 14, 'cols': 14, 'pairs': [(2, -base_sign, 4, 6)]},
            {'rows': 16, 'cols': 16, 'pairs': [(3, base_sign, 4, 5)]},
            {'rows': 20, 'cols': 20, 'pairs': [(1, -base_sign, 4, 4), (2, base_sign, 11, 8)]},
            {'rows': 15, 'cols': 15, 'pairs': [(2, -1, 2, 2)]},
        ]
        train = []
        test = []
        for index, gridvars in enumerate(examples):
            input_grid = self.create_input(taskvars, gridvars)
            pair = GridPair(input=input_grid, output=self.transform_input(input_grid, taskvars))
            if index < 4:
                train.append(pair)
            else:
                test.append(pair)
        return taskvars, TrainTestData(train=train, test=test)
