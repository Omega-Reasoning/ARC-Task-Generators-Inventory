from Framework.arc_task_generator import ARCTaskGenerator, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObject
from Framework.input_library import create_object, retry, Contiguity
import numpy as np
import random


class Taskf8a8fe49Generator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a rectangular {color('background_color')} field containing one enclosure drawn in {color('frame_color')}.",
            "2. One opposite pair of enclosure sides is longer and more complete than the other pair, identifying either the horizontal-boundary or vertical-boundary regime.",
            "3. One or more eight-connected {color('movable_color')} objects lie inside the enclosure near one of the two long active sides.",
            "4. Objects can occur near both members of the active side pair, and their shapes and sizes vary between examples.",
            "5. The background, frame, and movable-object color roles are shared throughout the task."
        ]
        transformation_reasoning_chain = [
            "1. Find the {color('frame_color')} enclosure and compare color coverage on the two horizontal bounding-box sides with coverage on the two vertical sides.",
            "2. Select the more strongly represented opposite side pair as the active reflection boundaries.",
            "3. Group the {color('movable_color')} cells into eight-connected objects and choose the nearer active boundary for each object.",
            "4. Remove every object from its original location and reflect all of its cells across that boundary line.",
            "5. Preserve the {color('frame_color')} enclosure and every {color('background_color')} cell not occupied by a reflected object."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        background = taskvars['background_color']
        frame_color = taskvars['frame_color']
        movable_color = taskvars['movable_color']
        height = gridvars['height']
        width = gridvars['width']
        orientation = gridvars['orientation']
        object_count = gridvars['object_count']
        grid = np.full((height, width), background, dtype=int)

        top = gridvars.get('top', 4)
        bottom = gridvars.get('bottom', height - 5)
        left = gridvars.get('left', 4)
        right = gridvars.get('right', width - 5)
        if not (0 <= top < bottom < height and 0 <= left < right < width):
            raise ValueError('frame bounds must lie inside the canvas')
        frame_cells = set()
        if orientation == 'horizontal':
            for col in range(left, right + 1):
                frame_cells.add((top, col, frame_color))
                frame_cells.add((bottom, col, frame_color))
            for row in (top, top + 1, bottom - 1, bottom):
                frame_cells.add((row, left, frame_color))
                frame_cells.add((row, right, frame_color))
        else:
            for row in range(top, bottom + 1):
                frame_cells.add((row, left, frame_color))
                frame_cells.add((row, right, frame_color))
            for col in (left, left + 1, right - 1, right):
                frame_cells.add((top, col, frame_color))
                frame_cells.add((bottom, col, frame_color))
        GridObject(frame_cells).paste(grid)

        side_visits = [0, 0]
        for index in range(object_count):
            side = index % 2 if index < 2 else 0
            slot = side_visits[side]
            side_visits[side] += 1
            object_height = gridvars.get(
                f"object_height_{index}",
                random.randint(2, 3),
            )
            object_width = gridvars.get(
                f"object_width_{index}",
                random.randint(2, 3),
            )

            def sample_object():
                return create_object(
                    object_height,
                    object_width,
                    movable_color,
                    contiguity=Contiguity.EIGHT,
                    background=background
                )

            try:
                shape = gridvars.get(
                    f"shape_{index}",
                    retry(
                        sample_object,
                        lambda value: int(np.sum(value == movable_color))
                        >= min(2, object_height * object_width),
                        max_attempts=30,
                    ),
                )
            except ValueError:
                fallback_shape = np.full(
                    (object_height, object_width),
                    movable_color,
                    dtype=int,
                )
                shape = gridvars.get(f"shape_{index}", fallback_shape)
            shape = np.asarray(shape, dtype=int)
            if shape.shape != (object_height, object_width):
                raise ValueError('object shape does not match its routed dimensions')

            if orientation == 'horizontal':
                default_row = (
                    top + 2 if side == 0 else bottom - object_height - 1
                )
                low_col = left + 2
                high_col = right - object_width - 1
                default_col = (
                    low_col if (side == 0 and slot == 0) else high_col
                )
            else:
                default_col = (
                    left + 2 if side == 0 else right - object_width - 1
                )
                low_row = top + 2
                high_row = bottom - object_height - 1
                default_row = (
                    low_row if (side == 0 and slot == 0) else high_row
                )
            row = gridvars.get(f'row_{index}', default_row)
            col = gridvars.get(f'col_{index}', default_col)
            GridObject.from_array(shape, offset=(row, col)).paste(grid)
        return grid

    def transform_input(self, grid, taskvars):
        background = taskvars['background_color']
        frame_color = taskvars['frame_color']
        movable_color = taskvars['movable_color']
        source = np.asarray(grid, dtype=int)
        output = source.copy()

        frame_points = np.argwhere(source == frame_color)
        top = int(np.min(frame_points[:, 0]))
        bottom = int(np.max(frame_points[:, 0]))
        left = int(np.min(frame_points[:, 1]))
        right = int(np.max(frame_points[:, 1]))
        horizontal_score = int(np.sum(source[top, left:right + 1] == frame_color))
        horizontal_score += int(np.sum(source[bottom, left:right + 1] == frame_color))
        vertical_score = int(np.sum(source[top:bottom + 1, left] == frame_color))
        vertical_score += int(np.sum(source[top:bottom + 1, right] == frame_color))

        movable_mask = np.full_like(source, background)
        movable_mask[source == movable_color] = movable_color
        objects = find_connected_objects(
            movable_mask,
            diagonal_connectivity=True,
            background=background,
            monochromatic=True
        )

        reflected_cells = set()
        for obj in objects:
            obj.cut(output, background=background)
            rows = [row for row, _, _ in obj]
            cols = [col for _, col, _ in obj]
            if horizontal_score > vertical_score:
                use_top = abs(float(np.mean(rows)) - top) <= abs(bottom - float(np.mean(rows)))
                for row, col, color in obj:
                    new_row = 2 * top - row if use_top else 2 * bottom - row
                    reflected_cells.add((new_row, col, color))
            else:
                use_left = abs(float(np.mean(cols)) - left) <= abs(right - float(np.mean(cols)))
                for row, col, color in obj:
                    new_col = 2 * left - col if use_left else 2 * right - col
                    reflected_cells.add((row, new_col, color))

        GridObject(reflected_cells).paste(output)
        output[source == frame_color] = frame_color
        return output

    def create_grids(self):
        frame_color, movable_color = random.sample(range(1, 10), 2)
        taskvars = {
            'background_color': 0,
            'frame_color': frame_color,
            'movable_color': movable_color
        }
        base_height = random.randint(20, 22)
        base_width = random.randint(20, 22)
        train_gridvars = [
            {
                'height': base_height,
                'width': base_width + 2,
                'orientation': 'horizontal',
                'object_count': 2,
                'object_height_0': 2,
                'object_width_0': 3,
                'object_height_1': 2,
                'object_width_1': 2,
            },
            {
                'height': base_height + 2,
                'width': base_width,
                'orientation': 'vertical',
                'object_count': 2,
                'object_height_0': 3,
                'object_width_0': 2,
                'object_height_1': 2,
                'object_width_1': 3,
            },
            {
                'height': base_height + 1,
                'width': base_width + 3,
                'orientation': 'horizontal',
                'object_count': 3,
                'top': 4,
                'bottom': base_height - 3,
                'left': 4,
                'right': base_width - 2,
                'object_height_0': 2,
                'object_width_0': 5,
                'object_height_1': 1,
                'object_width_1': 1,
                'object_height_2': 3,
                'object_width_2': 2,
            },
            {
                'height': base_height + 3,
                'width': base_width + 1,
                'orientation': 'vertical',
                'object_count': 3,
                'top': 3,
                'bottom': base_height - 2,
                'left': 3,
                'right': base_width - 4,
                'object_height_0': 4,
                'object_width_0': 2,
                'object_height_1': 3,
                'object_width_1': 1,
                'object_height_2': 2,
                'object_width_2': 2,
            },
        ]
        test_orientation = random.choice(['horizontal', 'vertical'])
        test_height = base_height + 4
        test_width = base_width + 4
        test_gridvars = [
            {
                'height': test_height,
                'width': test_width,
                'orientation': test_orientation,
                'object_count': 3,
                'top': 4 if test_orientation == 'horizontal' else 3,
                'bottom': test_height - 4,
                'left': 4 if test_orientation == 'horizontal' else 3,
                'right': test_width - 5,
                'object_height_0': 2 if test_orientation == 'horizontal' else 4,
                'object_width_0': 4 if test_orientation == 'horizontal' else 2,
                'object_height_1': 1 if test_orientation == 'horizontal' else 3,
                'object_width_1': 3 if test_orientation == 'horizontal' else 1,
                'object_height_2': 3 if test_orientation == 'horizontal' else 2,
                'object_width_2': 2,
            }
        ]

        train = []
        test = []
        for gridvars in train_gridvars:
            input_grid = self.create_input(taskvars, gridvars)
            train.append({'input': input_grid, 'output': self.transform_input(input_grid, taskvars)})
        for gridvars in test_gridvars:
            input_grid = self.create_input(taskvars, gridvars)
            test.append({'input': input_grid, 'output': self.transform_input(input_grid, taskvars)})
        return taskvars, TrainTestData(train=train, test=test)
