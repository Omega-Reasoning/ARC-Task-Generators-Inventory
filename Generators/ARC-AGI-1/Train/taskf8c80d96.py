from Framework.arc_task_generator import ARCTaskGenerator, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObject
import numpy as np
import random


class Taskf8c80d96Generator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a rectangular {color('background_color')} canvas containing two or more disconnected open rectangular outlines in one non-background color.",
            "2. Every outline uses the same selected subset of rectangle sides, such as an L, U, or C shape.",
            "3. From the smallest outline outward, its top, left, bottom, and right bounds change by one constant expansion vector.",
            "4. Only the first consecutive members of that expanding sequence are visible in the input; larger members are absent.",
            "5. The foreground color, outline orientation, expansion margin, base dimensions, and canvas dimensions vary between examples."
        ]
        transformation_reasoning_chain = [
            "1. Find the disconnected foreground outlines against {color('background_color')} and order them by bounding-box area.",
            "2. Determine which rectangle sides are drawn and infer the per-step bound changes from the two smallest outlines.",
            "3. Continue the same bound expansion, drawing and clipping each selected side, until another expansion has no cells inside the canvas.",
            "4. Preserve every sequence cell in its foreground color and replace all other cells with {color('completion_color')}."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        background = taskvars['background_color']
        height = gridvars['height']
        width = gridvars['width']
        foreground = gridvars['foreground_color']
        sides = gridvars['sides']
        delta = gridvars['delta']
        bounds = gridvars['base_bounds']
        visible_count = gridvars['visible_count']
        grid = np.full((height, width), background, dtype=int)

        for _ in range(visible_count):
            top, left, bottom, right = bounds
            cells = set()
            if 'top' in sides:
                cells.update((top, col, foreground) for col in range(left, right + 1))
            if 'bottom' in sides:
                cells.update((bottom, col, foreground) for col in range(left, right + 1))
            if 'left' in sides:
                cells.update((row, left, foreground) for row in range(top, bottom + 1))
            if 'right' in sides:
                cells.update((row, right, foreground) for row in range(top, bottom + 1))
            GridObject(cells).paste(grid)
            bounds = tuple(bounds[index] + delta[index] for index in range(4))
        return grid

    def transform_input(self, grid, taskvars):
        background = taskvars['background_color']
        completion_color = taskvars['completion_color']
        source = np.asarray(grid, dtype=int)
        height, width = source.shape
        foreground_values = [int(value) for value in np.unique(source) if int(value) != background]
        foreground = foreground_values[0]

        foreground_mask = np.full_like(source, background)
        foreground_mask[source == foreground] = foreground
        objects = find_connected_objects(
            foreground_mask,
            diagonal_connectivity=False,
            background=background,
            monochromatic=True
        )

        object_data = []
        for obj in objects:
            coords = obj.coords
            rows = [row for row, _ in coords]
            cols = [col for _, col in coords]
            bounds = (min(rows), min(cols), max(rows), max(cols))
            area = (bounds[2] - bounds[0] + 1) * (bounds[3] - bounds[1] + 1)
            object_data.append((area, bounds, coords))
        object_data.sort(key=lambda item: item[0])

        first_bounds = object_data[0][1]
        second_bounds = object_data[1][1]
        first_coords = object_data[0][2]
        top, left, bottom, right = first_bounds
        delta = tuple(second_bounds[index] - first_bounds[index] for index in range(4))

        sides = {
            'top': all((top, col) in first_coords for col in range(left, right + 1)),
            'bottom': all((bottom, col) in first_coords for col in range(left, right + 1)),
            'left': all((row, left) in first_coords for row in range(top, bottom + 1)),
            'right': all((row, right) in first_coords for row in range(top, bottom + 1))
        }

        output = np.full_like(source, completion_color)
        bounds = first_bounds
        for _ in range(4 * (height + width)):
            current_top, current_left, current_bottom, current_right = bounds
            cells = set()
            if sides['top'] and 0 <= current_top < height:
                for col in range(max(0, current_left), min(width - 1, current_right) + 1):
                    cells.add((current_top, col))
            if sides['bottom'] and 0 <= current_bottom < height:
                for col in range(max(0, current_left), min(width - 1, current_right) + 1):
                    cells.add((current_bottom, col))
            if sides['left'] and 0 <= current_left < width:
                for row in range(max(0, current_top), min(height - 1, current_bottom) + 1):
                    cells.add((row, current_left))
            if sides['right'] and 0 <= current_right < width:
                for row in range(max(0, current_top), min(height - 1, current_bottom) + 1):
                    cells.add((row, current_right))
            if not cells:
                break
            for row, col in cells:
                output[row, col] = foreground
            bounds = tuple(bounds[index] + delta[index] for index in range(4))
        return output

    def create_grids(self):
        completion_color = random.choice(range(1, 10))
        taskvars = {
            'background_color': 0,
            'completion_color': completion_color
        }
        foreground_choices = [color for color in range(1, 10) if color != completion_color]
        base_size = random.randint(13, 16)

        def make_gridvars(family, height, width, visible_count, step):
            foreground_color = random.choice(foreground_choices)
            if family == 'top_right_l':
                base_height = random.randint(2, 3)
                base_width = random.randint(2, 3)
                sides = ['top', 'right']
                base_bounds = (height - base_height, 0, height - 1, base_width - 1)
                delta = (-step, 0, 0, step)
            elif family == 'left_bottom_l':
                base_height = random.randint(2, 3)
                base_width = random.randint(2, 3)
                sides = ['left', 'bottom']
                base_bounds = (0, width - base_width, base_height - 1, width - 1)
                delta = (0, -step, step, 0)
            elif family == 'bottom_u':
                base_height = random.randint(2, 3)
                base_width = random.choice([3, 5])
                left = (width - base_width) // 2
                sides = ['left', 'right', 'bottom']
                base_bounds = (0, left, base_height - 1, left + base_width - 1)
                delta = (0, -step, step, step)
            elif family == 'right_c':
                base_height = random.choice([3, 5])
                base_width = random.randint(2, 3)
                top = (height - base_height) // 2
                sides = ['top', 'bottom', 'right']
                base_bounds = (top, 0, top + base_height - 1, base_width - 1)
                delta = (-step, 0, step, step)
            else:
                base_height = random.choice([3, 5])
                base_width = random.randint(2, 3)
                top = (height - base_height) // 2
                sides = ['top', 'bottom', 'left']
                base_bounds = (top, width - base_width, top + base_height - 1, width - 1)
                delta = (-step, -step, step, 0)
            return {
                'height': height,
                'width': width,
                'foreground_color': foreground_color,
                'sides': sides,
                'base_bounds': base_bounds,
                'delta': delta,
                'visible_count': visible_count
            }

        train_gridvars = [
            make_gridvars('top_right_l', base_size, base_size + 2, 2, random.choice([2, 3])),
            make_gridvars('left_bottom_l', base_size + 2, base_size + 1, 3, 2),
            make_gridvars('bottom_u', base_size + 1, base_size + 3, 2, random.choice([2, 3])),
            make_gridvars('right_c', base_size + 3, base_size + 2, 3, 2)
        ]
        test_gridvars = [
            make_gridvars('left_c', base_size + 4, base_size + 4, 2, 3)
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
