from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, GridObject, GridObjects


class Task56dc2b01Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "They contain a single completely filled column or row with {color('line_color1')} color, forming a single vertical or horizontal {color('line_color1')} line, and an object made of 4-way connected cells of {color('object_color')} color.",
            "The {color('object_color')} object must be separated from the vertical or horizontal {color('line_color1')} line by at least three empty (0) columns or rows, respectively.",
            "The position and shape of the {color('object_color')} object varies across examples.",
            "All remaining cells are empty (0)."
        ]

        transformation_reasoning_chain = [
            "Output grids are constructed by copying the input grids and identifying whether the {color('line_color1')} line is horizontal or vertical.",
            "If the line is horizontal, the {color('object_color')} object is moved vertically towards it so they become vertically connected.",
            "If the line is vertical, the {color('object_color')} object is moved horizontally towards it so they become horizontally connected.",
            "Once the {color('object_color')} object is connected to the {color('line_color1')} line, a new line of {color('line_color2')} is added on the opposite side of the object by completely filling a row or column, depending on whether the original {color('line_color1')} line was horizontal or vertical.",
            "The new line is oriented in the same direction as the {color('line_color1')} line—horizontal or vertical.",
            "The transformation preserves the original shape of the {color('object_color')} object."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {
            'object_color': random.randint(1, 9),
            'line_color1': 0,
            'line_color2': 0
        }

        # Ensure colors are different
        while taskvars['line_color1'] == 0 or taskvars['line_color1'] == taskvars['object_color']:
            taskvars['line_color1'] = random.randint(1, 9)

        while (
            taskvars['line_color2'] == 0
            or taskvars['line_color2'] == taskvars['object_color']
            or taskvars['line_color2'] == taskvars['line_color1']
        ):
            taskvars['line_color2'] = random.randint(1, 9)

        train_examples = []

        # Ensure we have at least one horizontal and one vertical example
        horizontal_added = False
        vertical_added = False

        for i in range(random.randint(3, 4)):
            if i == 0 and not vertical_added:
                is_horizontal = False
                vertical_added = True
            elif i == 1 and not horizontal_added:
                is_horizontal = True
                horizontal_added = True
            else:
                is_horizontal = random.choice([True, False])

            gridvars = {'is_horizontal': is_horizontal}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)

            train_examples.append({'input': input_grid, 'output': output_grid})

        test_examples = []

        # Horizontal test example
        gridvars = {'is_horizontal': True}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        test_examples.append({'input': input_grid, 'output': output_grid})

        # Vertical test example
        gridvars = {'is_horizontal': False}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        test_examples.append({'input': input_grid, 'output': output_grid})

        return taskvars, {'train': train_examples, 'test': test_examples}

    def create_input(self, taskvars, gridvars):
        # Determine grid size (between 8x8 and 30x30)
        height = random.randint(8, 30)
        width = random.randint(8, 30)

        grid = np.zeros((height, width), dtype=int)

        is_horizontal = gridvars.get('is_horizontal', random.choice([True, False]))

        line_color = taskvars['line_color1']
        obj_color = taskvars['object_color']

        # Place the line
        if is_horizontal:
            line_position = random.randint(0, height - 1)
            grid[line_position, :] = line_color
        else:
            line_position = random.randint(0, width - 1)
            grid[:, line_position] = line_color

        # Object constraints
        obj_min_size = 4
        obj_max_size = min(10, height // 2, width // 2)
        min_separation = 3

        def create_candidate(h, w):
            return create_object(
                height=h,
                width=w,
                color_palette=obj_color,
                contiguity=Contiguity.FOUR
            )

        def is_valid_object(mat):
            non_zero_count = int(np.sum(mat != 0))
            nz = np.where(mat != 0)
            has_multi_rows = len(set(nz[0])) > 1
            has_multi_cols = len(set(nz[1])) > 1
            return non_zero_count >= 5 and has_multi_rows and has_multi_cols

        # Robust placement: NEVER allow the object bbox to cross the line row/col
        max_place_attempts = 200
        for _ in range(max_place_attempts):
            obj_height = random.randint(obj_min_size, obj_max_size)
            obj_width = random.randint(obj_min_size, obj_max_size)

            if obj_height > height or obj_width > width:
                continue

            if is_horizontal:
                valid_rows = []
                for r0 in range(0, height - obj_height + 1):
                    r1 = r0 + obj_height - 1

                    # forbid bbox intersection with line row
                    if r0 <= line_position <= r1:
                        continue

                    # enforce min separation from the line row
                    if r1 <= line_position - min_separation or r0 >= line_position + min_separation:
                        valid_rows.append(r0)

                if not valid_rows:
                    continue

                obj_row = random.choice(valid_rows)
                obj_col = random.randint(0, width - obj_width)

            else:
                valid_cols = []
                for c0 in range(0, width - obj_width + 1):
                    c1 = c0 + obj_width - 1

                    # forbid bbox intersection with line col
                    if c0 <= line_position <= c1:
                        continue

                    # enforce min separation from the line col
                    if c1 <= line_position - min_separation or c0 >= line_position + min_separation:
                        valid_cols.append(c0)

                if not valid_cols:
                    continue

                obj_col = random.choice(valid_cols)
                obj_row = random.randint(0, height - obj_height)

            object_matrix = retry(
                lambda: create_candidate(obj_height, obj_width),
                is_valid_object,
                max_attempts=50
            )

            # Final safety check: never overwrite line cells
            ok = True
            for r in range(obj_height):
                for c in range(obj_width):
                    if object_matrix[r, c] != 0:
                        rr, cc = obj_row + r, obj_col + c
                        if grid[rr, cc] == line_color:
                            ok = False
                            break
                if not ok:
                    break

            if not ok:
                continue

            # Paste object
            for r in range(obj_height):
                for c in range(obj_width):
                    if object_matrix[r, c] != 0:
                        grid[obj_row + r, obj_col + c] = object_matrix[r, c]

            return grid

        # Extremely rare fallback: regenerate safely
        return self.create_input(taskvars, gridvars)

    def transform_input(self, grid, taskvars):
        output = grid.copy()

        line_color = taskvars['line_color1']
        object_color = taskvars['object_color']
        new_line_color = taskvars['line_color2']

        # Detect orientation by "full line" criterion (more robust than max count compare)
        row_counts = np.sum(grid == line_color, axis=1)
        col_counts = np.sum(grid == line_color, axis=0)

        full_row = np.where(row_counts == grid.shape[1])[0]
        full_col = np.where(col_counts == grid.shape[0])[0]

        if len(full_row) == 1 and len(full_col) == 0:
            is_horizontal = True
            line_position = int(full_row[0])
        elif len(full_col) == 1 and len(full_row) == 0:
            is_horizontal = False
            line_position = int(full_col[0])
        else:
            # Fallback: original heuristic
            is_horizontal = np.max(row_counts) > np.max(col_counts)
            line_position = int(np.argmax(row_counts) if is_horizontal else np.argmax(col_counts))

        objects = find_connected_objects(grid, diagonal_connectivity=False)
        object_cells = objects.with_color(object_color)
        if len(object_cells) == 0:
            return grid

        obj = object_cells[0]

        # Cut object from output
        obj.cut(output)

        if is_horizontal:
            obj_rows = [r for r, _, _ in obj.cells]
            obj_top = min(obj_rows)
            obj_bottom = max(obj_rows)

            if obj_top > line_position:
                # move up
                distance = obj_top - line_position - 1
                obj.translate(-distance, 0)

                # new line on opposite side => below object
                new_line_row = (obj_bottom - distance) + 1
                if 0 <= new_line_row < output.shape[0]:
                    output[new_line_row, :] = new_line_color
            else:
                # move down
                distance = line_position - obj_bottom - 1
                obj.translate(distance, 0)

                # new line on opposite side => above object
                new_line_row = (obj_top + distance) - 1
                if 0 <= new_line_row < output.shape[0]:
                    output[new_line_row, :] = new_line_color
        else:
            obj_cols = [c for _, c, _ in obj.cells]
            obj_left = min(obj_cols)
            obj_right = max(obj_cols)

            if obj_left > line_position:
                # move left
                distance = obj_left - line_position - 1
                obj.translate(0, -distance)

                # new line on opposite side => right of object
                new_line_col = (obj_right - distance) + 1
                if 0 <= new_line_col < output.shape[1]:
                    output[:, new_line_col] = new_line_color
            else:
                # move right
                distance = line_position - obj_right - 1
                obj.translate(0, distance)

                # new line on opposite side => left of object
                new_line_col = (obj_left + distance) - 1
                if 0 <= new_line_col < output.shape[1]:
                    output[:, new_line_col] = new_line_color

        obj.paste(output)
        return output
