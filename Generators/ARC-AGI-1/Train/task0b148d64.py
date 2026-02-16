from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity
from transformation_library import find_connected_objects

class Task0b148d64Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['columns']}.",
            "Each input grid has four 8-way connected objects where each object is placed in one corner of the grid.",
            "Three objects are of color a and one object is of color b, where a and b are different colors.",
            "The remaining cells are empty (0)."
        ]
        transformation_reasoning_chain = [
            "The output grid size is different from the input grid size.",
            "First identify all colored objects in the input grid.",
            "The output grid contains three objects of color a and one object of color b, where a and b are different colors.",
            "The object of color b is extracted from the input grid and placed in the output grid."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        cols = taskvars['columns']
        color_1 = gridvars['color_1']
        color_2 = gridvars['color_2']

        grid = np.zeros((rows, cols), dtype=int)

        corners = [
            (0, 0),                # top-left
            (0, None),             # top-right
            (None, 0),             # bottom-left
            (None, None)           # bottom-right
        ]

        colors = [color_1, color_1, color_1, color_2]
        random.shuffle(colors)

        for (row_anchor, col_anchor), color in zip(corners, colors):

            # Random object size (at least 2x2)
            sub_rows = random.randint(2, rows // 3)
            sub_cols = random.randint(2, cols // 3)

            start_row = 0 if row_anchor == 0 else rows - sub_rows
            start_col = 0 if col_anchor == 0 else cols - sub_cols

            obj = np.zeros((sub_rows, sub_cols), dtype=int)

            # Corner pixel inside the subgrid
            corner_pixel = (
                0 if row_anchor == 0 else sub_rows - 1,
                0 if col_anchor == 0 else sub_cols - 1
            )

            obj[corner_pixel] = color

            # Grow an 8-connected shape
            max_pixels = sub_rows * sub_cols
            min_pixels = 4  # ensures at least 2x2 occupied

            upper = max(min_pixels, max_pixels // 2)
            target_pixels = random.randint(min_pixels, upper)

            filled = 1

            while filled < target_pixels:
                colored = np.argwhere(obj == color)
                candidates = set()

                for x, y in colored:
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if (
                                0 <= nx < sub_rows and
                                0 <= ny < sub_cols and
                                obj[nx, ny] == 0
                            ):
                                candidates.add((nx, ny))

                if not candidates:
                    break

                x, y = random.choice(list(candidates))
                obj[x, y] = color
                filled += 1

            grid[start_row:start_row + sub_rows,
                start_col:start_col + sub_cols] = obj

        return grid


    def transform_input(self, grid, taskvars):
        # Find connected objects
        objects = find_connected_objects(grid, diagonal_connectivity=True)
        # Count how many objects contain each color
        color_counts = {}
        for obj in objects:
            for color in obj.colors:
                color_counts[color] = color_counts.get(color, 0) + 1

        # Prefer the color that appears exactly once. If none, pick the least
        # frequent color as a fallback (this avoids IndexError when none appear
        # exactly once due to unexpected grid compositions).
        candidates = [color for color, count in color_counts.items() if count == 1]
        if candidates:
            color_2 = candidates[0]
        else:
            if not color_counts:
                raise ValueError("No non-background objects found in the grid.")
            # Choose the color with minimal object count (tie-breaker arbitrary)
            color_2 = min(color_counts.items(), key=lambda kv: kv[1])[0]

        # Locate the object with color_2 and return its cropped array
        for obj in objects:
            if color_2 in obj.colors:
                # Use GridObject.to_array() which returns the minimal bounding array
                return obj.to_array()

        raise ValueError("Object with specified color not found in the grid.")

    def create_grids(self):
        taskvars = {
            'rows': random.randint(15, 26),
            'columns': random.randint(15, 26),
        }

        nr_train = random.randint(3, 4)
        nr_test = 1
        total_examples = nr_train + nr_test

        all_colors = []
        available_colors = list(range(1, 10))

        for _ in range(total_examples):
            color_1 = random.choice(available_colors)
            available_colors.remove(color_1)

            color_2 = random.choice(available_colors)
            all_colors.append((color_1, color_2))

            available_colors.append(color_1)
            random.shuffle(available_colors)

        train_examples = []
        for i in range(nr_train):
            gridvars = {
                'color_1': all_colors[i][0],
                'color_2': all_colors[i][1]
            }
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append(GridPair({"input": input_grid, "output": output_grid}))

        gridvars = {
            'color_1': all_colors[-1][0],
            'color_2': all_colors[-1][1]
        }
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [GridPair({"input": test_input, "output": test_output})]

        return taskvars, TrainTestData({"train": train_examples, "test": test_examples})


