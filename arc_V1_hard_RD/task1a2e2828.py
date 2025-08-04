from arc_task_generator import ARCTaskGenerator
import numpy as np
import random

class Task1a2e2828Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid contains colored vertical and horizontal bars on a black background.",
            "Bars are 1 to 3 cells thick and span the full height or width of the grid.",
            "Each bar is drawn in a specific order, and where they intersect, the last drawn color remains visible.",
            "The color that appears on top at all its intersections is the output color."
        ]

        transformation_reasoning_chain = [
            "The output grid is of size 1x1.",
            "At each intersection point between vertical and horizontal bars, determine which color is visible.",
            "The color that appears on top at all intersections it is involved in is the dominant color.",
            "Set the output cell to this color."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows, cols = taskvars['rows'], taskvars['cols']
        grid = np.zeros((rows, cols), dtype=int)

        available_colors = list(range(1, 10))
        random.shuffle(available_colors)

        num_horizontals = random.randint(1, 2)
        num_verticals = random.randint(3, 4)

        bars = []
        color_index = 0

        for _ in range(num_horizontals):
            thickness = random.choice([1, 2, 3])
            start_row = random.randint(0, rows - thickness)
            color = available_colors[color_index]
            color_index += 1
            bars.append(('horizontal', start_row, thickness, color))

        for _ in range(num_verticals):
            thickness = random.choice([1, 2, 3])
            start_col = random.randint(0, cols - thickness)
            color = available_colors[color_index]
            color_index += 1
            bars.append(('vertical', start_col, thickness, color))

        random.shuffle(bars)  # simulate drawing order
        self.draw_order = bars

        for direction, start, thickness, color in bars:
            if direction == 'horizontal':
                for r in range(start, start + thickness):
                    if 0 <= r < rows:
                        grid[r, :] = color
            else:
                for c in range(start, start + thickness):
                    if 0 <= c < cols:
                        grid[:, c] = color

        return grid

    def transform_input(self, grid, taskvars):
        rows, cols = grid.shape
        draw_order = self.draw_order
        color_index_map = {color: i for i, (_, _, _, color) in enumerate(draw_order)}

        intersections = {}

        for r in range(rows):
            for c in range(cols):
                current_color = grid[r, c]
                if current_color == 0:
                    continue

                # determine if this is an intersection
                horiz = any((d == 'horizontal' and s <= r < s + t) for d, s, t, _ in draw_order)
                vert = any((d == 'vertical' and s <= c < s + t) for d, s, t, _ in draw_order)

                if horiz and vert:
                    if (r, c) not in intersections:
                        intersections[(r, c)] = grid[r, c]

        # for each color, track how many intersections it owns
        top_count = {}
        for color in np.unique(list(intersections.values())):
            top_count[color] = sum(1 for v in intersections.values() if v == color)

        if top_count:
            # pick the color with the most top-appearances, break ties by draw order
            dominant = sorted(top_count.items(), key=lambda x: (-x[1], color_index_map.get(x[0], 999)))[0][0]
            return np.array([[dominant]], dtype=int)

        # fallback
        colors = np.unique(grid)
        colors = colors[colors != 0]
        return np.array([[colors[0]]] if len(colors) > 0 else [[0]], dtype=int)

    def create_grids(self):
        grid_rows = random.randint(10, 18)
        grid_cols = random.randint(10, 18)

        taskvars = {
            'rows': grid_rows,
            'cols': grid_cols,
        }

        num_train_examples = random.randint(3, 5)
        train_examples = []

        for _ in range(num_train_examples):
            gridvars = {}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })

        test_gridvars = {}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)

        test_examples = [{
            'input': test_input,
            'output': test_output
        }]

        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
