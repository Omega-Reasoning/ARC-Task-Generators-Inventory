import random
import numpy as np
from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry

class Task3428a4f5Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {2*vars['half_height']+1}×{vars['width']}.",
            "It contains two {vars['half_height']}×{vars['width']} panels, vertically stacked, separated by one full horizontal row of {color('separator_color')} color.",
            "Inside both panels, cells are either empty or filled with {color('input_color')} color."
        ]

        transformation_reasoning_chain = [
            "The output grid has size {vars['half_height']}×{vars['width']}.",
            "To construct the output grid, compare the top and bottom panels cell by cell.",
            "Whenever exactly one of the two corresponding cells contains a {color('input_color')} cell, place a {color('output_color')} cell in the output at that position.",
            "Whenever the two corresponding cells have the same value, place an empty cell (0) in the output.",
            "Thus, the output grid marks the cellwise XOR of the two panels."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        h = taskvars["half_height"]
        w = taskvars["width"]
        input_color = taskvars["input_color"]
        separator_color = taskvars["separator_color"]

        if "top_binary" in gridvars and "bottom_binary" in gridvars:
            top_bin = gridvars["top_binary"]
            bottom_bin = gridvars["bottom_binary"]
        else:
            total = h * w

            def gen():
                top = (np.random.rand(h, w) < random.uniform(0.25, 0.7)).astype(int)
                bottom = (np.random.rand(h, w) < random.uniform(0.25, 0.7)).astype(int)
                return top, bottom

            def predicate(tb):
                top, bottom = tb
                xor = top ^ bottom
                overlap = top & bottom
                both_zero = (top == 0) & (bottom == 0)

                return (
                    xor.sum() >= max(3, total // 6) and
                    overlap.sum() >= 1 and
                    both_zero.sum() >= 1 and
                    top.sum() >= max(2, total // 8) and
                    bottom.sum() >= max(2, total // 8) and
                    xor.sum() < total
                )

            top_bin, bottom_bin = retry(gen, predicate, max_attempts=200)

        top = np.where(top_bin == 1, input_color, 0)
        bottom = np.where(bottom_bin == 1, input_color, 0)
        separator = np.full((1, w), separator_color, dtype=int)

        return np.vstack([top, separator, bottom]).astype(int)

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        h = taskvars["half_height"]
        input_color = taskvars["input_color"]
        output_color = taskvars["output_color"]

        top = grid[:h, :]
        bottom = grid[h + 1:, :]

        top_mask = (top == input_color)
        bottom_mask = (bottom == input_color)
        xor_mask = np.logical_xor(top_mask, bottom_mask)

        return np.where(xor_mask, output_color, 0).astype(int)

    def create_grids(self):
        palette = list(range(1, 10))
        input_color = random.choice(palette)
        remaining = [c for c in palette if c != input_color]
        separator_color = random.choice(remaining)
        remaining = [c for c in remaining if c != separator_color]
        output_color = random.choice(remaining)

        taskvars = {
            "input_color": input_color,
            "separator_color": separator_color,
            "output_color": output_color,
            "half_height": random.randint(4, 13),
            "width": random.randint(4, 30),
        }

        h = taskvars["half_height"]
        w = taskvars["width"]
        total = h * w

        n_train = random.randint(3, 5)
        train = []

        for _ in range(n_train):
            def gen():
                top = (np.random.rand(h, w) < random.uniform(0.25, 0.7)).astype(int)
                bottom = (np.random.rand(h, w) < random.uniform(0.25, 0.7)).astype(int)
                return top, bottom

            def predicate(tb):
                top, bottom = tb
                xor = top ^ bottom
                overlap = top & bottom
                both_zero = (top == 0) & (bottom == 0)

                return (
                    xor.sum() >= max(3, total // 6) and
                    overlap.sum() >= 1 and
                    both_zero.sum() >= 1 and
                    top.sum() >= max(2, total // 8) and
                    bottom.sum() >= max(2, total // 8) and
                    xor.sum() < total
                )

            top_bin, bottom_bin = retry(gen, predicate, max_attempts=200)

            input_grid = self.create_input(
                taskvars,
                {"top_binary": top_bin, "bottom_binary": bottom_bin}
            )
            output_grid = self.transform_input(input_grid, taskvars)
            train.append({"input": input_grid, "output": output_grid})

        def gen():
            top = (np.random.rand(h, w) < random.uniform(0.25, 0.7)).astype(int)
            bottom = (np.random.rand(h, w) < random.uniform(0.25, 0.7)).astype(int)
            return top, bottom

        def predicate(tb):
            top, bottom = tb
            xor = top ^ bottom
            overlap = top & bottom
            both_zero = (top == 0) & (bottom == 0)

            return (
                xor.sum() >= max(3, total // 6) and
                overlap.sum() >= 1 and
                both_zero.sum() >= 1 and
                top.sum() >= max(2, total // 8) and
                bottom.sum() >= max(2, total // 8) and
                xor.sum() < total
            )

        top_bin, bottom_bin = retry(gen, predicate, max_attempts=200)

        test_input = self.create_input(
            taskvars,
            {"top_binary": top_bin, "bottom_binary": bottom_bin}
        )
        test_output = self.transform_input(test_input, taskvars)

        data = {
            "train": train,
            "test": [{"input": test_input, "output": test_output}]
        }

        return taskvars, data