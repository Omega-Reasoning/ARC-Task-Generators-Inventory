from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, GridObject, GridObjects


class Task6e02f1e3Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']} x {vars['grid_size']}.",
            "The grids are completely filled with colored objects, where each object consists of 4-way connected cells of the same color.",
            "The possible grid colors are: {color('object_color1')}, {color('object_color2')}, and {color('object_color3')}.",
            "The grid may be filled with one single color or a combination of two or three colors.",
            "The shapes of the objects should vary across examples."
        ]

        transformation_reasoning_chain = [
            "The output grid is constructed by identifying the number of different colors used in the input grid.",
            "If exactly one color is used, the first row of the output is filled with {color('object_color4')} color.",
            "If two colors are used, the main diagonal (top-left to bottom-right) of the output is filled with {color('object_color4')} color.",
            "If three colors are used, the inverse diagonal (top-right to bottom-left) of the output is filled with {color('object_color4')} color."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {
            'grid_size': random.randint(5, 30),
            'object_color1': random.randint(1, 9),  # A
            'object_color2': 0,                     # B
            'object_color3': 0,                     # C
            'object_color4': 0
        }

        # pick distinct colors
        pool = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        pool.remove(taskvars['object_color1'])
        taskvars['object_color2'] = random.choice(pool); pool.remove(taskvars['object_color2'])
        taskvars['object_color3'] = random.choice(pool); pool.remove(taskvars['object_color3'])
        taskvars['object_color4'] = random.choice(pool)

        A = taskvars['object_color1']
        B = taskvars['object_color2']
        C = taskvars['object_color3']

        train_data, test_data = [], []

        # Global uniqueness across train+test inputs
        seen_inputs = set()

        def key(g: np.ndarray) -> bytes:
            return g.tobytes()

        def gen_unique_input(gridvars: dict, max_tries: int = 800) -> np.ndarray:
            for _ in range(max_tries):
                g = self.create_input(taskvars, gridvars)
                k = key(g)
                if k not in seen_inputs:
                    seen_inputs.add(k)
                    return g
            raise RuntimeError("Could not generate a unique input grid after many attempts.")

        # -------------------------
        # TRAIN: exactly 4 examples (random order)
        #   - 1 single-color (randomly A/B/C)
        #   - AB
        #   - BC
        #   - ABC
        # -------------------------
        train_single_color = random.choice([A, B, C])

        train_plan = [
            {'colors': [train_single_color]},  # single
            {'colors': [A, B]},                # AB
            {'colors': [B, C]},                # BC
            {'colors': [A, B, C]},             # ABC
        ]
        random.shuffle(train_plan)

        for gridvars in train_plan:
            inp = gen_unique_input(gridvars)
            out = self.transform_input(inp, taskvars)
            train_data.append({'input': inp, 'output': out})

        # -------------------------
        # TEST: include a single-color example different from train single-color
        # plus 2-color + 3-color coverage (and all inputs unique).
        # -------------------------
        test_single_color = random.choice([c for c in [A, B, C] if c != train_single_color])

        test_plan = [
            {'colors': [test_single_color]},           # single (different from train single)
            {'colors': random.choice([[A, B], [B, C]])},  # a 2-color case
            {'colors': [A, B, C]},                     # 3-color case
        ]

        for gridvars in test_plan:
            inp = gen_unique_input(gridvars)
            out = self.transform_input(inp, taskvars)
            test_data.append({'input': inp, 'output': out})

        return taskvars, {'train': train_data, 'test': test_data}

    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']

        # Explicit colors (used by create_grids now)
        if 'colors' in gridvars and gridvars['colors'] is not None:
            colors = list(gridvars['colors'])
        else:
            available = [taskvars['object_color1'], taskvars['object_color2'], taskvars['object_color3']]
            random.shuffle(available)
            num_colors = gridvars.get('num_colors', random.randint(1, 3))
            colors = available[:num_colors]

        num_colors = len(colors)
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Single color: fill whole grid
        if num_colors == 1:
            grid.fill(colors[0])
            return grid

        # Multi-color: randomly distribute colors, avoid huge blobs
        max_blob_frac = 0.55 if num_colors == 2 else 0.45
        max_tries = 250

        for _ in range(max_tries):
            grid[:, :] = np.random.choice(colors, size=(grid_size, grid_size), replace=True)

            # ensure all colors appear
            if set(grid.flatten()) != set(colors):
                continue

            objs = find_connected_objects(grid, diagonal_connectivity=False, background=0)
            if not objs:
                continue

            largest = max(len(obj.coords) for obj in objs)
            if largest > int(grid_size * grid_size * max_blob_frac):
                continue

            return grid

        # fallback (rare)
        while True:
            grid[:, :] = np.random.choice(colors, size=(grid_size, grid_size), replace=True)
            if set(grid.flatten()) == set(colors):
                return grid

    def transform_input(self, grid, taskvars):
        grid_size = taskvars['grid_size']
        output_grid = np.zeros((grid_size, grid_size), dtype=int)
        highlight_color = taskvars['object_color4']

        unique_colors = set(grid.flatten()) - {0}
        num_colors = len(unique_colors)

        if num_colors == 1:
            output_grid[0, :] = highlight_color
        elif num_colors == 2:
            for i in range(grid_size):
                output_grid[i, i] = highlight_color
        elif num_colors == 3:
            for i in range(grid_size):
                output_grid[i, grid_size - 1 - i] = highlight_color

        return output_grid
