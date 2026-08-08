from Framework.arc_task_generator import ARCTaskGenerator
from Framework.input_library import Contiguity, create_object, random_cell_coloring, retry
import numpy as np
import random


class Taskb190f7f5Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. The input contains two equal square panels concatenated horizontally or vertically without a delimiter.",
            "2. One panel is a monochromatic {color('template_color')} template mask with empty cells.",
            "3. The other panel is a symbol grid containing empty cells and several colors other than {color('template_color')}.",
            "4. Panel side, orientation, template geometry, and symbol arrangement vary between examples."
        ]
        transformation_reasoning_chain = [
            "1. Split the input into equal square panels and identify the {color('template_color')} template panel.",
            "2. Treat the other panel as the symbol grid.",
            "3. Replace each nonempty symbol by a copy of the template mask recolored with that symbol, and each empty symbol by an empty block.",
            "4. Concatenate all blocks into the square Kronecker-product output."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {'template_color': random.randint(1, 9)}
        train_gridvars = [
            {'side': 2, 'orientation': 'horizontal'},
            {'side': 3, 'orientation': 'vertical'},
            {'side': 4, 'orientation': 'horizontal'},
            {'side': 3, 'orientation': 'horizontal'},
        ]
        test_gridvars = {'side': 4, 'orientation': 'vertical'}
        train = []
        for gridvars in train_gridvars:
            input_grid = self.create_input(taskvars, gridvars)
            train.append({'input': input_grid, 'output': self.transform_input(input_grid, taskvars)})
        test_input = self.create_input(taskvars, test_gridvars)
        return taskvars, {
            'train': train,
            'test': [{'input': test_input, 'output': self.transform_input(test_input, taskvars)}],
        }

    def create_input(self, taskvars, gridvars):
        side = gridvars['side']
        template_color = taskvars['template_color']

        def sample_template():
            return create_object(
                side,
                side,
                template_color,
                contiguity=Contiguity.FOUR,
                background=0,
            )

        try:
            sampled_template = retry(
                sample_template,
                lambda value: 2 <= np.count_nonzero(value) < side * side,
                max_attempts=60,
            )
        except ValueError:
            sampled_template = np.zeros((side, side), dtype=int)
            sampled_template[:, side // 2] = template_color
            sampled_template[side // 2, :] = template_color
        template = np.asarray(
            gridvars.get("template", sampled_template),
            dtype=int,
        )
        if (
            template.shape != (side, side)
            or not 2 <= np.count_nonzero(template) < side * side
            or not set(int(value) for value in np.unique(template)).issubset(
                {0, template_color}
            )
        ):
            raise ValueError("template must be a nontrivial monochromatic side-square")
        symbol_palette = [color for color in range(1, 10) if color != template_color]

        def sample_symbols():
            symbols = np.zeros((side, side), dtype=int)
            return random_cell_coloring(
                symbols,
                random.sample(symbol_palette, min(3, len(symbol_palette))),
                density=random.uniform(0.3, 0.65),
                background=0,
                overwrite=False,
            )

        try:
            sampled_symbols = retry(
                sample_symbols,
                lambda value: (
                    2 <= np.count_nonzero(value) < side * side
                    and len({int(color) for color in np.unique(value) if color != 0}) >= 2
                ),
                max_attempts=40,
            )
        except ValueError:
            sampled_symbols = np.zeros((side, side), dtype=int)
            sampled_symbols[0, 0] = symbol_palette[0]
            sampled_symbols[-1, -1] = symbol_palette[1]
        symbols = np.asarray(
            gridvars.get("symbols", sampled_symbols),
            dtype=int,
        )
        symbol_colors = {int(color) for color in np.unique(symbols) if color != 0}
        if (
            symbols.shape != (side, side)
            or not 2 <= np.count_nonzero(symbols) < side * side
            or len(symbol_colors) < 2
            or template_color in symbol_colors
        ):
            raise ValueError("symbols must be a nontrivial multicolor side-square")
        sampled_panel_order = [0, 1]
        random.shuffle(sampled_panel_order)
        panel_order = list(gridvars.get("panel_order", sampled_panel_order))
        if sorted(panel_order) != [0, 1]:
            raise ValueError("panel_order must contain template and symbols once each")
        source_panels = (template, symbols)
        panels = [source_panels[index] for index in panel_order]
        axis = 1 if gridvars['orientation'] == 'horizontal' else 0
        return np.concatenate(panels, axis=axis)

    def transform_input(self, grid, taskvars):
        template_color = taskvars['template_color']
        if grid.shape[1] == 2 * grid.shape[0]:
            side = grid.shape[0]
            panels = [grid[:, :side], grid[:, side:]]
        else:
            side = grid.shape[1]
            panels = [grid[:side, :], grid[side:, :]]
        first_values = set(int(value) for value in np.unique(panels[0]) if value != 0)
        if first_values and first_values.issubset({template_color}):
            template, symbols = panels[0], panels[1]
        else:
            template, symbols = panels[1], panels[0]
        mask = (template != 0).astype(int)
        return np.kron(symbols, mask)
