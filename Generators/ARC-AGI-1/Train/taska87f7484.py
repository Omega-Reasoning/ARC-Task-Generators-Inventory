from Framework.arc_task_generator import ARCTaskGenerator
from Framework.input_library import Contiguity, create_object, retry
import numpy as np
import random


class Taska87f7484Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. The input is a sequence of adjacent {vars['tile_size']}x{vars['tile_size']} tiles with no delimiter.",
            "2. The tile sequence may run horizontally or vertically.",
            "3. Every tile contains one nonzero color and empty cells.",
            "4. Exactly one tile contains strictly more nonempty cells than every other tile."
        ]
        transformation_reasoning_chain = [
            "1. Infer the sequence orientation and partition the input into {vars['tile_size']}x{vars['tile_size']} tiles.",
            "2. Count the nonempty cells in every tile.",
            "3. Select the uniquely densest tile.",
            "4. Return that tile unchanged as the {vars['tile_size']}x{vars['tile_size']} output."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {'tile_size': 3}
        train_gridvars = [
            {'orientation': 'vertical', 'tile_count': 3, 'winner_count': 7},
            {'orientation': 'horizontal', 'tile_count': 4, 'winner_count': 6},
            {'orientation': 'vertical', 'tile_count': 5, 'winner_count': 8},
            {'orientation': 'horizontal', 'tile_count': 3, 'winner_count': 5},
        ]
        test_gridvars = {'orientation': random.choice(['horizontal', 'vertical']), 'tile_count': 6, 'winner_count': 8}
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
        tile_size = taskvars['tile_size']
        tile_count = gridvars['tile_count']
        winner_count = gridvars['winner_count']
        winner_index = gridvars.get(
            'winner_index',
            random.randrange(tile_count),
        )
        colors = list(gridvars.get(
            'colors',
            [random.randint(1, 9) for _ in range(tile_count)],
        ))
        if not 0 <= winner_index < tile_count:
            raise ValueError('winner_index must select one tile')
        if (
            len(colors) != tile_count
            or any(color not in range(1, 10) for color in colors)
        ):
            raise ValueError('colors must contain one nonzero ARC color per tile')
        tiles = []
        for index in range(tile_count):
            count = (
                winner_count
                if index == winner_index
                else gridvars.get(
                    f'count_{index}',
                    random.randint(2, winner_count - 1),
                )
            )
            color = colors[index]

            def sample_tile():
                return create_object(
                    tile_size,
                    tile_size,
                    color,
                    contiguity=Contiguity.EIGHT,
                    background=0,
                )

            try:
                sampled_tile = retry(
                    sample_tile,
                    lambda value: int(np.count_nonzero(value)) == count,
                    max_attempts=80,
                )
            except ValueError:
                sampled_tile = np.zeros((tile_size, tile_size), dtype=int)
                snake = [(row, col) for row in range(tile_size) for col in range(tile_size)]
                for row, col in snake[:count]:
                    sampled_tile[row, col] = color
            tile = np.asarray(
                gridvars.get(f'tile_{index}', sampled_tile),
                dtype=int,
            )
            if (
                tile.shape != (tile_size, tile_size)
                or int(np.count_nonzero(tile)) != count
                or any(
                    int(value) not in (0, color)
                    for value in np.unique(tile)
                )
            ):
                raise ValueError(f'tile_{index} does not satisfy its count and color')
            tiles.append(tile)
        axis = 1 if gridvars['orientation'] == 'horizontal' else 0
        return np.concatenate(tiles, axis=axis)

    def transform_input(self, grid, taskvars):
        tile_size = taskvars['tile_size']
        tiles = []
        if grid.shape[0] == tile_size:
            for col in range(0, grid.shape[1], tile_size):
                tiles.append(grid[:, col:col + tile_size])
        else:
            for row in range(0, grid.shape[0], tile_size):
                tiles.append(grid[row:row + tile_size, :])
        counts = [int(np.count_nonzero(tile)) for tile in tiles]
        return tiles[int(np.argmax(counts))].copy()
