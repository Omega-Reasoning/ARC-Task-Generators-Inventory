
from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple
from Framework.transformation_library import find_connected_objects
from Framework.input_library import create_object, Contiguity


class Task1bfc4729Generator(ARCTaskGenerator):
	def __init__(self):
		input_reasoning_chain = [
			"The input grids are of size {vars['grid_size']} × {vars['grid_size']}.",
			"Each input grid contains exactly two differently colored cells, while the remaining cells are empty.",
			"The positions of these two colored cells follow a mirrored row-wise pattern.",
			"Depending on the grid size, the cells may appear in the 3rd and 3rd-last row, 4th and 4th-last row, or 5th and 5th-last row, but there must always be at least 3 rows separating the two colored cells.",
			"All examples use the same rows for placing the two cells. The column positions of the two colored cells can be arbitrary."
		]

		transformation_reasoning_chain = [
			"The output grids are created by copying the input grid and identifying the two colored cells.",
			"The first row and the row containing the top colored cell are filled entirely with the color of the top colored cell.",
			"The last row and the row containing the bottom colored cell are filled entirely with the color of the bottom colored cell.",
			"The first and last columns are also filled: from the first row down to the middle row they are filled with the color of the top colored cell; from the row below the middle row down to the last row they are filled with the color of the bottom colored cell."
		]

		super().__init__(input_reasoning_chain, transformation_reasoning_chain)

	def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
		# Choose an even grid size between 8 and 30 (even and >=8 satisfies "even, between 7 and 30")
		# Ensure chosen grid size allows a placement p in {3,4,5} with at least 3 rows between the two markers
		def valid_size(n: int) -> bool:
			# there must exist p in {3,4,5} such that n - 2*p >= 3
			return any(n - 2 * p >= 3 for p in (3, 4, 5))

		grid_size = random.choice([s for s in range(8, 31) if s % 2 == 0 and valid_size(s)])

		# Choose which row index (1-based p) to use for the top marker among {3,4,5} but compatible with grid_size
		allowed_p = [p for p in (3, 4, 5) if grid_size - 2 * p >= 3]
		p = random.choice(allowed_p)

		taskvars = {
			'grid_size': grid_size,
			'row_position_p': p  # 1-based position (e.g. 3 => 3rd and 3rd-last rows)
		}

		# Number of training examples (3-5) and one test
		n_train = random.randint(3, 5)

		train = []
		for _ in range(n_train):
			gridvars = {
				'top_color': random.randint(1, 9),
				'bottom_color': random.randint(1, 9),
				# pick random columns (0-based)
				'top_col': random.randrange(0, grid_size),
				'bottom_col': random.randrange(0, grid_size)
			}
			# ensure different colors
			while gridvars['bottom_color'] == gridvars['top_color']:
				gridvars['bottom_color'] = random.randint(1, 9)

			inp = self.create_input(taskvars, gridvars)
			out = self.transform_input(inp.copy(), taskvars)
			train.append({'input': inp, 'output': out})

		# test example: columns and colors vary
		test_gridvars = {
			'top_color': random.randint(1, 9),
			'bottom_color': random.randint(1, 9),
			'top_col': random.randrange(0, grid_size),
			'bottom_col': random.randrange(0, grid_size)
		}
		while test_gridvars['bottom_color'] == test_gridvars['top_color']:
			test_gridvars['bottom_color'] = random.randint(1, 9)

		test_inp = self.create_input(taskvars, test_gridvars)
		test_out = self.transform_input(test_inp.copy(), taskvars)

		return taskvars, {'train': train, 'test': [{'input': test_inp, 'output': test_out}]}

	def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
		n = taskvars['grid_size']
		p = taskvars['row_position_p']

		top_row = p - 1
		bottom_row = n - p

		grid = np.zeros((n, n), dtype=int)

		top_col = gridvars.get('top_col', random.randrange(0, n))
		bottom_col = gridvars.get('bottom_col', random.randrange(0, n))

		top_color = gridvars.get('top_color', random.randint(1, 9))
		bottom_color = gridvars.get('bottom_color', random.randint(1, 9))
		if bottom_color == top_color:
			# force different
			bottom_color = (top_color % 9) + 1

		grid[top_row, top_col] = top_color
		grid[bottom_row, bottom_col] = bottom_color

		return grid

	def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
		# Identify the two colored cells
		n = grid.shape[0]
		# find non-zero coords
		coords = [(r, c) for r in range(n) for c in range(n) if grid[r, c] != 0]
		if len(coords) != 2:
			# If not exactly two colored cells, return a copy (fallback)
			out = grid.copy()
			return out

		(r1, c1), (r2, c2) = coords
		# determine top and bottom by row index
		if r1 <= r2:
			top_r, top_c = r1, c1
			bot_r, bot_c = r2, c2
		else:
			top_r, top_c = r2, c2
			bot_r, bot_c = r1, c1

		top_color = grid[top_r, top_c]
		bot_color = grid[bot_r, bot_c]

		out = grid.copy()

		# Fill first row and the row with the top colored cell entirely with top_color
		out[0, :] = top_color
		out[top_r, :] = top_color

		# Fill last row and the row with the bottom colored cell entirely with bot_color
		out[-1, :] = bot_color
		out[bot_r, :] = bot_color

		# Fill first and last columns from top down to middle with top_color, then bottom half with bot_color
		middle = (n // 2) - 1  # for even n, this is the last row of the upper half
		for r in range(0, middle + 1):
			out[r, 0] = top_color
			out[r, -1] = top_color
		for r in range(middle + 1, n):
			out[r, 0] = bot_color
			out[r, -1] = bot_color

		return out


if __name__ == '__main__':
	# Quick test and visualization
	gen = Task1bfc4729Generator()
	taskvars, data = gen.create_grids()
	print('Task vars:', taskvars)
	ARCTaskGenerator.visualize_train_test_data(data)

