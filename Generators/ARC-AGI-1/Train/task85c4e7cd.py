from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from Framework.transformation_library import find_connected_objects
from Framework.input_library import random_cell_coloring


class Task85c4e7cdGenerator(ARCTaskGenerator):
	def __init__(self):
		input_reasoning_chain = [
			"The input grids are square and are completely filled with several nested rectangular colored rings.",
			"Each ring is a one cell wide, closed loop of a single color, and all rings together form a layered pattern from the outer border toward the center.",
			"The number of rings and their colors vary across examples, with one ring always being of {color('ring')} color,",
			"The rings do not overlap, and no empty (0) cells appear anywhere in the input.",
			"The colors of the rings may be repeated but no two consecutive rings get the same color."
		]

		transformation_reasoning_chain = [
			"The output grid has the same size and ring structure as the input grid.",
			"Only the colors of the rings change, and the sequence is reversed. The innermost ring in the output is recolored with the outermost ring color from the input, and each outer ring receives the color of the next inner ring. The outermost ring in the output therefore gets the innermost input color. All colors are reversed exactly as they appear in the input."
		]

		super().__init__(input_reasoning_chain, transformation_reasoning_chain)

	def create_grids(self) -> tuple[dict, TrainTestData]:
		"""
		Creates 3-5 train examples and 1 test example. Each example is a square grid
		with odd size between 5 and 30. The number of rings equals the number of
		available one-cell-wide layers for the chosen size (i.e., (size+1)//2).

		The test example is guaranteed to have a different number of rings (hence
		different size) from all train examples.
		"""
		nr_train = random.randint(3, 5)

		# choose a task-wide ring color and expose it as task variable 'ring'
		# (this color must appear in every example at least once)
		ring_color = random.choice(list(range(1, 10)))

		train = []
		train_sizes = set()
		for _ in range(nr_train):
			# choose odd grid size between 5 and 29 or 30 but odd -> 5..29 or 5..31? keep 5..29 inclusive
			size = random.choice(list(range(5, 31, 2)))
			train_sizes.add(size)
			grid = self.create_input({'ring': int(ring_color)}, {'grid_size': size})
			out = self.transform_input(grid.copy(), {})
			train.append({'input': grid, 'output': out})

		# choose test size (odd) different from all train sizes
		possible_sizes = [s for s in range(5, 31, 2) if s not in train_sizes]
		if not possible_sizes:
			# unlikely, but fallback: change one train example size to free up a size
			possible_sizes = [s for s in range(5, 31, 2)]
		test_size = random.choice(possible_sizes)
		test_grid = self.create_input({'ring': int(ring_color)}, {'grid_size': test_size})
		test_out = self.transform_input(test_grid.copy(), {})

		taskvars = {
			'ring': int(ring_color)
		}

		return taskvars, {'train': train, 'test': [{'input': test_grid, 'output': test_out}]}

	def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
		"""
		Build a square grid containing nested one-cell-wide rectangular rings.

		gridvars['grid_size'] may be provided; otherwise choose an odd size 5..29.
		All cells are non-zero. Rings colors are integers 1..9; consecutive rings
		are guaranteed to have different colors.
		"""
		size = gridvars.get('grid_size') or random.choice(list(range(5, 31, 2)))
		grid = np.zeros((size, size), dtype=int)

		# number of one-cell-wide layers (rings)
		n_rings = (size + 1) // 2

		# choose colors for each ring, allow repeats but not consecutive equal
		# ensure the task-wide ring color (if provided) appears at least once
		forced_color = None
		if taskvars and 'ring' in taskvars:
			forced_color = int(taskvars['ring'])

		# attempt generation until forced_color is included (or give up after tries)
		attempts = 0
		while True:
			colors = []
			for _ in range(n_rings):
				choices = list(range(1, 10))
				if colors:
					prev = colors[-1]
					if prev in choices:
						choices.remove(prev)
				colors.append(random.choice(choices))
			# if no forced color required or we succeeded, break
			if (forced_color is None) or (forced_color in colors):
				break
			attempts += 1
			if attempts > 20:
				# force inclusion by replacing a random layer, fixing neighbors if needed
				idx = random.randrange(n_rings)
				colors[idx] = forced_color
				# fix left neighbor if equal
				if idx - 1 >= 0 and colors[idx - 1] == colors[idx]:
					for c in range(1, 10):
						if c != colors[idx] and (idx - 2 < 0 or colors[idx - 2] != c):
							colors[idx - 1] = c
							break
				# fix right neighbor if equal
				if idx + 1 < n_rings and colors[idx + 1] == colors[idx]:
					for c in range(1, 10):
						if c != colors[idx] and (idx + 2 >= n_rings or colors[idx + 2] != c):
							colors[idx + 1] = c
							break
				break

		# fill rings from outer to inner
		for layer in range(n_rings):
			col = colors[layer]
			r0 = layer
			c0 = layer
			r1 = size - 1 - layer
			c1 = size - 1 - layer
			# top and bottom rows of ring
			grid[r0, c0:c1 + 1] = col
			grid[r1, c0:c1 + 1] = col
			# left and right columns
			grid[r0:r1 + 1, c0] = col
			grid[r0:r1 + 1, c1] = col

		# ensure no zeros remain (shouldn't) and return
		assert (grid != 0).all(), "Input grid must contain no zero cells"
		return grid

	def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
		"""
		Reverse the sequence of ring colors while keeping structure identical.
		"""
		out = grid.copy()
		size = grid.shape[0]
		n_rings = (size + 1) // 2

		# collect colors from outer to inner
		ring_colors = []
		for layer in range(n_rings):
			r0 = layer
			c0 = layer
			r1 = size - 1 - layer
			c1 = size - 1 - layer
			# pick one representative cell from the ring (top-left corner of ring)
			rep = grid[r0, c0]
			ring_colors.append(int(rep))

		# reversed sequence
		rev = list(reversed(ring_colors))

		# assign reversed colors back to rings
		for layer in range(n_rings):
			col = rev[layer]
			r0 = layer
			c0 = layer
			r1 = size - 1 - layer
			c1 = size - 1 - layer
			out[r0, c0:c1 + 1] = col
			out[r1, c0:c1 + 1] = col
			out[r0:r1 + 1, c0] = col
			out[r0:r1 + 1, c1] = col

		return out



