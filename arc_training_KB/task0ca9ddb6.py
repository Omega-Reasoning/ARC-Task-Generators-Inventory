from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple

from input_library import random_cell_coloring
from transformation_library import find_connected_objects


class Task0ca9ddb6Generator(ARCTaskGenerator):
	def __init__(self):
		input_reasoning_chain = [
			"The input grids have different sizes.",
			"Each input grid contains several colored cells of {color('cell_color1')}, {color('cell_color2')}, and some other colors, while the remaining cells are empty.",
			"The {color('cell_color1')} and {color('cell_color2')} cells appear such that each of them has a 3×3 subgrid of empty cells surrounding it.",
			"No two colored cells are connected to each other."
		]

		transformation_reasoning_chain = [
			"The output grid is created by copying the input grid and then identifying all {color('cell_color1')} and {color('cell_color2')} cells.",
			"Around each {color('cell_color1')} cell, four {color('cell_color3')} cells are added at the top-left, top-right, bottom-left, and bottom-right diagonal positions.",
			"Around each {color('cell_color2')} cell, four {color('cell_color4')} cells are added at the top, bottom, left, and right positions.",
			"The other colored cells do not receive any additional surrounding pattern."
		]

		super().__init__(input_reasoning_chain, transformation_reasoning_chain)

	def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
		# Initialize task variables
		taskvars: Dict[str, Any] = {
			# grid_size will be fixed across train/test examples for clarity but
			# create_input randomises actual sizes by choosing from this base when desired
			'grid_size': random.randint(5, 12),
			'cell_color1': random.randint(1, 9),
			'cell_color2': None,
			'cell_color3': None,
			'cell_color4': None
		}

		# choose distinct non-zero colors
		while taskvars['cell_color2'] is None or taskvars['cell_color2'] == taskvars['cell_color1']:
			taskvars['cell_color2'] = random.randint(1, 9)

		# pick colors for added patterns distinct from the above
		pool = [c for c in range(1, 10) if c not in (taskvars['cell_color1'], taskvars['cell_color2'])]
		taskvars['cell_color3'] = random.choice(pool)
		pool.remove(taskvars['cell_color3'])
		taskvars['cell_color4'] = random.choice(pool)

		# Create 3-5 train examples and 1 test example
		num_train = random.randint(3, 5)
		train_test_data = self.create_grids_default(num_train, 1, taskvars)

		# Ensure at least one train and each test example contains at least one
		# "other" colored cell (not c1 or c2). Some random generations may
		# accidentally produce grids containing only the special colors; in
		# that case we inject a single other-colored cell into one example.
		def _ensure_other_color(example):
			grid = example['input']
			# if grid already contains an other color, nothing to do
			c1 = taskvars['cell_color1']
			c2 = taskvars['cell_color2']
			other_colors = [c for c in range(1, 10) if c not in (c1, c2)]
			if any(((grid == c) .any()) for c in other_colors):
				return False

			h, w = grid.shape
			# build forbidden mask around existing centers so we don't place inside reserved areas
			forbidden = np.zeros_like(grid, dtype=bool)
			centers = list(zip(*np.where((grid == c1) | (grid == c2))))
			for (r, c) in centers:
				for rr in range(max(0, r-1), min(h, r+2)):
					for cc in range(max(0, c-1), min(w, c+2)):
						forbidden[rr, cc] = True
					for dr in (-1, -2, 1, 2):
						rr = r + dr
						if 0 <= rr < h:
							forbidden[rr, c] = True
					for dc in (-1, -2, 1, 2):
						cc = c + dc
						if 0 <= cc < w:
							forbidden[r, cc] = True

			# find a candidate empty cell not in forbidden and not adjacent (4-neigh) to other non-zero
			candidates = []
			for r in range(h):
				for c in range(w):
					if grid[r, c] != 0 or forbidden[r, c]:
						continue
					# ensure 4-neighbors are empty
					ok = True
					for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
						rr, cc = r+dr, c+dc
						if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] != 0:
							ok = False
							break
					if ok:
						candidates.append((r,c))

			if not candidates:
				# give up if no safe place; return False to indicate no change
				return False

			r, c = random.choice(candidates)
			grid[r, c] = random.choice(other_colors)
			# update corresponding output
			example['output'] = self.transform_input(grid, taskvars)
			return True

		# Ensure at least one training example contains an other color
		if not any(any(((ex['input'] == c).any()) for c in [x for x in range(1,10) if x not in (taskvars['cell_color1'], taskvars['cell_color2'])]) for ex in train_test_data['train']):
			# try to inject into a random train example
			_random_idx = random.randrange(len(train_test_data['train']))
			_ensured = _ensure_other_color(train_test_data['train'][_random_idx])
			# if failed, try others
			if not _ensured:
				for i, ex in enumerate(train_test_data['train']):
					if _ensure_other_color(ex):
						break

		# Ensure each test example contains an other color (typically only one)
		for ex in train_test_data['test']:
			_ensure_other_color(ex)

		return taskvars, train_test_data

	def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
		"""Create a single input grid satisfying the input reasoning chain.

		Important constraints enforced:
		- grid size between 5 and 30 (we use taskvars['grid_size'] as base)
		- at least one cell of cell_color1 and one of cell_color2
		- every cell of those two colors has its 8-neighbours empty (3x3 empty surrounding)
		- no two colored (non-zero) cells are 4-connected
		"""
		base_size = int(taskvars.get('grid_size', 8))
		# Slightly randomise actual grid size to increase variety
		h = w = random.randint(max(5, base_size - 2), min(30, base_size + 3))

		grid = np.zeros((h, w), dtype=int)

		# forbidden mask: marks cells that are part of an existing center's reserved
		# region (3x3 block and the two-cell cardinal directions). When True,
		# these cells cannot be used as part of another center's 3x3 empty
		# boundary — this guarantees the reserved areas are disjoint.
		forbidden = np.zeros_like(grid, dtype=bool)

		c1 = taskvars['cell_color1']
		c2 = taskvars['cell_color2']
		other_colors = [c for c in range(1, 10) if c not in (c1, c2)]

		# helper to check placement: require 3x3 empty neighborhood AND
		# two empty rows/cols in cardinal directions (positions at distance 1 and 2)
		def can_place_center(r, c):
			# must be at least 2 cells away from all borders to allow 2-row cardinal empties
			if not (2 <= r < h-2 and 2 <= c < w-2):
				return False
			# check the 3x3 block is empty
			for rr in range(r-1, r+2):
				for cc in range(c-1, c+2):
					# must be physically empty and not reserved by another center
					if grid[rr, cc] != 0 or forbidden[rr, cc]:
						return False
			# check two-cell cardinal empties (up/down/left/right distances 1 and 2)
			for dr in (-1, -2, 1, 2):
				rr = r + dr
				if 0 <= rr < h and grid[rr, c] != 0:
					return False
			for dc in (-1, -2, 1, 2):
				cc = c + dc
				if 0 <= cc < w and grid[r, cc] != 0:
					return False
			# Also ensure the two-cell cardinal direction cells aren't reserved
			for dr in (-1, -2, 1, 2):
				rr = r + dr
				if 0 <= rr < h and forbidden[rr, c]:
					return False
			for dc in (-1, -2, 1, 2):
				cc = c + dc
				if 0 <= cc < w and forbidden[r, cc]:
					return False
			return True

		# Place at least one of each special color, and optionally more
		placed_c1 = 0
		placed_c2 = 0

		# Attempt to place 1-3 of each, but ensure constraints
		targets_c1 = random.randint(1, min(3, (h*w)//20 + 1))
		targets_c2 = random.randint(1, min(3, (h*w)//20 + 1))

		attempts = 0
		max_attempts = 500
		while (placed_c1 < targets_c1 or placed_c2 < targets_c2) and attempts < max_attempts:
			attempts += 1
			r = random.randint(1, h-2)
			c = random.randint(1, w-2)
			# randomly decide whether to place c1 or c2 if both still needed
			want = None
			if placed_c1 < targets_c1 and placed_c2 < targets_c2:
				want = random.choice([c1, c2])
			elif placed_c1 < targets_c1:
				want = c1
			elif placed_c2 < targets_c2:
				want = c2
			else:
				break

			if can_place_center(r, c):
				grid[r, c] = want
				# mark the 3x3 neighbourhood and two-cell cardinal directions as reserved
				# so no later center can reuse any of these cells (ensures disjoint reserved areas)
				for rr in range(max(0, r-1), min(h, r+2)):
					for cc in range(max(0, c-1), min(w, c+2)):
						forbidden[rr, cc] = True
				for dr in (-1, -2, 1, 2):
					rr = r + dr
					if 0 <= rr < h:
						forbidden[rr, c] = True
				for dc in (-1, -2, 1, 2):
					cc = c + dc
					if 0 <= cc < w:
						forbidden[r, cc] = True
				if want == c1:
					placed_c1 += 1
				else:
					placed_c2 += 1

		# If failed to place both at least once (very unlikely), force placements at deterministic positions
		if placed_c1 == 0:
			for r in range(1, h-1):
				for c in range(1, w-1):
					if can_place_center(r, c):
						grid[r, c] = c1
						placed_c1 += 1
						# mark forbidden for forced placement as well
						for rr in range(max(0, r-1), min(h, r+2)):
							for cc in range(max(0, c-1), min(w, c+2)):
								forbidden[rr, cc] = True
							for dr in (-1, -2, 1, 2):
								rr = r + dr
								if 0 <= rr < h:
									forbidden[rr, c] = True
							for dc in (-1, -2, 1, 2):
								cc = c + dc
								if 0 <= cc < w:
									forbidden[r, cc] = True
						break
				if placed_c1:
					break

		if placed_c2 == 0:
			for r in range(h-2, 0, -1):
				for c in range(w-2, 0, -1):
					if can_place_center(r, c):
						grid[r, c] = c2
						placed_c2 += 1
						# mark forbidden for forced placement as well
						for rr in range(max(0, r-1), min(h, r+2)):
							for cc in range(max(0, c-1), min(w, c+2)):
								forbidden[rr, cc] = True
							for dr in (-1, -2, 1, 2):
								rr = r + dr
								if 0 <= rr < h:
									forbidden[rr, c] = True
							for dc in (-1, -2, 1, 2):
								cc = c + dc
								if 0 <= cc < w:
									forbidden[r, cc] = True
						break
				if placed_c2:
					break


		# Decide total number of colored cells in the grid (inclusive of c1 and c2)
		min_total = max(2, placed_c1 + placed_c2)
		# User requested: max number of colored cells = number of rows // 2
		cap = max(2, h // 2)
		max_total = min(9, cap)
		# Ensure max_total is at least min_total to avoid randint errors
		if max_total < min_total:
			max_total = min_total
		total_colored = random.randint(min_total, max_total)

		# Candidate positions are those where the full 3x3 neighborhood is empty
		# and also the two-cell cardinal empties are available (use can_place_center)
		empty_positions = [(r, c) for r in range(2, h-2) for c in range(2, w-2)
			if grid[r, c] == 0 and not forbidden[r, c] and can_place_center(r, c)]
		random.shuffle(empty_positions)

		# Place additional colored cells until we reach total_colored or run out of candidates
		current_count = int((grid != 0).sum())
		idx = 0
		while current_count < total_colored and idx < len(empty_positions):
			r, c = empty_positions[idx]
			# double-check placement still valid (cells might have been placed earlier in loop)
			if can_place_center(r, c) and grid[r, c] == 0 and not forbidden[r, c]:
				grid[r, c] = random.choice(other_colors)
				current_count += 1
			idx += 1

		# Final safety check: ensure no two non-zero cells are 4-connected; if found, remove extras not in forbidden zones
		objs = find_connected_objects(grid, diagonal_connectivity=False, background=0)
		for obj in objs:
			if len(obj) > 1:
				cells = list(obj.coords)
				# keep the first cell, clear others if they are not special-centre neighbouring
				for (rr, cc) in cells[1:]:
					if not forbidden[rr, cc]:
						grid[rr, cc] = 0

		return grid

	def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
		"""Apply the transformation chain to produce the output grid.

		For every cell of color cell_color1 add diagonally adjacent cells of color cell_color3.
		For every cell of color cell_color2 add orthogonally adjacent cells of color cell_color4.
		"""
		out = grid.copy()
		h, w = grid.shape
		c1 = taskvars['cell_color1']
		c2 = taskvars['cell_color2']
		c3 = taskvars['cell_color3']
		c4 = taskvars['cell_color4']

		# Find centers
		centers_c1 = list(zip(*np.where(grid == c1)))
		centers_c2 = list(zip(*np.where(grid == c2)))

		# For c1: diagonals
		for (r, c) in centers_c1:
			for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
				rr, cc = r + dr, c + dc
				if 0 <= rr < h and 0 <= cc < w:
					out[rr, cc] = c3

		# For c2: orthogonals
		for (r, c) in centers_c2:
			for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
				rr, cc = r + dr, c + dc
				if 0 <= rr < h and 0 <= cc < w:
					out[rr, cc] = c4

		return out


if __name__ == '__main__':
	# Quick manual test to visualize generated examples
	gen = Task0ca9ddb6Generator()
	taskvars, data = gen.create_grids()
	print('taskvars:', taskvars)
	ARCTaskGenerator.visualize_train_test_data(data)

