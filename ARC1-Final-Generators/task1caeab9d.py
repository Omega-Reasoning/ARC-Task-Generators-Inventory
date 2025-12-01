from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry
from transformation_library import find_connected_objects
import numpy as np
import random

class Task1caeab9dGenerator(ARCTaskGenerator):
	def __init__(self):
		input_reasoning_chain = [
			"The input grid has size {vars['rows']} X {vars['cols']}.",
			"The input grid consists of exactly three objects which are either a square or rectangle and they have the same dimensions but different colors.",
			"The dimensions of the square can be 2x2 or 3x3.",
			"The dimension of the rectangle can be one of these 1x2, 2x1, 3x2.",
			"The objects are placed in the grid such that they do not overlap each other.",
			"The First object is colored {color('color_1')}, the second with {color('color_2')} and the third with {color('color_3')}."
		]
		
		transformation_reasoning_chain = [
			"The output grid has the same size as the input grid.",
			"Copy the input grid to the output grid.",
			"First identify all the three objects.",
			"All the three objects should be aligned, i.e., the objects with color {color('color_2')} and {color('color_3')} should be placed at same row position of the subgrid which has object of {color('color_3')}."
		]

		super().__init__(input_reasoning_chain, transformation_reasoning_chain)

	def create_input(self, taskvars, gridvars):
		rows = taskvars['rows']
		cols = taskvars['cols']
		color_1 = taskvars['color_1']
		color_2 = taskvars['color_2']
		color_3 = taskvars['color_3']

		# Ensure grid is wide enough for non-overlapping columns
		if cols < 3 * 3:  # Minimum width needed for three 3-wide objects
			raise ValueError("Grid must be at least 9 columns wide")

		grid = np.zeros((rows, cols), dtype=int)

		# First decide if we're using squares or rectangles
		shape_type = random.choice(['square', 'rectangle'])
	
		# Choose a single size for all three objects
		if shape_type == 'square':
			size = random.choice([(2, 2), (3, 3)])
		else:  # rectangle
			size = random.choice([(2, 3), (3, 2), (1, 2), (2, 1)])

		# Calculate the width needed for each section to avoid column overlap
		object_width = size[1]
		section_width = cols // 3
	
		# Create three objects with the same size but different colors
		colors = [color_1, color_2, color_3]
		for i, color in enumerate(colors):
			placed = False
			max_attempts = 100
			attempts = 0
		
			# Calculate the valid column range for this object
			section_start = i * section_width
			section_end = section_start + section_width - object_width
		
			while not placed and attempts < max_attempts:
				# Try to place the object at a random position
				start_r = random.randint(0, rows - size[0])
				start_c = random.randint(section_start, max(section_start, section_end))
			
				# Check if the position is valid (no overlap)
				valid = True
				for r in range(size[0]):
					for c in range(size[1]):
						if grid[start_r + r, start_c + c] != 0:
							valid = False
							break
					if not valid:
						break
			
				# If position is valid, place the object
				if valid:
					for r in range(size[0]):
						for c in range(size[1]):
							grid[start_r + r, start_c + c] = color
					placed = True
			
				attempts += 1
		
			if not placed:
				raise ValueError(f"Could not place object after {max_attempts} attempts")

		return grid


	def transform_input(self, grid, taskvars):
		# Find all objects in the grid
		objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)

		# Map colors to objects
		color_map = {
			taskvars['color_1']: None,
			taskvars['color_2']: None,
			taskvars['color_3']: None
		}

		# Identify each object by its color
		for obj in objects:
			if obj.is_monochromatic:
				color = next(iter(obj.colors))
				if color in color_map:
					color_map[color] = obj

		if not all(color_map.values()):
			raise ValueError("Not all objects with the specified colors were found")

		# Get the row position of color_2 object (reference object)
		color_2_obj = color_map[taskvars['color_2']]
		ref_row = min(r for r, c in color_2_obj.coords)  # Get the topmost row of color_2 object

		# Create output grid
		output_grid = np.zeros_like(grid)

		# First place the reference object (color_2) unchanged
		color_2_obj.paste(output_grid)

		# Move color_1 and color_3 objects to align with color_2's row
		for color in [taskvars['color_1'], taskvars['color_3']]:
			obj = color_map[color]
			current_row = min(r for r, c in obj.coords)  # Get the topmost row of current object
		
			# Calculate how much to move the object
			row_shift = ref_row - current_row
		
			# Translate the object
			obj.translate(dx=row_shift, dy=0, grid_shape=grid.shape)
			obj.paste(output_grid)

		return output_grid

	def create_grids(self):
		rows = random.randint(12, 20)
		cols = random.randint(12, 20)
		colors = random.sample(range(1, 10), 3)

		taskvars = {
			'rows': rows,
			'cols': cols,
			'color_1': colors[0],
			'color_2': colors[1],
			'color_3': colors[2]
		}

		train_data = []
		for _ in range(random.randint(3, 4)):
			input_grid = self.create_input(taskvars, {})
			output_grid = self.transform_input(input_grid, taskvars)
			train_data.append(GridPair(input=input_grid, output=output_grid))

		test_input = self.create_input(taskvars, {})
		test_output = self.transform_input(test_input, taskvars)

		test_data = [GridPair(input=test_input, output=test_output)]

		return taskvars, TrainTestData(train=train_data, test=test_data)