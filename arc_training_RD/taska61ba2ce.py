from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, BorderBehavior
from input_library import Contiguity, create_object

class Taska61ba2ceGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square grids of size {vars['grid_size']} x {vars['grid_size']}.",
            "The grid consists of an L shape which is specifically 2x2 size.",
            "There exist exactly 4 L shaped objects, but you must note that all these are rotated 90 degrees and are placed randomly across the grid.",
            "None of the rotations must repeat in the grid.",
            "Each L shaped object rotation has a distinct color."
        ]

        transformation_reasoning_chain = [
            "The output grid is a 4x4 grid.", 
            "There are empty (0) cells of size 2x2 exactly in the center of the grid.",
            "The corners of the grid are filled with the L shaped rotations forming a square shape.",
            "The output grid looks like the corners being filled with L shaped rotations from the input grid and a void of 2x2 filled with empty cells in the middle."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        num_train_pairs = random.randint(3, 5)
        train_pairs = []

        # Choose a random grid size
        grid_size = random.randint(5, 20)

        taskvars = {
            'grid_size': grid_size,  # <-- integer for logic
        }

        # Replace grid_size placeholders in reasoning chains
        self.input_reasoning_chain = [
            chain.replace("{vars['grid_size']} x {vars['grid_size']}", f"{taskvars['grid_size']} x {taskvars['grid_size']}")
            for chain in self.input_reasoning_chain
        ]
        
        # Create train and test data
        for _ in range(num_train_pairs):
            input_grid = self.create_input(taskvars)
            output_grid = self.transform_input(input_grid.copy(), taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Create test pair
        test_input = self.create_input(taskvars)
        test_output = self.transform_input(test_input.copy(), taskvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)

    def create_input(self, taskvars):
        # Get grid size from taskvars
        grid_size = taskvars['grid_size']
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Randomly select 4 different colors for each grid
        available_colors = list(range(1, 10))
        object_colors = random.sample(available_colors, 4)
        
        # Create the 4 L-shaped objects with different rotations
        l_shapes = []
        
        # Base L shape (2x2)
        base_l = np.array([
            [1, 0],
            [1, 1]
        ])
        
        # Generate all 4 rotations
        l_shapes.append(base_l)                    # Original L
        l_shapes.append(np.rot90(base_l, k=1))     # Rotated 90 degrees
        l_shapes.append(np.rot90(base_l, k=2))     # Rotated 180 degrees
        l_shapes.append(np.rot90(base_l, k=3))     # Rotated 270 degrees
        
        # Place each L shape with its assigned color at random positions
        for l_shape, color in zip(l_shapes, object_colors):
            while True:
                # Find a random position where the L shape fits
                r = random.randint(0, grid_size - l_shape.shape[0])
                c = random.randint(0, grid_size - l_shape.shape[1])
                
                # Check if the position is empty
                region = grid[r:r+l_shape.shape[0], c:c+l_shape.shape[1]]
                if np.all(region[l_shape > 0] == 0):
                    # Place the L shape
                    temp_region = region.copy()
                    temp_region[l_shape > 0] = color
                    grid[r:r+l_shape.shape[0], c:c+l_shape.shape[1]] = temp_region
                    break
        
        return grid

    def transform_input(self, input_grid, taskvars):
        # Output grid is always 4x4
        output_grid = np.zeros((4, 4), dtype=int)
        
        # Find all L shapes in the input grid
        objects = find_connected_objects(input_grid, diagonal_connectivity=False, background=0)
        
        # Map each L shape to its position in the output grid
        # Positions are: top-left, top-right, bottom-left, bottom-right corners
        positions = [
            [(0, 0), (0, 1), (1, 0)],    # Top-left L shape
            [(0, 2), (0, 3), (1, 3)],    # Top-right L shape
            [(2, 0), (3, 0), (3, 1)],    # Bottom-left L shape
            [(2, 3), (3, 2), (3, 3)]     # Bottom-right L shape
        ]
        
        # Sort objects by their rotation (based on their shape pattern)
        # We need to identify which L rotation corresponds to which position
        def get_rotation_index(obj):
            obj_array = obj.to_array()
            # The pattern of cells in the object determines its rotation
            if obj_array.shape == (2, 2):
                cell_pattern = tuple((r, c) for r in range(2) for c in range(2) if obj_array[r, c] != 0)
                patterns = [
                    ((0, 0), (1, 0), (1, 1)),  # Original L shape
                    ((0, 0), (0, 1), (1, 0)),  # Rotated 90 degrees
                    ((0, 0), (0, 1), (1, 1)),  # Rotated 180 degrees
                    ((0, 1), (1, 0), (1, 1))   # Rotated 270 degrees
                ]
                for i, pattern in enumerate(patterns):
                    if set(cell_pattern) == set(pattern):
                        return i
            return -1  # Default if no match
        
        rotation_to_position = {
            0: 0,  # Original L -> top-left
            1: 1,  # Rotated 90 degrees -> top-right
            2: 3,  # Rotated 180 degrees -> bottom-right
            3: 2   # Rotated 270 degrees -> bottom-left
        }
        
        # Place each L shape in its corresponding position
        for obj in objects.objects:
            rotation_idx = get_rotation_index(obj)
            if rotation_idx != -1:
                position_idx = rotation_to_position[rotation_idx]
                color = list(obj.colors)[0]  # L shapes are monochromatic
                
                # Place the L shape in its position on the output grid
                for r, c in positions[position_idx]:
                    output_grid[r, c] = color
        
        return output_grid