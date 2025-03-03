from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Task3aa6fb7aGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain several {color('object_color')} L-shaped objects, where each L-shaped object is made of three cells and follows the form [[L, 0], [L, 0], [L, L]].",
            "In each input grid one L-shaped object remains in its original shape, while all others are rotated.",
            "Each L-shaped object is fully separated from the others by empty (0) cells."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and transforming each L-shaped object into a 2x2 block by coloring specific empty (0) cells with {color('fill_color')} color.",
            "An empty cell is filled if it is 4-way connected to two {color('object_color')} cells (one above or below and one to the left or right) and diagonally connected to one {color('object_color')} cell."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {}
        
        grid_size = random.randint(5, 30)
        object_color = random.randint(1, 9)
        fill_color = random.choice([c for c in range(1, 10) if c != object_color])
        
        taskvars['grid_size'] = grid_size
        taskvars['object_color'] = object_color
        taskvars['fill_color'] = fill_color
        
        nr_train_examples = random.randint(3, 5)
        nr_test_examples = 1
        
        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)
        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        object_color = taskvars['object_color']
        
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        num_objects = random.randint(2, min(6, (grid_size * grid_size) // 5))

        # Define the L-shape with exactly 3 cells
        L_shapes = [
            np.array([[object_color, 0], [object_color, object_color]]),  # Normal L
            np.array([[object_color, object_color], [0, object_color]]),  # Rotated 90°
            np.array([[object_color, object_color], [object_color, 0]]),  # Rotated 180°
            np.array([[0, object_color], [object_color, object_color]])   # Rotated 270°
        ]
        
        orientations = [0] + [random.randint(1, 3) for _ in range(num_objects - 1)]
        random.shuffle(orientations)

        placed_cells = set()

        def can_place(shape, top, left):
            """Check if the shape can be placed at (top, left) while maintaining full separation."""
            shape_h, shape_w = shape.shape
            for r in range(shape_h):
                for c in range(shape_w):
                    if shape[r, c] != 0:
                        rr, cc = top + r, left + c
                        if not (0 <= rr < grid_size and 0 <= cc < grid_size):
                            return False
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                nr, nc = rr + dr, cc + dc
                                if (nr, nc) in placed_cells and shape[r, c] != 0:
                                    return False
            return True

        def place_shape(shape, top, left):
            """Place the shape in the grid and mark occupied cells."""
            shape_h, shape_w = shape.shape
            for r in range(shape_h):
                for c in range(shape_w):
                    if shape[r, c] != 0:
                        rr, cc = top + r, left + c
                        grid[rr, cc] = shape[r, c]
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                placed_cells.add((rr + dr, cc + dc))

        for orientation in orientations:
            L_shape = np.rot90(L_shapes[0], k=orientation)
            shape_h, shape_w = L_shape.shape
            placed = False

            for _ in range(200):
                top = random.randint(0, grid_size - shape_h)
                left = random.randint(0, grid_size - shape_w)
                if can_place(L_shape, top, left):
                    place_shape(L_shape, top, left)
                    placed = True
                    break

            if not placed:
                break

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        grid_size = taskvars['grid_size']
        object_color = taskvars['object_color']
        fill_color = taskvars['fill_color']

        out_grid = np.copy(grid)

        for r in range(grid_size):
            for c in range(grid_size):
                if out_grid[r, c] == 0:
                    vertical_match = (r > 0 and out_grid[r - 1, c] == object_color) or \
                                     (r < grid_size - 1 and out_grid[r + 1, c] == object_color)
                    horizontal_match = (c > 0 and out_grid[r, c - 1] == object_color) or \
                                       (c < grid_size - 1 and out_grid[r, c + 1] == object_color)
                    diagonal_match = any(
                        0 <= r + dr < grid_size and 0 <= c + dc < grid_size and 
                        out_grid[r + dr, c + dc] == object_color
                        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                    )

                    if vertical_match and horizontal_match and diagonal_match:
                        out_grid[r, c] = fill_color

        return out_grid


