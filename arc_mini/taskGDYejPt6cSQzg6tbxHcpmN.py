from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject
from input_library import create_object, retry, Contiguity
import numpy as np
import random

class TaskGDYejPt6cSQzg6tbxHcpmNGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Set the input reasoning chain
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain exactly two objects of {color('object_color')} color, with one object consisting of three cells and the other consisting of four cells."
        ]
        
        # 2) Set the transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and changing only the color of the four-cell object, from {color('object_color')} to {color('changed_color')}."
        ]
        
        # 3) Call super().__init__()
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        """
        Initialize variables, create training/test grids, and ensure variety.
        """
        # Step 1: Pick the colors
        object_color = random.randint(1, 9)
        changed_color = random.choice([c for c in range(1, 10) if c != object_color])
        
        taskvars = {
            "object_color": object_color,
            "changed_color": changed_color
        }
        
        # Step 2: Number of training examples
        nr_train = random.randint(3, 6)
        nr_test = 1
        
        # Generate training and test grids
        data = self.create_grids_default(nr_train, nr_test, taskvars)
        
        return taskvars, data
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """
        Create an input grid ensuring:
        - Random size (5..30 in each dimension)
        - Exactly two distinct objects of color `object_color`
          * One has exactly 3 cells
          * The other has exactly 4 cells
        - Objects do not overlap and remain separated by empty cells.
        """
        object_color = taskvars['object_color']
        
        rows = random.randint(5, 12)  # Keep within range to avoid tight spaces
        cols = random.randint(5, 12)

        def generate_grid():
            grid = np.zeros((rows, cols), dtype=int)

            # Generate the 3-cell object
            obj3 = retry(
                generator=lambda: create_object(
                    height=random.randint(2, 3),
                    width=random.randint(2, 3),
                    color_palette=object_color,
                    contiguity=Contiguity.FOUR,
                    background=0
                ),
                predicate=lambda m: np.count_nonzero(m) == 3
            )

            # Generate the 4-cell object
            obj4 = retry(
                generator=lambda: create_object(
                    height=random.randint(2, 4),
                    width=random.randint(2, 4),
                    color_palette=object_color,
                    contiguity=Contiguity.FOUR,
                    background=0
                ),
                predicate=lambda m: np.count_nonzero(m) == 4
            )

            go3 = GridObject.from_array(obj3, (0, 0))
            go4 = GridObject.from_array(obj4, (0, 0))

            # Try to place them without overlap and ensure separation
            for _ in range(50):
                # Pick random positions within bounds
                r3 = random.randint(0, rows - go3.height)
                c3 = random.randint(0, cols - go3.width)

                r4 = random.randint(0, rows - go4.height)
                c4 = random.randint(0, cols - go4.width)

                def shift_cells(gobj, dr, dc):
                    shifted = set()
                    for (r, c, col) in gobj.cells:
                        new_r, new_c = r + dr, c + dc
                        if 0 <= new_r < rows and 0 <= new_c < cols:
                            shifted.add((new_r, new_c, col))
                        else:
                            return None
                    return GridObject(shifted)

                placed_go3 = shift_cells(go3, r3, c3)
                placed_go4 = shift_cells(go4, r4, c4)

                if placed_go3 and placed_go4:
                    coords3 = placed_go3.coords
                    coords4 = placed_go4.coords

                    # Ensure objects do NOT touch (add 1-cell buffer)
                    if all(abs(r1 - r2) > 1 or abs(c1 - c2) > 1 for (r1, c1) in coords3 for (r2, c2) in coords4):
                        # Place objects on grid
                        for (r, c, col) in placed_go3.cells:
                            grid[r, c] = col
                        for (r, c, col) in placed_go4.cells:
                            grid[r, c] = col
                        return grid

            return None

        # Ensure we always get a valid grid
        final_grid = retry(generator=generate_grid, predicate=lambda x: x is not None)
        
        return final_grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """
        Transformation:
          - Find the four-cell object in `object_color`
          - Change its color to `changed_color`
        """
        object_color = taskvars['object_color']
        changed_color = taskvars['changed_color']

        output_grid = grid.copy()

        objs = find_connected_objects(
            output_grid, diagonal_connectivity=False, background=0, monochromatic=True
        )

        objs_same_color = objs.filter(lambda o: o.is_monochromatic and object_color in o.colors)

        four_cell_objs = objs_same_color.filter(lambda o: len(o) == 4)

        for obj4 in four_cell_objs:
            for (r, c, _) in obj4.cells:
                output_grid[r, c] = changed_color

        return output_grid



