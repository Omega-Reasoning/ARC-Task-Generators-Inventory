from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple
# (Optional) import from transformation_library if you need transformations/detections
# But here, we only need a simple fill of interior cells, so no complicated transformations are used.
# from transformation_library import find_connected_objects, GridObject, GridObjects

# (Optional) import from input_library if you need advanced random object creation
# But here, we only need a uniform fill, so we'll just create the array directly.
# from input_library import create_object, retry, random_cell_coloring

class Task835KeL6Q7CdGEz9HhVZqmUGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        self.input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each input grid is completely filled with same-colored cells, using either {color('object_color1')} or {color('object_color2')} color."
        ]
        
        # 2) Transformation reasoning chain
        self.transformation_reasoning_chain = [
            "The output grids are created by copying the input grid and filling the interior cells with a different color based on the input grid color, while keeping the border unchanged.",
            "The interior cells are changed to {color('object_color3')} if the input grid is {color('object_color1')}, otherwise, they are changed to {color('object_color4')}."
        ]
        
        # 3) We call the super constructor (two-argument version, matching arc_task_generator.py)
        super().__init__(self.input_reasoning_chain, self.transformation_reasoning_chain)

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create a grid completely filled with a single color (either object_color1 or object_color2).
        We choose the fill color and size from gridvars.
        """
        rows = gridvars['rows']
        cols = gridvars['cols']
        fill_color = gridvars['fill_color']
        
        grid = np.full((rows, cols), fill_color, dtype=int)
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:

        """
        Transform the input grid by filling its interior with:
          - object_color3 if the input color is object_color1,
          - object_color4 otherwise.
        The border remains the original color.
        """
        color1 = taskvars["object_color1"]
        color2 = taskvars["object_color2"]
        color3 = taskvars["object_color3"]
        color4 = taskvars["object_color4"]
        
        # The entire input grid is uniform, so we can detect the 'input color' via the top-left cell.
        input_color = grid[0, 0]
        
        # Copy the grid so we don't overwrite the input
        output_grid = grid.copy()
        
        # Decide new interior color based on input color
        if input_color == color1:
            fill_color = color3
        elif input_color == color2:
            fill_color = color4
        else:
            # If somehow a color outside {color1, color2} was generated, default to color4
            fill_color = color4
        
        # Fill the interior if the grid is large enough to have an interior
        if output_grid.shape[0] > 2 and output_grid.shape[1] > 2:
            output_grid[1:-1, 1:-1] = fill_color
        
        return output_grid

    def create_grids(self) -> (dict, TrainTestData):
        """
        Creates 3 training examples and 2 test examples, ensuring:
          - All grids use distinct sizes.
          - Colors are distinct among object_color1, object_color2, object_color3, object_color4.
          - At least one train and one test grid uses object_color1, and at least one train
            and one test grid uses object_color2.
        """
        # Step 1: Pick 4 distinct colors in the range [1..9]
        colors = random.sample(range(1, 10), 4)
        color1, color2, color3, color4 = colors
        
        # Prepare the dictionary of task variables
        taskvars = {
            "object_color1": color1,
            "object_color2": color2,
            "object_color3": color3,
            "object_color4": color4
        }
        
        # Step 2: Pick 5 distinct sizes to ensure each input grid differs
        # All are between 5x5 and 30x30 to respect ARC constraints
        possible_sizes = list(range(5, 31))
        chosen_sizes = random.sample(possible_sizes, 5)
        
        # We'll create 3 training grids, 2 test grids
        train_sizes = chosen_sizes[:3]
        test_sizes  = chosen_sizes[3:]
        
        # For training, we want at least one with color1 and one with color2.
        # We'll do:
        #  Train 1 -> color1
        #  Train 2 -> color2
        #  Train 3 -> randomly color1 or color2
        train_fill_colors = [color1, color2, random.choice([color1, color2])]
        
        # For test, we want exactly 2 grids, one with color1, one with color2
        test_fill_colors = [color1, color2]
        
        train_data = []
        for i in range(3):
            gridvars = {
                "rows": train_sizes[i],
                "cols": train_sizes[i],
                "fill_color": train_fill_colors[i]
            }
            inp = self.create_input(taskvars, gridvars)
            outp = self.transform_input(inp, taskvars)
            train_data.append({
                'input': inp,
                'output': outp
            })
        
        test_data = []
        for i in range(2):
            gridvars = {
                "rows": test_sizes[i],
                "cols": test_sizes[i],
                "fill_color": test_fill_colors[i]
            }
            inp = self.create_input(taskvars, gridvars)
            outp = self.transform_input(inp, taskvars)
            test_data.append({
                'input': inp,
                'output': outp
            })
        
        # Combine into final TrainTestData
        train_test_data: TrainTestData = {
            'train': train_data,
            'test': test_data
        }
        return taskvars, train_test_data



