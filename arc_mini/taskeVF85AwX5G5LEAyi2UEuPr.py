import numpy as np
import random

# Required imports from the provided libraries
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject
from input_library import create_object, Contiguity, retry

class TaskeVF85AwX5G5LEAyi2UEuPrGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Define the input reasoning chain
        self.input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain several {color('object_color1')} and {color('object_color2')} objects, along with empty cells."
        ]
        
        # 2) Define the transformation reasoning chain
        self.transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and changing the color of a {color('object_color1')} object to  {color('object_color2')}, only if the {color('object_color1')} object is 4-way connected to a {color('object_color2')} object.",
            "All other cells remain unchanged."]
        
        # 3) Call the base class constructor
        super().__init__(self.input_reasoning_chain, self.transformation_reasoning_chain)
    
    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create an input grid ensuring:
         - Each object contains at least two cells
         - At least one {object_color1} object is 4-way connected to a {object_color2} object
         - At least one isolated {object_color1} and {object_color2} object
         - No diagonal connectivity
        """

        color1 = taskvars['object_color1']
        color2 = taskvars['object_color2']

        def generator():
            height = random.randint(5, 15)
            width = random.randint(5, 15)
            grid = np.zeros((height, width), dtype=int)

            def place_object(grid, color, min_size=2):
                while True:
                    obj = create_object(
                        height=random.randint(1, max(1, height // 2)),
                        width=random.randint(1, max(1, width // 2)),
                        color_palette=color,
                        contiguity=Contiguity.FOUR,
                        background=0
                    )
                    if np.count_nonzero(obj) >= min_size:
                        break
                r_off = random.randint(0, height - obj.shape[0])
                c_off = random.randint(0, width - obj.shape[1])
                for r in range(obj.shape[0]):
                    for c in range(obj.shape[1]):
                        if obj[r, c] != 0:
                            grid[r_off + r, c_off + c] = obj[r, c]

            nb_objects_color1 = random.randint(2, 4)
            nb_objects_color2 = random.randint(2, 4)
            for _ in range(nb_objects_color1):
                place_object(grid, color1)
            for _ in range(nb_objects_color2):
                place_object(grid, color2)
            return grid

        def valid_constraints(grid: np.ndarray) -> bool:
            color1_objs = []
            color2_objs = []
            objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)
            for obj in objects:
                if obj.has_color(taskvars['object_color1']):
                    color1_objs.append(obj)
                elif obj.has_color(taskvars['object_color2']):
                    color2_objs.append(obj)
            if not color1_objs or not color2_objs:
                return False
            def are_4way_adjacent(o1: GridObject, o2: GridObject) -> bool:
                return o1.touches(o2, diag=False)
            cond1 = any(are_4way_adjacent(o1, o2) for o1 in color1_objs for o2 in color2_objs)
            if not cond1:
                return False
            cond2 = any(not any(are_4way_adjacent(o1, o2) for o2 in color2_objs) for o1 in color1_objs)
            cond3 = any(not any(are_4way_adjacent(o2, o1) for o1 in color1_objs) for o2 in color2_objs)
            return cond2 and cond3

        return retry(generator, valid_constraints, max_attempts=200)
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        color1 = taskvars['object_color1']
        color2 = taskvars['object_color2']
        output_grid = grid.copy()
        objects = find_connected_objects(output_grid, diagonal_connectivity=False, background=0, monochromatic=True)
        for obj in objects:
            if obj.has_color(color1) and any(obj.touches(o, diag=False) and o.has_color(color2) for o in objects):
                for (r, c, _) in obj.cells:
                    output_grid[r, c] = color2
        return output_grid

    def create_grids(self) -> (dict, TrainTestData):
        color1, color2 = random.sample(range(1, 10), 2)
        taskvars = {'object_color1': color1, 'object_color2': color2}
        nr_train = random.randint(3, 6)
        return taskvars, self.create_grids_default(nr_train, 1, taskvars)


