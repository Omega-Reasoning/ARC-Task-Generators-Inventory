import numpy as np
import random
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, Contiguity
from transformation_library import find_connected_objects

class ARCTask2013d3e2Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['rows']}.",
            "A single 8-way connected object is placed in the input grid.",
            "The object is constructed to have even and equal dimensions (square shape).",
            "The cells in the object can have color (between 1-9).",
            "The remaining cells of the grid are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has a different size than the input grid.",
            "First identify the 8-way connected object and the subgrid which contains it.",
            "The first quadrant of the above subgrid is the output grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        rows = taskvars['rows']
        grid = np.zeros((rows, rows), dtype=int)
        
        size = random.randint(rows//8, rows// 4) * 2
        # Create a random 8-way connected object with square dimensions
        while True:
            object_matrix = create_object(
                height=size,
                width=size,  # Same as height to ensure square
                color_palette=[1,2,3,4,5,6,7,8,9],
                contiguity=Contiguity.EIGHT
            )
            
            # Check if each row and column has at least one non-zero cell
            row_has_cell = np.any(object_matrix > 0, axis=1)
            col_has_cell = np.any(object_matrix > 0, axis=0)
            
            if np.all(row_has_cell) and np.all(col_has_cell):
                break
        
        # Randomly place the object ensuring at least 1 cell offset from edges
        max_row_offset = rows - object_matrix.shape[0] - 1
        max_col_offset = rows - object_matrix.shape[1] - 1
        
        # Ensure at least one valid position
        row_offset = random.randint(1, max(1, max_row_offset))
        col_offset = random.randint(1, max(1, max_col_offset))
        
        grid[row_offset:row_offset + object_matrix.shape[0],
             col_offset:col_offset + object_matrix.shape[1]] = object_matrix
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        objects = find_connected_objects(grid, diagonal_connectivity=True,monochromatic=False)
        if not objects:
            raise ValueError("No 8-way connected object found in grid")
        print("objects")
        print(len(objects))
        obj = objects[0]  # Assuming a single object
        bbox = obj.bounding_box
        
        subgrid = grid[bbox[0], bbox[1]]
        output_grid = subgrid[:subgrid.shape[0] // 2, :subgrid.shape[1] // 2]
        
        return output_grid
    
    def create_grids(self) -> tuple:
        taskvars = {'rows': random.randint(10, 30)}
        train_test_data = {
            'train': [],
            'test': []
        }
        
        for _ in range(random.randint(2, 3)):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_test_data['train'].append({'input': input_grid, 'output': output_grid})
        
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        train_test_data['test'].append({'input': test_input, 'output': test_output})
        
        return taskvars, train_test_data
