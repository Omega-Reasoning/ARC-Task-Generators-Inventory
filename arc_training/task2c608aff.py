import numpy as np
import random
from typing import Dict, List, Any, Tuple
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import retry
from transformation_library import find_connected_objects, GridObject, GridObjects

class ARCTask2c608affGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['cols']}.",
            "The entire input grid is filled with color grid_color(between 1 and 9).",
            "There is a single 4-way connected object which is either a square or rectangle of color obj_color(between 1 and 9).",
            "There are a random number of cells placed in the input grid which are of color cell_color(between 1 and 9)."
        ]
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "Copy the input grid to the output grid.",
            "Identify the subgrid which contains the 4 way connected object.",
            "If the cells with the color cell_color are aligned in the same row or column as the object, then extend that row or column with additional cells of the same color until they reach the objects boundary."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables ensuring distinct colors
        rows = random.randint(13, 30)
        cols = random.randint(13, 30)
        taskvars = {
            'rows': rows,
            'cols': cols,
        }
        
        # Create 3-4 train examples and 1 test example
        nr_train = random.randint(3, 4)
        
        def generate_gridvars():
            while True:
                colors = random.sample(range(1, 10), 3)
                grid_color, obj_color, cell_color = colors
                if len(set(colors)) == 3:
                    return {
                        'grid_color': grid_color,
                        'obj_color': obj_color,
                        'cell_color': cell_color
                    }
        train_examples = []
        for i in range(nr_train):
            gridvars = generate_gridvars()
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append(GridPair({"input": input_grid, "output": output_grid}))

        # Create test example with the last color pair
        gridvars = generate_gridvars()
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [GridPair({"input": test_input, "output": test_output})]
        
        return taskvars, TrainTestData({"train": train_examples, "test": test_examples})
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        grid_color = gridvars['grid_color']
        obj_color = gridvars['obj_color']
        cell_color = gridvars['cell_color']
        
        grid = np.full((rows, cols), grid_color, dtype=int)
        
        # Generate object dimensions and position
        max_h = rows // 3
        max_w = cols // 3
        h = random.randint(4, max_h)
        w = random.randint(4, max_w)
        r_start = random.randint(0, rows - h)
        c_start = random.randint(0, cols - w)
        grid[r_start:r_start+h, c_start:c_start+w] = obj_color
        max_cells = (rows - h) * (cols - w) // 8
        
        # Place cell_color cells
        num_cells = random.randint(4, rows//2)
        for _ in range(num_cells):
            # Use retry to ensure placement in valid position
            def generate_cell():
                r = random.randint(0, rows - 1)
                c = random.randint(0, cols - 1)
                # Check if position overlaps with the rectangle/square object
                if r_start <= r < r_start + h and c_start <= c < c_start + w:
                    return None
                return (r, c)
            
            cell = retry(lambda: generate_cell(), lambda pos: pos is not None and grid[pos] == grid_color)
            if cell:
                grid[cell] = cell_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output = np.copy(grid)
        # Determine colors from the grid
        unique_colors = np.unique(grid)
        if len(unique_colors) != 3:
            return output  # Invalid grid state
        
        print(unique_colors)
        grid_color = np.bincount(grid.flatten()).argmax()
        
        # Find all objects
        objects = find_connected_objects(output, diagonal_connectivity=False, background=grid_color)
        if len(objects) == 0:
            return output
            
        # The rectangle/square will be the largest connected component
        largest_obj = max(objects, key=lambda x: x.size)
        obj_color = largest_obj.colors.pop()
        
        # Cell color will be the remaining color
        cell_color = next(c for c in unique_colors if c != grid_color and c != obj_color)
        
        # Find the object
        objects = objects.with_color(obj_color)
        if len(objects) != 1:
            return output
        obj = objects[0]
        r_start, r_end = obj.bounding_box[0].start, obj.bounding_box[0].stop
        c_start, c_end = obj.bounding_box[1].start, obj.bounding_box[1].stop
        
        # Process each cell_color cell
        cell_coords = np.argwhere(output == cell_color)
        for r, c in cell_coords:
            in_row = r_start <= r < r_end
            in_col = c_start <= c < c_end
            if in_row:
                if c < c_start:
                    start, end = c + 1, c_start - 1
                    if start <= end:
                        output[r, start:end+1] = cell_color
                elif c >= c_end:
                    start, end = c_end, c - 1
                    if start <= end:
                        output[r, start:end+1] = cell_color
            elif in_col:
                if r < r_start:
                    start, end = r + 1, r_start - 1
                    if start <= end:
                        output[start:end+1, c] = cell_color
                elif r >= r_end:
                    start, end = r_end, r - 1
                    if start <= end:
                        output[start:end+1, c] = cell_color
        
        return output