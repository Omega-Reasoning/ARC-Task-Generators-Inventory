from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import GridObject, GridObjects, find_connected_objects
from Framework.input_library import create_object, retry, Contiguity
import numpy as np
import random

class Task0bb8deeeGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} x {vars['cols']}.",
            "Each grid contains exactly one vertical and one horizontal line, both sharing the same color that varies across examples.",
            "There are four single-colored objects, each composed of 8-way connected cells; all other cells are empty (0).",
            "The vertical and horizontal lines intersect to divide the grid into four subgrids, each at least 3×3 in size.",
            "Each of the four objects is located in a different subgrid.",
            "The four objects each have a distinct color different from the color of the vertical and horizontal lines.",
            "Each object is shaped and sized to perfectly fit within a 3×3 subgrid, ensuring at least one cell is present in every row and every column of that subgrid."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are created by identifying the vertical and horizontal lines dividing the grid into four parts, with each subgrid containing one object.",
            "The output grid is 6×6 and is constructed by copying the four colored objects from the input and placing them in the corresponding quadrants of the output grid.",
            "The position of each in the output grid matches its respective position in the input grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        
        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Choose line color
        line_color = gridvars.get('line_color', random.randint(1, 9))
        
        # Choose position for the lines
        # Ensure each subgrid is at least 3x3
        v_line_pos = random.randint(3, cols - 4)  # vertical line position
        h_line_pos = random.randint(3, rows - 4)  # horizontal line position
        
        # Draw vertical line
        grid[:, v_line_pos] = line_color
        
        # Draw horizontal line
        grid[h_line_pos, :] = line_color
        
        # Define the four subgrids
        subgrids = [
            (0, h_line_pos, 0, v_line_pos),  # top-left
            (0, h_line_pos, v_line_pos + 1, cols),  # top-right
            (h_line_pos + 1, rows, 0, v_line_pos),  # bottom-left
            (h_line_pos + 1, rows, v_line_pos + 1, cols)  # bottom-right
        ]
        
        # Choose 4 distinct colors for objects (different from line color)
        available_colors = list(range(1, 10))
        available_colors.remove(line_color)
        object_colors = random.sample(available_colors, 4)
        
        # Place one object in each subgrid
        for i, (r1, r2, c1, c2) in enumerate(subgrids):
            # Find a 3x3 region within this subgrid
            subgrid_height = r2 - r1
            subgrid_width = c2 - c1
            
            # Random position for 3x3 object within subgrid
            max_obj_r = min(r2 - 3, r1 + subgrid_height - 3)
            max_obj_c = min(c2 - 3, c1 + subgrid_width - 3)
            obj_r = random.randint(r1, max_obj_r)
            obj_c = random.randint(c1, max_obj_c)
            
            # Create object that fits in 3x3 and has at least one cell in each row/col
            obj = retry(
                lambda: create_object(3, 3, object_colors[i], Contiguity.EIGHT, background=0),
                lambda x: np.all(np.any(x != 0, axis=1)) and np.all(np.any(x != 0, axis=0))
            )
            
            # Place object in grid, avoiding the lines
            for r in range(3):
                for c in range(3):
                    if obj[r, c] != 0:
                        grid_r = obj_r + r
                        grid_c = obj_c + c
                        if grid[grid_r, grid_c] == 0:  # Don't overwrite lines
                            grid[grid_r, grid_c] = obj[r, c]
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        # Find the line positions by looking for complete lines
        v_line_pos = None
        h_line_pos = None
        line_color = None
        
        # Find vertical line
        for c in range(grid.shape[1]):
            col = grid[:, c]
            if len(np.unique(col)) == 1 and col[0] != 0:
                v_line_pos = c
                line_color = col[0]
                break
        
        # Find horizontal line
        for r in range(grid.shape[0]):
            row = grid[r, :]
            if len(np.unique(row)) == 1 and row[0] != 0 and row[0] == line_color:
                h_line_pos = r
                break
        
        # Create output grid
        output = np.zeros((6, 6), dtype=int)
        
        # Define quadrants in input and corresponding positions in output
        quadrants = [
            ((0, h_line_pos, 0, v_line_pos), (0, 3, 0, 3)),  # top-left
            ((0, h_line_pos, v_line_pos + 1, grid.shape[1]), (0, 3, 3, 6)),  # top-right
            ((h_line_pos + 1, grid.shape[0], 0, v_line_pos), (3, 6, 0, 3)),  # bottom-left
            ((h_line_pos + 1, grid.shape[0], v_line_pos + 1, grid.shape[1]), (3, 6, 3, 6))  # bottom-right
        ]
        
        for (in_r1, in_r2, in_c1, in_c2), (out_r1, out_r2, out_c1, out_c2) in quadrants:
            # Extract subgrid
            subgrid = grid[in_r1:in_r2, in_c1:in_c2].copy()
            
            # Remove line color
            subgrid[subgrid == line_color] = 0
            
            # Find the object in this subgrid
            objects = find_connected_objects(subgrid, diagonal_connectivity=True)
            
            if len(objects) > 0:
                # Get the first (and should be only) object
                obj = objects[0]
                
                # Get the bounding box of the object
                if len(obj.cells) > 0:
                    rows, cols, _ = zip(*obj.cells)
                    min_r, max_r = min(rows), max(rows)
                    min_c, max_c = min(cols), max(cols)
                    
                    # Place object in output quadrant
                    for r, c, color in obj.cells:
                        # Calculate relative position within the object's bounding box
                        rel_r = r - min_r
                        rel_c = c - min_c
                        
                        # Place in output if within 3x3 bounds
                        if rel_r < 3 and rel_c < 3:
                            output[out_r1 + rel_r, out_c1 + rel_c] = color
        
        return output
    
    def create_grids(self) -> tuple:
        # Random grid size
        rows = random.randint(10, 30)
        cols = random.randint(10, 30)
        
        taskvars = {
            'rows': rows,
            'cols': cols
        }
        
        # Generate train and test examples
        num_train = random.randint(3, 6)
        
        def generate_example():
            # Generate random line color for this example
            line_color = random.randint(1, 9)
            gridvars = {'line_color': line_color}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            return {'input': input_grid, 'output': output_grid}
        
        train_examples = [generate_example() for _ in range(num_train)]
        test_examples = [generate_example()]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data

