from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Task6f8cd79bGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "All input grids are completely filled with empty (0) cells with no cell being colored."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grids and adding {color('frame_color')} cells.",
            "The {color('frame_color')} cells are added to the grid border only.",
            "The border is completely filled with {color('frame_color')} cells while the interior remains empty (0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        # Get grid size from gridvars
        height = gridvars['height']
        width = gridvars['width']
        
        # Create grid filled with zeros
        grid = np.zeros((height, width), dtype=int)
        
        return grid
    
    def transform_input(self, grid, taskvars):
        # Create a copy of the input grid
        output = grid.copy()
        
        # Get the frame color from taskvars
        frame_color = taskvars['frame_color']
        
        # Add frame color to the border
        # Top and bottom rows
        output[0, :] = frame_color
        output[-1, :] = frame_color
        
        # Left and right columns (excluding corners which are already colored)
        output[1:-1, 0] = frame_color
        output[1:-1, -1] = frame_color
        
        return output
    
    def create_grids(self):
        # Create task variables
        frame_color = random.randint(1, 9)  # Random color between 1 and 9
        taskvars = {'frame_color': frame_color}
        
        # Generate unique grid sizes for train and test grids
        grid_sizes = []
        for _ in range(4):  # 3 train + 1 test
            while True:
                height = random.randint(5, 30)
                width = random.randint(5, 30)
                if (height, width) not in grid_sizes:
                    grid_sizes.append((height, width))
                    break
        
        # Create train and test grids
        train_grids = []
        for i in range(3):  # 3 train grids
            height, width = grid_sizes[i]
            gridvars = {'height': height, 'width': width}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_grids.append({'input': input_grid, 'output': output_grid})
        
        # Create test grid
        height, width = grid_sizes[3]
        gridvars = {'height': height, 'width': width}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        test_grids = [{'input': input_grid, 'output': output_grid}]
        
        # Return task variables and train/test data
        return taskvars, {'train': train_grids, 'test': test_grids}

