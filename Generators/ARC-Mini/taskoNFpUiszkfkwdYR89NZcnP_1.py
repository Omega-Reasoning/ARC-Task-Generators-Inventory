from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from scipy.ndimage import label

class TaskoNFpUiszkfkwdYR89NZcnP_1Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain {color('object_color1')} and {color('object_color2')} objects, which are one-cell-wide rectangular frames enclosing empty (0) cells."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and filling all empty (0) cells enclosed by {color('object_color2')} cells with {color('object_color2')} color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        c1, c2 = random.sample(range(1, 10), 2)
        taskvars = {
            "object_color1": c1,
            "object_color2": c2
        }
        nr_train_examples = random.randint(3, 5)
        nr_test_examples = 1
        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)
        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        object_color1 = taskvars['object_color1']
        object_color2 = taskvars['object_color2']
        
        # Always ensure the grid is large enough (minimum 12x12)
        rows = random.randint(12, 30)  
        cols = random.randint(12, 30)
        grid = np.zeros((rows, cols), dtype=int)
        
        # Simple, reliable positioning approach:
        # Always create a color2 frame in top left and color1 frame in bottom right
        
        # Color2 frame in top left quadrant
        top2 = 1
        left2 = 1
        bottom2 = top2 + random.randint(2, 4)
        right2 = left2 + random.randint(2, 4)
        self.draw_frame(grid, top2, left2, bottom2, right2, object_color2)
        
        # Color1 frame in bottom right quadrant
        top1 = rows - 6
        left1 = cols - 6
        bottom1 = rows - 2
        right1 = cols - 2
        self.draw_frame(grid, top1, left1, bottom1, right1, object_color1)
        
        return grid

    def draw_frame(self, g, top, left, bottom, right, color):
        # Draw top and bottom edges
        g[top, left:right+1] = color
        g[bottom, left:right+1] = color
        # Draw left and right edges
        g[top:bottom+1, left] = color
        g[top:bottom+1, right] = color
        
    def frames_overlap(self, frame1, frame2):
        t1, l1, b1, r1 = frame1
        t2, l2, b2, r2 = frame2
        # Add a buffer of 1 cell to avoid frames being adjacent
        return not (b1+1 < t2 or b2+1 < t1 or r1+1 < l2 or r2+1 < l1)

    def transform_input(self, grid, taskvars):
        object_color2 = taskvars['object_color2']
        output_grid = np.copy(grid)
        
        # Find all color2 cells
        mask = (grid == object_color2)
        
        # Find connected components (each should be a frame)
        labeled_array, num_features = label(mask)
        
        # Process each component
        for i in range(1, num_features + 1):
            # Get coordinates of this component
            rows, cols = np.where(labeled_array == i)
            
            if len(rows) < 4:  # Skip if too small to be a frame
                continue
                
            # Get bounds
            top, bottom = min(rows), max(rows)
            left, right = min(cols), max(cols)
            
            # Only process if it could be a valid rectangle frame
            if bottom - top > 1 and right - left > 1:
                # Fill interior empty cells
                for r in range(top+1, bottom):
                    for c in range(left+1, right):
                        if grid[r, c] == 0:  # Only fill empty cells
                            output_grid[r, c] = object_color2
        
        return output_grid