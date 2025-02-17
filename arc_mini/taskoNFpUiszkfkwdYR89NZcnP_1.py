from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

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
        nr_train_examples = random.randint(3, 6)
        nr_test_examples = 1
        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)
        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        object_color1 = taskvars['object_color1']
        object_color2 = taskvars['object_color2']
        rows = random.randint(8, 30)
        cols = random.randint(8, 30)
        grid = np.zeros((rows, cols), dtype=int)
        
        def draw_frame(g, top, left, bottom, right, color):
            g[top, left:right+1] = color
            g[bottom, left:right+1] = color
            g[top:bottom+1, left] = color
            g[top:bottom+1, right] = color

        def random_frame_positions(r, c, existing_frames=[]):
            while True:
                top = random.randint(0, r - 3)
                bottom = random.randint(top + 2, r - 1)
                left = random.randint(0, c - 3)
                right = random.randint(left + 2, c - 1)
                new_frame = (top, left, bottom, right)
                if all(not self.frames_overlap(new_frame, f) for f in existing_frames):
                    return new_frame

        existing_frames = []
        t1, l1, b1, r1 = random_frame_positions(rows, cols)
        draw_frame(grid, t1, l1, b1, r1, object_color1)
        existing_frames.append((t1, l1, b1, r1))

        t2, l2, b2, r2 = random_frame_positions(rows, cols, existing_frames)
        draw_frame(grid, t2, l2, b2, r2, object_color2)
        
        return grid

    def frames_overlap(self, frame1, frame2):
        t1, l1, b1, r1 = frame1
        t2, l2, b2, r2 = frame2
        return not (b1 < t2 or b2 < t1 or r1 < l2 or r2 < l1)

    def transform_input(self, grid, taskvars):
        object_color2 = taskvars['object_color2']
        output_grid = np.copy(grid)
        rows_color2, cols_color2 = np.where(grid == object_color2)
        if len(rows_color2) == 0:
            return output_grid
        top2, bottom2 = min(rows_color2), max(rows_color2)
        left2, right2 = min(cols_color2), max(cols_color2)
        if bottom2 - top2 > 1 and right2 - left2 > 1:
            fill_region = (slice(top2+1, bottom2), slice(left2+1, right2))
            interior = output_grid[fill_region]
            interior[interior == 0] = object_color2
        return output_grid


