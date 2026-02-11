from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import np, random
from transformation_library import find_connected_objects
from typing import Dict, Any, Tuple, List

class TaskRfsSJyB9K5LMhHtRYJQgQWGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They only contain 4-way connected {color('object_color1')} or {color('object_color2')} cells, with the remaining cells being empty (0).",
            "The colored cells form closed objects, which are one-cell-wide rectangular frames enclosing empty (0) interior cells."
        ]
        
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and filling the empty (0) interior cells of the frame.",
            "The fill color is determined by the frame color: it is {color('object_color3')} if the frame is {color('object_color1')}, otherwise it is {color('object_color4')}."
        ]
        
        # 3) Initialize superclass
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        1) Randomly choose rows/cols between 5 and 30
        2) Randomly choose distinct object_color1, object_color2, object_color3, object_color4 in [1..9]
        3) Generate 3-4 training examples and 2 test examples, ensuring variety:
           - At least one training example has a frame of object_color1 and one of object_color2
           - At least one test example has a frame of object_color1 and one of object_color2
           - Randomize positions/shapes of the frames
        """
        
        # 1) Randomly choose rows, cols
        rows = random.randint(7, 30)
        cols = random.randint(7, 30)
        
        # 2) Randomly choose distinct colors
        color_choices = random.sample(range(1, 10), 4)  # 4 distinct numbers from 1..9
        object_color1, object_color2, object_color3, object_color4 = color_choices
        
        # Prepare the dictionary of task variables
        taskvars = {
            'rows': rows,
            'cols': cols,
            'object_color1': object_color1,
            'object_color2': object_color2,
            'object_color3': object_color3,
            'object_color4': object_color4,
        }
        
        # Decide how many training examples (3 or 4)
        nr_train = random.choice([3, 4])
        nr_test = 2  # fixed per instructions

        # We want to ensure that among the training set, 
        # at least one example uses object_color1 and at least one uses object_color2
        # Similarly for test. 
        # We will fix the colors for each example in advance to ensure coverage:
        
        # For training:
        #   - Force 1 example with frame = object_color1
        #   - Force 1 example with frame = object_color2
        #   - The rest can be random from {object_color1, object_color2}
        train_colors = []
        train_colors.append(object_color1)
        train_colors.append(object_color2)
        while len(train_colors) < nr_train:
            train_colors.append(random.choice([object_color1, object_color2]))
        # Shuffle them for variety
        random.shuffle(train_colors)

        # For test (2 examples):
        #   - 1 example with object_color1
        #   - 1 example with object_color2
        test_colors = [object_color1, object_color2]
        random.shuffle(test_colors)
        
        # Now generate the actual train/test data
        train_data = []
        for fcol in train_colors:
            grid_in = self.create_input(taskvars, {'frame_color': fcol})
            grid_out = self.transform_input(grid_in, taskvars)
            train_data.append({'input': grid_in, 'output': grid_out})
        
        test_data = []
        for fcol in test_colors:
            grid_in = self.create_input(taskvars, {'frame_color': fcol})
            grid_out = self.transform_input(grid_in, taskvars)
            test_data.append({'input': grid_in, 'output': grid_out})
        
        train_test_data = {
            'train': train_data,
            'test': test_data
        }
        
        return taskvars, train_test_data

    def create_input(self,
                     taskvars: Dict[str, Any],
                     gridvars: Dict[str, Any]) -> np.ndarray:
        """
        Creates an input grid (rows x cols) filled with 0 except for a 
        single one-cell-wide rectangular frame of color = gridvars['frame_color'].
        
        The frame encloses interior cells that are all 0.
        The frame is guaranteed to be 4-way connected and forms a closed polygon.
        """
        rows = taskvars['rows']
        cols = taskvars['cols']
        frame_color = gridvars['frame_color']
        
        grid = np.zeros((rows, cols), dtype=int)
        
        # Randomly select a rectangle that is at least 3x3 to ensure an interior
        # so top+1 < bottom-1 and left+1 < right-1
        # top can go from 0..(rows-3), bottom from (top+2)..rows
        top = random.randint(0, rows - 5)
        bottom = random.randint(top + 3, rows)
        # left can go from 0..(cols-3), right from (left+2)..cols
        left = random.randint(0, cols - 5)
        right = random.randint(left + 3, cols)
        
        # Draw a one-cell-wide rectangular frame:
        # top and bottom edges
        for c in range(left, right):
            grid[top, c] = frame_color
            grid[bottom - 1, c] = frame_color
        # left and right edges
        for r in range(top, bottom):
            grid[r, left] = frame_color
            grid[r, right - 1] = frame_color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        According to the transformation reasoning chain:
        1) Copy the input grid
        2) Fill the empty (0) interior of each frame:
           - If the frame is object_color1 => fill with object_color2
           - If the frame is object_color2 => fill with object_color4
        """
        object_color1 = taskvars['object_color1']
        object_color2 = taskvars['object_color2']
        object_color3 = taskvars['object_color3']
        object_color4 = taskvars['object_color4']
        
        out_grid = grid.copy()
        
        # Detect frame color by looking at any non-zero cell (all frames in this example are same color).
        # Because we only create a single frame, we can detect it from the first non-zero cell found.
        # If you wanted multiple frames, you'd detect them more carefully.
        frame_color = None
        rows, cols = out_grid.shape
        for r in range(rows):
            for c in range(cols):
                if out_grid[r, c] != 0:
                    frame_color = out_grid[r, c]
                    break
            if frame_color is not None:
                break
        
        if frame_color is None:
            # No frame found => nothing to do
            return out_grid
        
        if frame_color == object_color1:
            fill_color = object_color3
        else:
            # If we followed the instructions strictly, we only expect frames of color1 or color2
            # If it's not color1, treat it as color2 for fill logic
            fill_color = object_color4

        # Fill the interior by bounding box approach:
        # Find the bounding box of the frame
        top, left, bottom, right = self._find_frame_bounding_box(out_grid, frame_color)
        
        # Fill the interior
        # (Guard for the case top+1 < bottom and left+1 < right)
        for rr in range(top+1, bottom):
            for cc in range(left+1, right):
                if out_grid[rr, cc] == 0:
                    out_grid[rr, cc] = fill_color
        
        return out_grid

    def _find_frame_bounding_box(self, grid: np.ndarray, color: int) -> Tuple[int, int, int, int]:
        """
        Given a grid and a frame color, return (top, left, bottom, right) 
        bounding box that contains all cells with 'color'.
        """
        rows, cols = grid.shape
        top = rows
        left = cols
        bottom = 0
        right = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == color:
                    if r < top:    top = r
                    if r >= bottom: bottom = r
                    if c < left:   left = c
                    if c >= right: right = c
        # bottom/right should be one index beyond the actual max if you think in slicing terms,
        # but for our fill, we'll treat them as inclusive boundaries in the loop. 
        # We'll be consistent in the fill code.
        return top, left, bottom+1, right+1

