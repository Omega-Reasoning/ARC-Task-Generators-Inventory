import numpy as np
import random
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry, Contiguity
from transformation_library import find_connected_objects

class Task32597951Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain {color('object_color')}, {color('cell_color1')} and empty (0) cells.",
            "The {color('object_color')} cells are evenly distributed across the grid and sometimes form repeated uniform patterns.",
            "An imaginary rectangular frame is placed randomly in the grid, and all empty (0) cells within the frame are colored {color('cell_color1')}, forming a rectangular block made of {color('object_color')} and {color('cell_color1')} cells.",
            "The size and position of the rectangular frame vary across examples."
        ]

        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying the rectangular block made of {color('object_color')} and {color('cell_color1')} cells.",
            "All {color('object_color')} cells within this block are colored {color('cell_color2')}."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows, cols = taskvars['rows'], taskvars['cols']
        object_color = taskvars['object_color']
        cell_color1 = taskvars['cell_color1']
        
        grid = np.zeros((rows, cols), dtype=int)
        
        # Ensure structured pattern in one example
        if gridvars.get("structured", False):
            for r in range(0, rows, 2):  # Even rows
                for c in range(0, cols, 4):
                    grid[r, c:min(c+3, cols)] = object_color
        else:
            grid = create_object(rows, cols, object_color, contiguity=Contiguity.NONE)
        
        # Create a random frame
        frame_height = random.choice([4, 5])
        frame_width = random.choice([4, 5])
        top = random.randint(0, rows - frame_height)
        left = random.randint(0, cols - frame_width)

        # Fill the frame with cell_color1, preserving existing object_color cells
        for r in range(top, top + frame_height):
            for c in range(left, left + frame_width):
                if grid[r, c] == 0:
                    grid[r, c] = cell_color1
        
        return grid

    def transform_input(self, grid, taskvars):
        object_color = taskvars['object_color']
        cell_color1 = taskvars['cell_color1']
        cell_color2 = taskvars['cell_color2']
        
        output_grid = grid.copy()
        
        # Identify the rectangular block with cell_color1
        frame_cells = np.argwhere(grid == cell_color1)
        if frame_cells.size > 0:
            top_left = frame_cells.min(axis=0)
            bottom_right = frame_cells.max(axis=0)
            
            for r in range(top_left[0], bottom_right[0] + 1):
                for c in range(top_left[1], bottom_right[1] + 1):
                    if grid[r, c] == object_color:
                        output_grid[r, c] = cell_color2
        
        return output_grid

    def create_grids(self):
        rows = random.choice([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
        cols = random.randint(10, 30)
        object_color, cell_color1, cell_color2 = random.sample(range(1, 10), 3)

        taskvars = {
            'rows': rows,
            'cols': cols,
            'object_color': object_color,
            'cell_color1': cell_color1,
            'cell_color2': cell_color2
        }

        train_examples = []
        num_train = random.randint(3, 4)
        
        for i in range(num_train):
            gridvars = {"structured": (i == 0)}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})

        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]

        return taskvars, {'train': train_examples, 'test': test_examples}

