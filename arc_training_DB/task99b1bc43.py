from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, random_cell_coloring, Contiguity
from transformation_library import find_connected_objects, GridObject

class Task99b1bc43Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has dimensions {vars['rows']} x {vars['cols']}.",
            "One full row is a horizontal divider colored {color('divider')}, splitting the grid into a top half and a bottom half.",
            "Above the divider, there is a single 4-way connected object colored {color('top_color')}.",
            "Below the divider, there is a single 4-way connected object colored {color('bottom_color')}.",
            "All other cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has dimensions {vars['output_rows']} x {vars['cols']}.",
            "Ignore the divider row and compare the top and bottom halves column-wise for each corresponding cell position.",
            "Color an output cell {color('fill')} exactly when one of the two corresponding cells (top vs bottom) is non-empty and the other is empty (XOR).",
            "All remaining output cells are empty (0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # rows: odd between 5 and 30 (inclusive)
        rows = random.choice(list(range(5, 31, 2)))
        # cols: between 5 and 30 (inclusive)
        cols = random.randint(5, 30)
        
        taskvars = {
            'rows': rows,
            'output_rows': rows // 2,
            'cols': cols,
            'divider': random.choice([5, 6, 7, 8, 9]),
            'top_color': random.choice([1, 2, 3, 4]),
            'bottom_color': random.choice([1, 2, 3, 4]),
            'fill': random.choice([1, 2, 3, 4])
        }
        
        # Ensure all colors are different
        while len(set([taskvars['divider'], taskvars['top_color'], taskvars['bottom_color'], taskvars['fill']])) < 4:
            taskvars['divider'] = random.choice([5, 6, 7, 8, 9])
            taskvars['top_color'] = random.choice([1, 2, 3, 4])
            taskvars['bottom_color'] = random.choice([1, 2, 3, 4])
            taskvars['fill'] = random.choice([1, 2, 3, 4])
        
        num_train = random.randint(3, 4)
        
        train_data = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})
        
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_data = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_data, 'test': test_data}
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        divider_color = taskvars['divider']
        top_color = taskvars['top_color']
        bottom_color = taskvars['bottom_color']
        
        grid = np.zeros((rows, cols), dtype=int)
        
        divider_row = rows // 2
        grid[divider_row, :] = divider_color
        
        top_height = divider_row
        top_object = create_object(
            height=top_height,
            width=cols,
            color_palette=top_color,
            contiguity=Contiguity.FOUR,
            background=0
        )
        
        bottom_height = rows - divider_row - 1
        bottom_object = create_object(
            height=bottom_height,
            width=cols,
            color_palette=bottom_color,
            contiguity=Contiguity.FOUR,
            background=0
        )
        
        for r in range(top_height):
            for c in range(cols):
                if top_object[r, c] != 0:
                    grid[r, c] = top_object[r, c]
        
        for r in range(bottom_height):
            for c in range(cols):
                if bottom_object[r, c] != 0:
                    grid[divider_row + 1 + r, c] = bottom_object[r, c]
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        divider_color = taskvars['divider']
        fill_color = taskvars['fill']
        
        divider_row = None
        for r in range(rows):
            if np.all(grid[r, :] == divider_color):
                divider_row = r
                break
        if divider_row is None:
            divider_row = rows // 2
        
        output_rows = rows // 2
        output_grid = np.zeros((output_rows, cols), dtype=int)
        
        for r in range(output_rows):
            for c in range(cols):
                top_half = grid[r, c] != 0
                bottom_half = grid[divider_row + 1 + r, c] != 0
                if top_half != bottom_half:  # XOR
                    output_grid[r, c] = fill_color
        
        return output_grid
