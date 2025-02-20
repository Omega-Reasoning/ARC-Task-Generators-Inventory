from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects
from input_library import create_object, retry, Contiguity

class Task363442eeGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain a completely filled fourth column with {color('cell_color1')} colored cells and a 3x3 block made of multi-colored (1-9) cells, positioned at the top-left corner of the grid.",
            "Several {color('cell_color2')} cells are placed on the right side of the {color('cell_color1')} column.",
            "Each {color('cell_color2')} cell must have at least two consecutive empty (0) cells connected to it in all 8-way directions."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and pasting the 3x3 multi-colored block from the top-left corner onto all {color('cell_color2')} cells.",
            "The center of the 3x3 multi-colored block must always be aligned with the {color('cell_color2')} cell it is pasted on, while the rest of the grid remains unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        rows, cols = taskvars['rows'], taskvars['cols']
        cell_color1, cell_color2 = taskvars['cell_color1'], taskvars['cell_color2']
        grid = np.zeros((rows, cols), dtype=int)

        # Fill the fourth column with cell_color1
        grid[:, 3] = cell_color1

        # Create a 3x3 multi-colored block in the top-left corner
        block_colors = list(set(range(1, 10)) - {cell_color1, cell_color2})
        block = np.random.choice(block_colors, (3, 3))
        grid[:3, :3] = block

        # Place several cell_color2 cells ensuring the required spacing
        available_positions = [(r, c) for r in range(2, rows-2) for c in range(6, cols-2)]
        random.shuffle(available_positions)

        cell_color2_positions = []
        max_cells = random.randint(3, 6)  # Dynamic range of cell_color2 cells
        for r, c in available_positions:
            if (grid[r-2:r+3, c-2:c+3] == 0).all():
                grid[r, c] = cell_color2
                cell_color2_positions.append((r, c))
                if len(cell_color2_positions) >= max_cells:
                    break

        # Store positions in gridvars instead of taskvars
        gridvars['cell_color2_positions'] = cell_color2_positions

        return grid
    
    def transform_input(self, grid: np.ndarray, gridvars: dict) -> np.ndarray:
        cell_color2_positions = gridvars['cell_color2_positions']
        transformed_grid = grid.copy()
        block = grid[:3, :3]  # Extract the original block

        for r, c in cell_color2_positions:
            transformed_grid[r-1:r+2, c-1:c+2] = block

        return transformed_grid
    
    def create_grids(self) -> tuple:
        rows, cols = random.randint(8, 30), random.randint(11, 30)
        
        valid_colors = list(range(1, 10))
        cell_color1, cell_color2 = random.sample(valid_colors, 2)
        
        taskvars = {
            'rows': rows,
            'cols': cols,
            'cell_color1': cell_color1,
            'cell_color2': cell_color2
        }
        
        train_examples = []
        for _ in range(random.randint(3, 4)):
            gridvars = {}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, gridvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        gridvars = {}
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, gridvars)
        
        train_test_data = {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }
        
        return taskvars, train_test_data
    

