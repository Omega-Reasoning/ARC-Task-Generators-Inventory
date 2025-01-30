from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class ARCTask2dc579daGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has a random odd size between 11x11 and 29x29.",
            "All the cells except the middle row and column are colored with a base color (between 1-9).",
            "The input grid dimensions are always odd.",
            "The cells of the middle column and the middle row are either empty (0) or colored with a different color (between 1-9).",
            "The input grid has four quadrants which are divided by the middle row and column.",
            "A random quadrant cell is colored with a third unique color (between 1-9)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has a different size than the input grid.",
            "First identify the quadrant which has a different color than all the other cells.",
            "The output grid is the quadrant which has one cell color different from the rest of the colored cells."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        # Generate random odd size between 11 and 29
        rows = random.choice(range(11, 30, 2))
        
        # Generate three unique random colors
        colors = []
        while len(colors) < 3:
            new_color = random.randint(1, 9)
            if new_color not in colors:
                colors.append(new_color)
        
        color_1, color_2, color_3 = colors
        
        grid = np.full((rows, rows), color_1, dtype=int)
        
        mid = rows // 2
        # Randomly decide if middle row and column should be empty or colored
        middle_value = random.choice([0, color_2])
        grid[mid, :] = middle_value
        grid[:, mid] = middle_value
        
        quadrant = random.choice([(0, 0), (0, mid+1), (mid+1, 0), (mid+1, mid+1)])
        q_row = random.randint(quadrant[0], quadrant[0] + mid - 1)
        q_col = random.randint(quadrant[1], quadrant[1] + mid - 1)
        grid[q_row, q_col] = color_3
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        rows = len(grid)  # Get rows from grid instead of taskvars
        mid = rows // 2
        
        quadrants = {
            (0, 0): grid[:mid, :mid],
            (0, mid+1): grid[:mid, mid+1:],
            (mid+1, 0): grid[mid+1:, :mid],
            (mid+1, mid+1): grid[mid+1:, mid+1:]
        }
        
        for key, quadrant in quadrants.items():
            unique_colors = np.unique(quadrant)
            if len(unique_colors) > 1:
                return quadrant

    def create_grids(self) -> tuple:
        taskvars = {}  # Empty dict since we don't need any task variables
        
        train_data = [
            {
                'input': (inp := self.create_input(taskvars, {})),
                'output': self.transform_input(inp, taskvars)
            }
            for _ in range(random.randint(3, 4))
        ]
        
        test_data = [{
            'input': (test_inp := self.create_input(taskvars, {})),
            'output': self.transform_input(test_inp, taskvars)
        }]
        
        return taskvars, {'train': train_data, 'test': test_data}
