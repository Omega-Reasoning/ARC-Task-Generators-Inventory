from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import retry

class Task496994bdGenerator(ARCTaskGenerator):
    def __init__(self):
        # Input reasoning chain from requirements
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain two or three completely filled rows, with the remaining cells being empty (0).",
            "Each row must be completely filled with the same colored cells, with two distinct colors used in each grid, and the colors varying across grids.",
            "The colored rows should occupy the first two or three rows of the input grids as required."
        ]
        
        # Transformation reasoning chain from requirements
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the colored rows.",
            "The identified colored rows are then duplicated and placed at the bottom of the grid.",
            "When placing the rows at the bottom, their order is swapped, meaning the row that was on top in the input grid now appears at the bottom, and the row that was at the bottom now appears on top."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Define rows and cols as specified in constraints
        rows = random.randint(10, 30)
        cols = random.randint(10, 30)
        
        # Define task variables
        taskvars = {
            'rows': rows,
            'cols': cols,
        }
        
        # Generate 3-4 train examples and 1 test example
        num_train_examples = random.randint(3, 4)
        train_examples = []
        
        # Keep track of color pairs used to ensure variety
        used_color_pairs = set()
        
        for _ in range(num_train_examples):
            # Select two different colors for each example
            objectcol1 = random.randint(1, 9)
            objectcol2 = random.randint(1, 9)
            
            # Make sure colors are different from each other and the pair wasn't used before
            while objectcol2 == objectcol1 or (objectcol1, objectcol2) in used_color_pairs:
                objectcol1 = random.randint(1, 9)
                objectcol2 = random.randint(1, 9)
                if objectcol2 == objectcol1:
                    objectcol2 = (objectcol2 % 9) + 1  # Ensure different colors
            
            used_color_pairs.add((objectcol1, objectcol2))
            
            # Create input grid with specified colors
            gridvars = {
                'objectcol1': objectcol1,
                'objectcol2': objectcol2,
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create test example with different colors from all training examples
        test_objectcol1 = random.randint(1, 9)
        test_objectcol2 = random.randint(1, 9)
        while test_objectcol2 == test_objectcol1 or (test_objectcol1, test_objectcol2) in used_color_pairs:
            test_objectcol1 = random.randint(1, 9)
            test_objectcol2 = random.randint(1, 9)
            if test_objectcol2 == test_objectcol1:
                test_objectcol2 = (test_objectcol2 % 9) + 1
        
        test_gridvars = {
            'objectcol1': test_objectcol1,
            'objectcol2': test_objectcol2,
        }
        
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        objectcol1 = gridvars['objectcol1']
        objectcol2 = gridvars['objectcol2']
        
        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Decide how many rows to color (2 or 3)
        num_colored_rows = random.choice([2, 3]) if rows >= 6 else 2  # Ensure there's enough room
        
        if num_colored_rows == 2:
            # Simple case: two different colored rows
            grid[0, :] = objectcol1
            grid[1, :] = objectcol2
        else:  # 3 rows
            # Choose one of the valid patterns for three rows where same-colored rows are adjacent
            pattern = random.choice([
                # Pattern 1: [A, A, B] - First two rows same color
                [objectcol1, objectcol1, objectcol2],
                
                # Pattern 2: [A, B, B] - Last two rows same color
                [objectcol1, objectcol2, objectcol2]
                
                # Pattern [A, B, A] is NOT valid as it places the different colored row between same colored rows
            ])
            
            # Apply the pattern
            for i in range(3):
                grid[i, :] = pattern[i]
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        rows = taskvars['rows']
        
        # Create a copy of the input grid to modify
        output_grid = grid.copy()
        
        # Find the rows that contain colored cells (non-zero values)
        colored_rows = []
        for r in range(rows):
            if np.any(grid[r] != 0):  # Check if row contains any non-zero (colored) values
                colored_rows.append(r)
                
            # We only need to check the first few rows as per the requirements
            if r > 4:  # Practical limit to check for rows with colors
                break
        
        # Duplicate the colored rows in reverse order at the bottom of the grid
        # Position the duplicated rows at the bottom, maintaining enough space
        bottom_position = rows - len(colored_rows)
        
        for i, row_idx in enumerate(reversed(colored_rows)):
            output_grid[bottom_position + i] = grid[row_idx].copy()
        
        return output_grid

