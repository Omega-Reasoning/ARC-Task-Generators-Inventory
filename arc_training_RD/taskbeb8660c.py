from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import Contiguity, create_object, retry
import numpy as np
import random

class Tasktaskbeb8660cGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can vary in size, with width limited to 9 columns(Why only 9? Because we have only 9 colors available).",
            "Each row contains a horizontal line of colored blocks.",
            "The number of colored blocks in each row forms a sequence from 1 to N, where N is the grid width.",
            "The rows are randomly arranged, except for the bottom row which is always completely filled.",
            "Each row has a unique color (1-9) and a unique length.",
            "All rows are right-aligned in their random positions."
        ]
        
        transformation_reasoning_chain = [
            "The output grid maintains the same size as the input grid.",
            "The bottom row remains unchanged - completely filled with its original color.",
            "Above the bottom row, arrange other rows by length in ascending order (1 to N-1).",
            "All rows are right-aligned to form a perfect staircase pattern.",
            "Each row keeps its original color while being repositioned.",
            "The final pattern shows a perfect staircase from 1 block at top to N blocks at bottom."
        ]
        
        taskvars_definitions = {}
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, rows=None, cols=None):
        # Ensure valid grid size
        if rows is None:
            rows = random.randint(5, 12)
        if cols is None:
            cols = random.randint(5, 9)
        
        # Ensure cols is within 9 (for colors 1-9)
        cols = min(cols, 9)
        
        # Determine number of rows needed for pattern (1 to cols)
        num_pattern_rows = cols  # We need exactly cols number of rows for lengths 1 to cols
        
        # Ensure we have enough rows for the pattern
        rows = max(rows, num_pattern_rows)
        
        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Generate colors for lengths 1 to cols
        colors = random.sample(range(1, 10), cols)
        
        # Create row lengths (1 to cols)
        row_lengths = list(range(1, cols + 1))  # 1, 2, 3, ..., cols
        
        # Calculate available rows for pattern (ensure enough space at bottom)
        available_rows = list(range(rows - cols, rows))
        random.shuffle(available_rows[:-1])  # Shuffle all except last row
        
        # Place the rows
        for length, position, color in zip(row_lengths, available_rows, colors):
            if position == rows - 1:
                # Last row is always completely filled
                grid[position, :] = color
            else:
                # Place other rows right-aligned with proper length
                start_col = cols - length  # Right align
                grid[position, start_col:start_col+length] = color
        
        return grid
    
    def transform_input(self, input_grid):
        # Create output grid of same size
        output_grid = np.zeros_like(input_grid)
        rows, cols = input_grid.shape
        
        # Find all horizontal rows
        objects = find_connected_objects(input_grid, diagonal_connectivity=False, background=0)
        
        # Sort objects by width in ascending order (shortest to longest)
        objects_by_width = sorted(objects.objects, key=lambda obj: obj.width)
        
        # Start placing from bottom up
        row_position = rows - len(objects_by_width)
        
        # Place rows in ascending order
        for obj in objects_by_width:
            color = list(obj.colors)[0]
            width = obj.width
            
            # Place the row right-aligned
            col_position = cols - width
            output_grid[row_position, col_position:] = color
            row_position += 1
        
        return output_grid
        
    def create_grids(self):
        # Randomize number of train examples
        num_train = random.randint(3, 5)
        
        # Create train pairs with different grid sizes
        train_pairs = []
        for _ in range(num_train):
            # Ensure cols is within 9 for proper staircase
            rows = random.randint(5, 12)
            cols = random.randint(5, 9)  # Limit to 9 for available colors
            
            input_grid = self.create_input(rows, cols)
            output_grid = self.transform_input(input_grid)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Create one test example
        test_rows = random.randint(8, 12)
        test_cols = random.randint(5, 9)  # Keep within 9
        
        test_input = self.create_input(test_rows, test_cols)
        test_output = self.transform_input(test_input)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return {}, TrainTestData(train=train_pairs, test=test_pairs)