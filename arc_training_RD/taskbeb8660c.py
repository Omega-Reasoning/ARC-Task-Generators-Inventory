from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import Contiguity, create_object, retry
import numpy as np
import random

class Tasktaskbeb8660cGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can vary in size, with width limited to 9 columns (Why only 9? Because we have only 9 colors available).",
            "Each row contains exactly one horizontal line of colored blocks.",
            "The number of colored blocks in each row forms a sequence from 1 to N, where N is the grid width.",
            "The rows are randomly scattered throughout the grid - not arranged in any particular order.",
            "The last row is always completely filled with colored blocks of the same color across all grids.",
            "Each row has a unique color (1-9) and a unique length within each grid.",
            "The colored blocks in each row can be positioned anywhere horizontally within that row."
        ]
        
        transformation_reasoning_chain = [
            "The output grid maintains the same size as the input grid.",
            "Arrange all rows by their block length in ascending order (1 to N blocks).",
            "All rows are right-aligned to form a perfect staircase pattern.",
            "Each row keeps its original color while being repositioned.",
            "Empty rows are placed at the top of the grid.",
            "The final pattern shows a perfect staircase from 1 block at top to N blocks at bottom.",
            "The bottom row remains completely filled with the consistent color used across all grids."
        ]
        
        taskvars_definitions = {}
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        # Get the consistent last row color from taskvars
        last_row_color = taskvars['last_row_color']
        
        # First determine the grid width (this will be the length of the bottom row)
        cols = random.randint(4, 9)  # Limit to 9 for available colors
        
        # Determine how many different block lengths we'll have (1 to cols)
        # We need exactly cols different lengths: 1, 2, 3, ..., cols
        num_blocks = cols
        
        # Ensure we have enough rows (at least num_blocks, but can have more empty rows)
        min_rows = num_blocks
        rows = random.randint(min_rows, min_rows + 4)  # Add some extra empty rows
        
        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Generate unique colors for each length, excluding the last row color
        available_colors = [c for c in range(1, 10) if c != last_row_color]
        colors = random.sample(available_colors, num_blocks - 1)  # Need one less since last row color is fixed
        
        # Add the fixed last row color to the colors list
        colors.append(last_row_color)
        
        # Create lengths from 1 to cols (cols is the width of the bottom row)
        lengths = list(range(1, cols + 1))
        
        # The last row must always be completely filled with the consistent color
        grid[rows - 1, :] = last_row_color  # Fill entire last row
        
        # For the remaining blocks (lengths 1 to cols-1), randomly assign to other rows
        remaining_lengths = lengths[:-1]  # All lengths except the last one (cols)
        remaining_colors = colors[:-1]    # All colors except the last one
        
        # Shuffle the remaining colors to randomize assignment
        random.shuffle(remaining_colors)
        
        # Select random rows for the remaining blocks (exclude the last row)
        available_rows = list(range(rows - 1))  # All rows except the last one
        selected_rows = random.sample(available_rows, len(remaining_lengths))
        
        # Place the remaining blocks in selected rows
        for row_idx, length, color in zip(selected_rows, remaining_lengths, remaining_colors):
            # Randomly position the horizontal block within this row
            if length <= cols:
                start_col = random.randint(0, cols - length)
                # Place the horizontal block
                grid[row_idx, start_col:start_col + length] = color
        
        return grid
    
    def transform_input(self, input_grid, taskvars):
        # Create output grid of same size
        output_grid = np.zeros_like(input_grid)
        rows, cols = input_grid.shape
        
        # Find all horizontal blocks
        objects = find_connected_objects(input_grid, diagonal_connectivity=False, background=0)
        
        if len(objects.objects) == 0:
            return output_grid
        
        # Sort objects by width in ascending order (shortest to longest)
        objects_by_width = sorted(objects.objects, key=lambda obj: obj.width)
        
        # Calculate starting row position to place the staircase
        # We want to place them consecutively ending at the last row
        num_objects = len(objects_by_width)
        start_row = rows - num_objects
        
        # Place rows in ascending order by width, right-aligned
        for i, obj in enumerate(objects_by_width):
            color = list(obj.colors)[0]
            width = obj.width
            row_position = start_row + i
            
            # Place the row right-aligned
            col_position = cols - width
            output_grid[row_position, col_position:col_position + width] = color
        
        return output_grid
        
    def create_grids(self):
        # Choose a consistent color for the last row across all grids
        last_row_color = random.randint(1, 9)
        
        # Set up taskvars with the consistent last row color
        taskvars = {
            'last_row_color': last_row_color
        }
        
        # Randomize number of train examples
        num_train = random.randint(3, 5)
        
        # Create train pairs - all will use the same last_row_color
        train_pairs = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Create one test example - also uses the same last_row_color
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)
