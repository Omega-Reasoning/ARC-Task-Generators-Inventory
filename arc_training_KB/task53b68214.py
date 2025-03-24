from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, GridObject

class Task53b68214Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes, with the number of columns always being {vars['cols']} and the number of rows is a multiple of 3 always less than the columns.",
            "They contain exactly one colored object made of 4-way connected, same-colored cells, with the remaining cells being empty (0).",
            "This colored object extends from the first row to the last row but always leaves at least the last three columns completely empty (0).",
            "The object follows a repeating pattern, where a specific structural element—one of the following: a three cell long vertical line, T-shaped element, H-shaped element or Cross-shaped element, is consistently repeated throughout its shape.",
            "A T-shaped element in the input grid follows the form:[[c, c, c], [0, c, 0], [0, c, 0]] for a color c.",
            "An H-shaped element in the input grid follows the form: [[c, 0, c], [c, c, c], [c, 0, c]] for a color c.",
            "A cross-shaped element in the input grid follows the form: [[0, c, 0], [c, 0, c], [0, c, 0]] for a color c.",
            "This way we have multiple instances of the repeated structural element completing the object.",
            "The color and shape of the object vary across examples.",
            "The remaining cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "Output grids are of size {vars['cols']}x{vars['cols']}.",
            "They are constructed by copying the input grid and identifying the single object made of 4-way connected cells, along with its repeated structural element throughout the shape.",
            "The repeated structural element is determined by dividing the number of rows by 3 and considering the remaining portion of the object as the repeating unit.",
            "The existing object is then extended by adding more instances of its repeated structural element, placed directly below where the input object finishes.",
            "The extension continues until the object reaches the grid boundaries while maintaining structural consistency.",
            "In case a full structural element cannot be placed, only a partial portion of it is added until the grid edge is reached.",
            "The color and repeated structural elements of the object remain unchanged."
        ]
        
        taskvars_definitions = {'cols': random.randint(10, 30)}
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        taskvars = {'cols': random.randint(10, 30)}
        
        # Define all four shape types
        shape_types = ['vertical', 'T', 'H', 'cross']
        
        # Randomly decide which shape will be used for testing
        test_shape_idx = random.randint(0, 3)
        test_shape = shape_types[test_shape_idx]
        
        # The remaining three shapes will be used for training
        train_shapes = [shape for i, shape in enumerate(shape_types) if i != test_shape_idx]
        
        # Generate 3 train examples with each of the three remaining shapes
        train_examples = []
        used_colors = set()
        
        for shape_type in train_shapes:
            # Select a color not previously used
            color = random.randint(1, 9)
            while color in used_colors:
                color = random.randint(1, 9)
            used_colors.add(color)
            
            # Create gridvars dictionary for this example
            gridvars = {
                'shape_type': shape_type,
                'color': color
            }
            
            # Generate the input grid
            input_grid = self.create_input(taskvars, gridvars)
            
            # Generate the output grid
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate one test example with the fourth shape
        test_color = random.randint(1, 9)
        while test_color in used_colors:
            test_color = random.randint(1, 9)
            
        gridvars = {
            'shape_type': test_shape,
            'color': test_color
        }
        
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }

    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        cols = taskvars['cols']
        shape_type = gridvars['shape_type']
        color = gridvars['color']
        
        # Determine number of rows (multiple of 3, less than cols)
        # At least 6 rows but at most cols-3
        min_rows = 6
        max_rows = cols - 3
        possible_row_counts = [r for r in range(min_rows, max_rows + 1) if r % 3 == 0]
        rows = random.choice(possible_row_counts)
        
        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Define the structural element based on shape_type
        if shape_type == 'vertical':
            # Vertical line element is simply a column of color
            element = np.array([[color], [color], [color]])
            element_width = 1
        elif shape_type == 'T':
            element = np.array([
                [color, color, color],
                [0, color, 0],
                [0, color, 0]
            ])
            element_width = 3
        elif shape_type == 'H':
            element = np.array([
                [color, 0, color],
                [color, color, color],
                [color, 0, color]
            ])
            element_width = 3
        elif shape_type == 'cross':
            element = np.array([
                [0, color, 0],
                [color, 0, color],
                [0, color, 0]
            ])
            element_width = 3
        
        # Determine starting column that ensures we leave at least 3 columns empty at the end
        max_start_col = cols - element_width - 3
        start_col = random.randint(0, max_start_col)
        
        # Place the element repeatedly down the grid
        for r in range(0, rows, 3):
            grid[r:r+3, start_col:start_col+element_width] = element
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        cols = taskvars['cols']
        input_rows = grid.shape[0]
        
        # Create a square output grid of size cols x cols
        output = np.zeros((cols, cols), dtype=int)
        
        # Copy the input grid to the output grid
        output[:input_rows, :grid.shape[1]] = grid
        
        # Find the object's color
        non_zero_mask = grid > 0
        if np.any(non_zero_mask):
            color = grid[non_zero_mask][0]  # Get the color of the first non-zero cell
            
            # Find the width and position of the object
            non_zero_cols = np.any(non_zero_mask, axis=0)
            col_indices = np.where(non_zero_cols)[0]
            if len(col_indices) > 0:
                col_start = col_indices[0]
                col_end = col_indices[-1] + 1
                
                # Get the repeating element (always 3 rows high)
                element_height = 3
                
                # Identify the structural element by taking the first 3 rows
                element = grid[:element_height, col_start:col_end].copy()
                
                # Continue extending the pattern downward
                for r in range(input_rows, cols, element_height):
                    # Calculate how many rows we can still copy
                    remaining_rows = cols - r
                    rows_to_copy = min(element_height, remaining_rows)
                    
                    # Copy either the full element or a partial element
                    output[r:r+rows_to_copy, col_start:col_end] = element[:rows_to_copy]
                    
                    # If we couldn't fit a full element, we're done
                    if rows_to_copy < element_height:
                        break
                        
        return output

