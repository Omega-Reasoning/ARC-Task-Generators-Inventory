from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import create_object, Contiguity, retry
from Framework.transformation_library import find_connected_objects, GridObject

class Task53b68214Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes, with the number of columns always being {vars['cols']} and the number of rows is a multiple of 3 always less than the columns.",
            "They contain exactly one colored object made of 4-way connected, same-colored cells, with the remaining cells being empty (0).",
            "This colored object extends from the first row to the last row but always leaves at least the last three columns completely empty (0).",
            "The object is constructed by vertically repeating a 3-row structural element (the repeated element may have varying widths and internal patterns) down the grid.",
            "Each repeated element is a 3×W block (W>=1) composed of the color of the object and empty cells, and the element is placed directly below the previous one to form the full object.",
            "This way we have multiple instances of the repeated structural element completing the object.",
            "The color and exact repeated structural element vary across examples.",
            "The remaining cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "Output grids are of size {vars['cols']}x{vars['cols']}.",
            "They are constructed by copying the input grid and identifying the single object made of 4-way connected cells, along with its repeated 3×W structural element.",
            "The identified 3-row repeated element is then appended downward repeatedly until the output grid boundary is reached.",
            "If a full element does not fit at the bottom, a partial portion of the element is copied to the remaining rows.",
            "The color and repeated structural element remain unchanged during extension."
        ]
        
        taskvars_definitions = {'cols': random.randint(10, 30)}
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        taskvars = {'cols': random.randint(10, 30)}
        
        
        # Instead of fixed named shapes, create arbitrary 3-row element templates.
        # We'll generate 4 different element templates and withhold one for testing.
        def make_element_template(width: int, color: int) -> np.ndarray:
            # Ensure a vertical spine so stacked elements remain connected
            el = np.zeros((3, width), dtype=int)
            center_col = width // 2
            # always set center column to color for connectivity
            el[:, center_col] = color
            # randomly add additional colored cells to create varied shapes
            for r in range(3):
                for c in range(width):
                    if el[r, c] == 0 and random.random() < 0.4:
                        el[r, c] = color
            return el

        element_templates = []
        used_colors = set()
        for _ in range(4):
            color = random.randint(1, 9)
            while color in used_colors:
                color = random.randint(1, 9)
            used_colors.add(color)
            width = random.randint(1, min(5, taskvars['cols'] - 4))
            element_templates.append({'element': make_element_template(width, color), 'color': color})

        test_idx = random.randint(0, 3)
        test_element = element_templates[test_idx]
        train_elements = [e for i, e in enumerate(element_templates) if i != test_idx]
        
        # Generate 3 train examples with each of the three remaining shapes
        train_examples = []
        used_colors = set()
        
        for elem in train_elements:
            gridvars = {
                'element': elem['element'],
                'color': elem['color']
            }
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate one test example with the fourth shape
        test_input = self.create_input(taskvars, {'element': test_element['element'], 'color': test_element['color']})
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
        # Accept an explicit 3-row element template in gridvars or fall back to a single-column spine.
        element = gridvars.get('element')
        color = gridvars.get('color')

        # Determine number of rows (multiple of 3, less than cols)
        # At least 6 rows but at most cols-3
        min_rows = 6
        max_rows = cols - 3
        possible_row_counts = [r for r in range(min_rows, max_rows + 1) if r % 3 == 0]
        rows = random.choice(possible_row_counts)

        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)

        # If no explicit element provided, use a default vertical spine
        if element is None:
            element = np.array([[color], [color], [color]])
        element = np.array(element, dtype=int)
        if element.shape[0] != 3:
            raise ValueError("Element templates must be 3 rows high")
        element_height, element_width = element.shape

        # Determine starting column that ensures we leave at least 3 columns empty at the end
        max_start_col = cols - element_width - 3
        if max_start_col < 0:
            start_col = 0
        else:
            start_col = random.randint(0, max_start_col)

        # Place the element repeatedly down the grid (rows is a multiple of 3)
        for r in range(0, rows, element_height):
            grid[r:r+element_height, start_col:start_col+element_width] = element

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

