from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, GridObject, GridObjects

class Task7468f01aGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain a colored rectangular object, with all remaining cells being empty (0).",
            "The rectangular object is located within the interior of the grid and never touches the grid border.",
            "The rectangular object contains one or more small colored objects inside it.",
            "These inner objects can be shaped differently and are significantly smaller than the rectangular object.",
            "All objects inside the rectangular object must be of the same color, which is different from the color of the rectangular object.",
            "The colors and size of both the rectangular object and the inner objects vary across examples."
        ]

        transformation_reasoning_chain = [
            "Output grids are constructed by identifying the large rectangular object that contains smaller objects inside it.",
            "The size of the output grid is exactly the same as the size of the rectangular object.",
            "Once the object has been identified, reflect it horizontally and paste it into the output grid.",
            "The reflection results in a mirrored version of the objects inside the rectangular object."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Generate random grid dimensions between 10 and 30
        rows = random.randint(10, 30)
        cols = random.randint(10, 30)
        
        taskvars = {'rows': rows, 'cols': cols}
        
        # Create train and test data
        train_examples = []
        
        # Ensure we have at least one example with multiple inner objects
        has_multiple_inner_objects = False
        
        for i in range(3):  # 3 training examples
            # For the second example, ensure multiple inner objects
            if i == 1:
                gridvars = {'num_inner_objects': random.randint(2, 4)}
                has_multiple_inner_objects = True
            else:
                # First and third examples can have any number
                gridvars = {'num_inner_objects': random.randint(1, 3)}
                if gridvars['num_inner_objects'] >= 2:
                    has_multiple_inner_objects = True
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # If we still don't have an example with multiple inner objects, make the test case have multiple
        test_gridvars = {'num_inner_objects': random.randint(2, 4) if has_multiple_inner_objects else random.randint(1, 3)}
        
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        return taskvars, {
            'train': train_examples,
            'test': [{
                'input': test_input,
                'output': test_output
            }]
        }

    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        rows, cols = taskvars['rows'], taskvars['cols']
        
        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Select random colors for the outer rectangle and inner objects
        outer_color = random.randint(1, 9)
        inner_color = random.choice([c for c in range(1, 10) if c != outer_color])
        
        # Determine rectangle size (at least 4x3 or 3x4)
        rect_height = random.randint(4, min(rows-4, 15))
        rect_width = random.randint(4, min(cols-4, 15))
        
        # Minimum size constraint
        if rect_height < 3:
            rect_width = max(rect_width, 4)
        if rect_width < 3:
            rect_height = max(rect_height, 4)
        
        # Place rectangle with padding to ensure it doesn't touch borders
        row_start = random.randint(1, rows - rect_height - 1)
        col_start = random.randint(1, cols - rect_width - 1)
        
        # Create rectangle
        for r in range(row_start, row_start + rect_height):
            for c in range(col_start, col_start + rect_width):
                grid[r, c] = outer_color
        
        # Determine number of inner objects to create
        num_inner_objects = gridvars.get('num_inner_objects', random.randint(1, 3))
        
        # Track inner object positions to avoid overlap
        inner_positions = []
        
        # Create inner objects
        for _ in range(num_inner_objects):
            # Size of inner object (significantly smaller than rectangle)
            inner_height = random.randint(1, max(1, min(rect_height // 3, 3)))
            inner_width = random.randint(1, max(1, min(rect_width // 3, 3)))
            
            # Try to place inner object without overlapping
            max_attempts = 20
            for _ in range(max_attempts):
                # Place within the rectangle with padding
                inner_row = random.randint(row_start + 1, row_start + rect_height - inner_height - 1)
                inner_col = random.randint(col_start + 1, col_start + rect_width - inner_width - 1)
                
                # Check for overlap with existing inner objects
                overlap = False
                for pos in inner_positions:
                    pos_row, pos_col, pos_height, pos_width = pos
                    if (inner_row < pos_row + pos_height and inner_row + inner_height > pos_row and
                        inner_col < pos_col + pos_width and inner_col + inner_width > pos_col):
                        overlap = True
                        break
                
                if not overlap:
                    # Place inner object
                    for r in range(inner_row, inner_row + inner_height):
                        for c in range(inner_col, inner_col + inner_width):
                            grid[r, c] = inner_color
                    
                    # Record position
                    inner_positions.append((inner_row, inner_col, inner_height, inner_width))
                    break
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        # Find all objects in the grid
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        
        # Find the largest object (the rectangle)
        if len(objects) == 0:
            return np.zeros((1, 1), dtype=int)  # Fallback if no objects found
        
        rectangle = objects.sort_by_size(reverse=True)[0]
        
        # Get the bounding box
        box = rectangle.bounding_box
        
        # Extract the rectangular region
        rect_region = grid[box[0], box[1]].copy()
        
        # Flip horizontally
        output_grid = np.fliplr(rect_region)
        
        return output_grid

