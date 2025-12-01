from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Task67a423a3Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square and vary in size.",
            "They contain one completely filled row and one completely filled column, with all remaining cells being empty (0).",
            "The filled row and column use different colors, which vary across examples.",
            "The filled row and column can be placed at any position except the first and last rows or columns.",
            "Any of the two lines (either the row or the column) can take priority for the intersection cell color"
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same size as the input.",
            "They are constructed by copying the input grid and identifying the cell where the completely filled row and column intersect.",
            "Once identified, a 3x3 {color('frame_color')} frame is drawn around the intersection cell.",
            "The {color('frame_color')} frame overlaps two cells from the filled row and two from the filled column.",
            "The intersection cell remains unchanged."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'frame_color': random.randint(1, 9)
        }
        
        # Create 3-4 training examples
        num_train_examples = random.randint(3, 4)
        train_examples = []
        
        # Track used positions and color combinations to ensure diversity
        # store (grid_size, row_pos, col_pos) to avoid accidental reuse across different sizes
        used_positions = set()
        used_color_pairs = set()
        
        # Ensure we have both row priority and column priority in the training examples
        has_row_priority_example = False
        has_col_priority_example = False
        
        for i in range(num_train_examples):
            # Determine row priority (ensure both cases are covered)
            if not has_row_priority_example:
                row_priority = True
                has_row_priority_example = True
            elif not has_col_priority_example:
                row_priority = False
                has_col_priority_example = True
            else:
                row_priority = random.choice([True, False])
            
            # Generate row and column positions (grid size chosen per-example)
            grid_size = random.randint(7, 30)
            while True:
                row_pos = random.randint(1, grid_size-2)  # Avoid first and last
                col_pos = random.randint(1, grid_size-2)  # Avoid first and last

                if (grid_size, row_pos, col_pos) not in used_positions:
                    used_positions.add((grid_size, row_pos, col_pos))
                    break
            
            # Generate unique colors for row and column
            frame_color = taskvars['frame_color']
            while True:
                available_colors = [i for i in range(1, 10) if i != frame_color]
                row_color = random.choice(available_colors)
                available_colors.remove(row_color)
                col_color = random.choice(available_colors)
                
                color_pair = (row_color, col_color)
                if color_pair not in used_color_pairs and (col_color, row_color) not in used_color_pairs:
                    used_color_pairs.add(color_pair)
                    break
            
            # Create grid variables for this example
            gridvars = {
                'grid_size': grid_size,
                'row_pos': row_pos,
                'col_pos': col_pos,
                'row_color': row_color,
                'col_color': col_color,
                'row_priority': row_priority
            }
            
            # Create input and output grids
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)

            train_examples.append({'input': input_grid, 'output': output_grid})

        # Create test example
        # Generate a grid size and new unique position for the test
        test_grid_size = random.randint(7, 30)
        while True:
            test_row_pos = random.randint(1, test_grid_size-2)
            test_col_pos = random.randint(1, test_grid_size-2)

            if (test_grid_size, test_row_pos, test_col_pos) not in used_positions:
                used_positions.add((test_grid_size, test_row_pos, test_col_pos))
                break

        # Generate new unique colors
        frame_color = taskvars['frame_color']
        while True:
            available_colors = [i for i in range(1, 10) if i != frame_color]
            test_row_color = random.choice(available_colors)
            available_colors.remove(test_row_color)
            test_col_color = random.choice(available_colors)

            color_pair = (test_row_color, test_col_color)
            if color_pair not in used_color_pairs and (test_col_color, test_row_color) not in used_color_pairs:
                break

        # Create test grid variables
        test_gridvars = {
            'grid_size': test_grid_size,
            'row_pos': test_row_pos,
            'col_pos': test_col_pos,
            'row_color': test_row_color,
            'col_color': test_col_color,
            'row_priority': random.choice([True, False])
        }

        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)

        # Prepare train/test data
        train_test_data = {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }

        return taskvars, train_test_data
    
    def create_input(self, taskvars, gridvars) -> np.ndarray:
        # Prefer per-example grid size; fall back to taskvars bounds if not present
        grid_size = gridvars.get('grid_size', taskvars.get('grid_size', taskvars.get('min_grid_size', 7)))
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Fill the row
        row_pos = gridvars['row_pos']
        row_color = gridvars['row_color']
        grid[row_pos, :] = row_color
        
        # Fill the column
        col_pos = gridvars['col_pos']
        col_color = gridvars['col_color']
        grid[:, col_pos] = col_color
        
        # Set intersection cell color based on priority
        if gridvars['row_priority']:
            grid[row_pos, col_pos] = row_color
        else:
            grid[row_pos, col_pos] = col_color
            
        return grid
    
    def transform_input(self, grid, taskvars) -> np.ndarray:
        # Copy the input grid
        output_grid = grid.copy()
        frame_color = taskvars['frame_color']
        
        # Find the intersection point
        rows_filled = []
        cols_filled = []
        
        # Identify filled rows and columns (excluding first and last)
        for i in range(1, grid.shape[0]-1):
            if np.all(grid[i, :] != 0):
                rows_filled.append(i)
        
        for j in range(1, grid.shape[1]-1):
            if np.all(grid[:, j] != 0):
                cols_filled.append(j)
        
        # There should be exactly one filled row and one filled column
        row_pos = rows_filled[0]
        col_pos = cols_filled[0]
        
        # Draw a 3x3 frame around the intersection
        for i in range(row_pos-1, row_pos+2):
            for j in range(col_pos-1, col_pos+2):
                # Create frame (outer cells only)
                if (i == row_pos-1 or i == row_pos+1 or 
                    j == col_pos-1 or j == col_pos+1):
                    # Ensure in bounds
                    if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
                        output_grid[i, j] = frame_color
        
        return output_grid


