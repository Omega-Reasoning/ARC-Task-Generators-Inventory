from arc_task_generator import ARCTaskGenerator, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple

class Task8d510a79(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['grid_size']} × {vars['grid_size']}.",
            "Each input grid contains a completely filled row with {color('middle_row')} color , which can be any row within two rows above or below the middle row.",
            "Several {color('cell_color1')} and {color('cell_color2')} cells are placed above and below the {color('middle_row')} line.",
            "There can be at most one colored cell in each column above and at most one colored cell in each column below the {color('middle_row')} line."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the {color('middle_row')} as well as the {color('cell_color1')} and {color('cell_color2')} cells.",
            "The goal is to extend the colored cells based on their type.",
            "The {color('cell_color1')} cells are extended toward the {color('middle_row')}. If a {color('cell_color1')} lies above the line, it extends downward; if it lies below, it extends upward.",
            "The {color('cell_color2')} cells are extended toward the nearest boundary row. If a {color('cell_color2')} lies above the {color('middle_row')}, it extends upward toward the first row; if it lies below, it extends downward toward the last row.",
            "The {color('middle_row')} remains unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        middle_row_color = taskvars['middle_row']
        cell_color1 = taskvars['cell_color1']
        cell_color2 = taskvars['cell_color2']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Determine middle row position (within 2 rows of center)
        center = grid_size // 2
        middle_row_pos = gridvars.get('middle_row_pos', 
                                    random.randint(max(2, center - 2), min(grid_size - 3, center + 2)))
        
        # Fill the middle row completely
        grid[middle_row_pos, :] = middle_row_color
        
        # Get available positions above and below middle row
        # Exclude first/last rows and rows adjacent to middle row
        above_rows = list(range(1, middle_row_pos - 1)) if middle_row_pos > 2 else []
        below_rows = list(range(middle_row_pos + 2, grid_size - 1)) if middle_row_pos < grid_size - 3 else []
        
        # Get all valid columns
        all_cols = list(range(grid_size))
        
        # Ensure we have at least 2 cells of each color, with cells both above and below middle row
        color1_count = 0
        color2_count = 0
        placed_positions = set()
        cells_above = 0
        cells_below = 0
        
        # First, ensure minimum requirements: at least one cell above and below middle row
        while color1_count < 2 or color2_count < 2 or cells_above == 0 or cells_below == 0:
            # Choose random position, prioritizing areas that need cells
            if above_rows and below_rows:
                if cells_above == 0:
                    available_rows = above_rows
                elif cells_below == 0:
                    available_rows = below_rows
                else:
                    use_above = random.choice([True, False])
                    available_rows = above_rows if use_above else below_rows
            elif above_rows:
                available_rows = above_rows
            elif below_rows:
                available_rows = below_rows
            else:
                break  # No valid positions available
            
            if not available_rows:
                break
                
            col = random.choice(all_cols)
            row = random.choice(available_rows)
            
            # Skip if position already occupied or column already has a cell above/below middle
            if (row, col) in placed_positions:
                continue
                
            # Check if this column already has a colored cell above or below middle
            has_cell_above = any(grid[r, col] != 0 for r in range(0, middle_row_pos) if r != middle_row_pos)
            has_cell_below = any(grid[r, col] != 0 for r in range(middle_row_pos + 1, grid_size) if r != middle_row_pos)
            
            if (row < middle_row_pos and has_cell_above) or (row > middle_row_pos and has_cell_below):
                continue  # Column already has a cell on this side
            
            # Decide which color to place based on what we need
            if color1_count < 2 and color2_count < 2:
                color = random.choice([cell_color1, cell_color2])
            elif color1_count < 2:
                color = cell_color1
            else:
                color = cell_color2
            
            grid[row, col] = color
            placed_positions.add((row, col))
            
            if color == cell_color1:
                color1_count += 1
            else:
                color2_count += 1
                
            # Track whether we have cells above and below
            if row < middle_row_pos:
                cells_above += 1
            else:
                cells_below += 1
        
        # Add additional random cells with probability
        for col in range(grid_size):
            # Skip columns that already have cells
            has_cell_above = any(grid[r, col] != 0 for r in range(0, middle_row_pos) if r != middle_row_pos)
            has_cell_below = any(grid[r, col] != 0 for r in range(middle_row_pos + 1, grid_size) if r != middle_row_pos)
            
            if random.random() < 0.3:  # 30% chance for additional cells
                # Above the middle row
                if above_rows and not has_cell_above and random.random() < 0.5:
                    row = random.choice(above_rows)
                    color = random.choice([cell_color1, cell_color2])
                    grid[row, col] = color
                
                # Below the middle row  
                elif below_rows and not has_cell_below and random.random() < 0.5:
                    row = random.choice(below_rows)
                    color = random.choice([cell_color1, cell_color2])
                    grid[row, col] = color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        middle_row_color = taskvars['middle_row']
        cell_color1 = taskvars['cell_color1']
        cell_color2 = taskvars['cell_color2']
        
        output_grid = grid.copy()
        
        # Find the middle row (completely filled with middle_row_color)
        middle_row_pos = None
        for row in range(grid_size):
            if np.all(grid[row, :] == middle_row_color) and np.any(grid[row, :] != 0):
                middle_row_pos = row
                break
        
        if middle_row_pos is None:
            return output_grid  # No middle row found, return unchanged
        
        # Process each column
        for col in range(grid_size):
            # Find cell_color1 and cell_color2 cells in this column
            for row in range(grid_size):
                if row == middle_row_pos:
                    continue  # Skip the middle row
                
                cell_value = grid[row, col]
                
                if cell_value == cell_color1:
                    # Extend toward middle row
                    if row < middle_row_pos:
                        # Above middle row, extend downward
                        for extend_row in range(row + 1, middle_row_pos):
                            output_grid[extend_row, col] = cell_color1
                    elif row > middle_row_pos:
                        # Below middle row, extend upward
                        for extend_row in range(middle_row_pos + 1, row):
                            output_grid[extend_row, col] = cell_color1
                
                elif cell_value == cell_color2:
                    # Extend toward boundary
                    if row < middle_row_pos:
                        # Above middle row, extend upward to boundary
                        for extend_row in range(0, row):
                            output_grid[extend_row, col] = cell_color2
                    elif row > middle_row_pos:
                        # Below middle row, extend downward to boundary
                        for extend_row in range(row + 1, grid_size):
                            output_grid[extend_row, col] = cell_color2
        
        return output_grid

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        grid_size = random.choice([7, 9, 11, 13, 15, 17, 19])  # Odd sizes between 5 and 30
        
        # Choose three different colors
        colors = list(range(1, 10))  # Colors 1-9 (excluding 0 which is background)
        chosen_colors = random.sample(colors, 3)
        
        taskvars = {
            'grid_size': grid_size,
            'middle_row': chosen_colors[0],
            'cell_color1': chosen_colors[1], 
            'cell_color2': chosen_colors[2]
        }
        
        # Generate training examples (3-5)
        num_train = random.randint(3, 5)
        train_examples = []
        
        for _ in range(num_train):
            # Vary middle row position for diversity
            center = grid_size // 2
            middle_row_pos = random.randint(max(2, center - 2), min(grid_size - 3, center + 2))
            gridvars = {'middle_row_pos': middle_row_pos}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        center = grid_size // 2
        test_middle_row_pos = random.randint(max(2, center - 2), min(grid_size - 3, center + 2))
        test_gridvars = {'middle_row_pos': test_middle_row_pos}
        
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data

