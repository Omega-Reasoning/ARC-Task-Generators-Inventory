from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import random_cell_coloring, retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taske3497940Generator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} x n, where n is an odd integer number and varies in each input grid.",
            "Each grid contains a central vertical column (at index n // 2), consistently colored with {color('middle_color')}, effectively dividing the grid into left and right halves.",
            "On both sides of the central column, a random number of cells are filled with two distinct random colors. These colored cells primarily form short horizontal bar-like shapes perpendicular to the central column.",
            "The density of these colored cells is highest near the middle column and gradually decreases toward the outer edges. Specifically, columns n//2 - 1 and n//2 + 1 have the highest number of colored cells, followed by n//2 - 2 and n//2 + 2, and so on.",
            "For each colored cell, its symmetric counterpart—reflected across the middle column—is either empty or colored with the same color.",
            "Colored cells of both colors are present in the grid.",
            "At most half of the cells of each half are colored."
        ]
        
        transformation_reasoning_chain = [
            "Output grids are of size {vars['rows']}x (n // 2).",
            "The central vertical column of the input grid, located at index n // 2, is first identified.",
            "The right half of the input grid (the columns to the right of the central column) is then extracted.",
            "The left half of the input grid (the columns to the left of the central column) is then extracted.",
            "This right half is horizontally flipped.",
            "The output grid is the union of the flipped right half and the left half of the input grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def get_symmetric_position(self, row: int, col: int, middle_col: int) -> Tuple[int, int]:
        """Get the symmetric position of a cell across the middle column."""
        symmetric_col = middle_col + (middle_col - col)
        return row, symmetric_col
    
    def is_valid_placement(self, grid: np.ndarray, row: int, col: int, color: int, middle_col: int) -> bool:
        """Check if placing a color at (row, col) violates the symmetry constraint."""
        rows, cols = grid.shape
        
        # Get symmetric position
        sym_row, sym_col = self.get_symmetric_position(row, col, middle_col)
        
        # If symmetric position is out of bounds, placement is valid
        if not (0 <= sym_col < cols):
            return True
        
        # If symmetric position is the middle column, placement is valid
        if sym_col == middle_col:
            return True
            
        # Check the symmetry constraint
        existing_color = grid[sym_row, sym_col]
        # Valid if symmetric cell is empty OR has the same color
        return existing_color == 0 or existing_color == color
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = gridvars['cols']
        middle_color = taskvars['middle_color']
        color1 = gridvars['color1']
        color2 = gridvars['color2']
        
        # Initialize grid with background (0)
        grid = np.zeros((rows, cols), dtype=int)
        
        # Create central vertical column
        middle_col = cols // 2
        grid[:, middle_col] = middle_color
        
        # Calculate half width (excluding middle column)
        half_width = middle_col
        
        colors = [color1, color2]
        color_counts = {color1: 0, color2: 0}
        
        # Define target counts for each distance - strict decreasing order
        max_cells = min(rows - 1, max(4, rows // 2))
        target_counts = {}
        for distance in range(1, half_width + 1):
            target_counts[distance] = max(1, max_cells - (distance - 1) * 2)
        
        # Place cells column by column, respecting density gradient
        for distance in range(1, half_width + 1):
            left_col = middle_col - distance
            right_col = middle_col + distance
            target_cells = target_counts[distance]
            
            # Get available rows
            available_rows = list(range(rows))
            random.shuffle(available_rows)
            
            # Place cells in left column
            left_placed = 0
            for row in available_rows:
                if left_placed >= target_cells:
                    break
                    
                if left_col >= 0 and grid[row, left_col] == 0:
                    # Choose color and bar length
                    bar_color = random.choice(colors)
                    bar_length = random.randint(1, min(3, distance + 1))
                    
                    # Place horizontal bar extending leftward
                    for i in range(bar_length):
                        place_col = left_col - i
                        if place_col < 0 or grid[row, place_col] != 0:
                            break
                            
                        # Check if this placement is valid according to symmetry constraint
                        if self.is_valid_placement(grid, row, place_col, bar_color, middle_col):
                            grid[row, place_col] = bar_color
                            color_counts[bar_color] += 1
                            left_placed += 1
            
            # Place cells in right column
            available_rows = list(range(rows))
            random.shuffle(available_rows)
            
            right_placed = 0
            for row in available_rows:
                if right_placed >= target_cells:
                    break
                    
                if right_col < cols and grid[row, right_col] == 0:
                    # Choose color and bar length
                    bar_color = random.choice(colors)
                    bar_length = random.randint(1, min(3, cols - right_col))
                    
                    # Place horizontal bar extending rightward
                    placed_in_this_bar = 0
                    for i in range(bar_length):
                        place_col = right_col + i
                        if place_col >= cols or grid[row, place_col] != 0:
                            break
                            
                        # Check if this placement is valid according to symmetry constraint
                        if self.is_valid_placement(grid, row, place_col, bar_color, middle_col):
                            # Also add some randomness to make it less symmetric
                            if random.random() < 0.7:  # 70% chance to actually place
                                grid[row, place_col] = bar_color
                                color_counts[bar_color] += 1
                                right_placed += 1
                                placed_in_this_bar += 1
                    
                    # If we placed any cells in this bar, count it as progress
                    if placed_in_this_bar == 0:
                        continue
        
        # Ensure both colors are present
        if color_counts[color1] == 0:
            # Find a valid position for color1
            for row in range(rows):
                for col in [middle_col - 1, middle_col + 1]:
                    if 0 <= col < cols and grid[row, col] == 0:
                        if self.is_valid_placement(grid, row, col, color1, middle_col):
                            grid[row, col] = color1
                            break
                else:
                    continue
                break
            
        if color_counts[color2] == 0:
            # Find a valid position for color2
            for row in range(rows):
                for col in [middle_col + 1, middle_col - 1]:
                    if 0 <= col < cols and grid[row, col] == 0:
                        if self.is_valid_placement(grid, row, col, color2, middle_col):
                            grid[row, col] = color2
                            break
                else:
                    continue
                break
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows, cols = grid.shape
        middle_col = cols // 2
        
        # Extract left half (columns 0 to middle_col-1)
        left_half = grid[:, :middle_col]
        
        # Extract right half (columns middle_col+1 to end)
        right_half = grid[:, middle_col+1:]
        
        # Flip right half horizontally
        flipped_right_half = np.fliplr(right_half)
        
        # Create output by combining (union) left half and flipped right half
        output = np.maximum(left_half, flipped_right_half)
        
        return output
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'rows': random.randint(8, 15),  # Ensure rows > cols constraint
            'middle_color': random.randint(1, 9)
        }
        
        # Generate different colors for objects
        available_colors = [i for i in range(1, 10) if i != taskvars['middle_color']]
        
        # Create training examples
        num_train = random.randint(3, 6)
        train_pairs = []
        
        for _ in range(num_train):
            # Generate grid-specific variables
            cols = random.choice([7, 9, 11, 13, 15])  # Odd numbers, ensure cols < rows
            while cols >= taskvars['rows']:
                cols = random.choice([7, 9, 11, 13, 15])
            
            colors = random.sample(available_colors, 2)
            gridvars = {
                'cols': cols,
                'color1': colors[0],
                'color2': colors[1]
            }
            
            # Generate input and output
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_pairs.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        cols = random.choice([7, 9, 11, 13, 15])
        while cols >= taskvars['rows']:
            cols = random.choice([7, 9, 11, 13, 15])
            
        colors = random.sample(available_colors, 2)
        gridvars = {
            'cols': cols,
            'color1': colors[0],
            'color2': colors[1]
        }
        
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        
        test_pairs = [{
            'input': input_grid,
            'output': output_grid
        }]
        
        return taskvars, {
            'train': train_pairs,
            'test': test_pairs
        }
