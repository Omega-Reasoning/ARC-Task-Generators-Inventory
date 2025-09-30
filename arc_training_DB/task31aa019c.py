import numpy as np
import random
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import retry, create_object
from transformation_library import find_connected_objects

class ARCTask31aa019cGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['rows']}.",
            "The grid contains different colors cells(numbers between 1-9), where exactly one cell has a unique color that appears only once.",
            "All other colored cells appear at least twice in the grid.",
            "The fill color {color('fill_color')} never appears in the input grid.",
            "The remaining cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "Find the cell with the color which only appears once in the input grid.",
            "Keep this unique colored cell in the output grid.",
            "Color all the cells surrounding this unique cell (including diagonals) with {color('fill_color')}.",
            "All other cells are empty(0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        grid = np.zeros((rows, rows), dtype=int)
        fill_color = taskvars['fill_color']
        
        # Choose a unique color (different from fill_color)
        available_colors = list(set(range(1, 10)) - {fill_color})
        unique_color = random.choice(available_colors)
        
        # Choose position for unique color (not on edge)
        row, col = random.randint(1, rows - 2), random.randint(1, rows - 2)
        grid[row, col] = unique_color
        
        # Select colors that will appear multiple times (excluding unique_color and fill_color)
        other_colors = list(set(available_colors) - {unique_color})
        num_other_colors = random.randint(4, min(6, len(other_colors)))  # Use 2-4 different colors
        selected_colors = random.sample(other_colors, num_other_colors)
        
        # First, ensure each selected color appears at least twice
        for color in selected_colors:
            placed = 0
            while placed < 2:  # Place each color at least twice
                r, c = random.randint(0, rows - 1), random.randint(0, rows - 1)
                if grid[r, c] == 0:  # Only fill empty cells
                    grid[r, c] = color
                    placed += 1
        
        # Then add some more random occurrences of these colors
        for _ in range(random.randint(rows, rows * 2)):
            r, c = random.randint(0, rows - 1), random.randint(0, rows - 1)
            if grid[r, c] == 0:  # Only fill empty cells
                color = random.choice(selected_colors)  # Only use colors we've already used
                grid[r, c] = color
        
        return grid
    
    def transform_input(self, grid, taskvars):
        rows, cols = grid.shape
        output_grid = np.zeros_like(grid)
        unique_color = None
        
        # Find the unique color
        unique_color_count = {}
        for r in range(rows):
            for c in range(cols):
                color = grid[r, c]
                if color != 0:
                    unique_color_count[color] = unique_color_count.get(color, 0) + 1
        
        for color, count in unique_color_count.items():
            if count == 1:
                unique_color = color
                break
        
        if unique_color is None:
            return output_grid  # Fallback case
        
        # Locate the unique color's position
        unique_position = None
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == unique_color:
                    unique_position = (r, c)
                    break
            if unique_position:
                break
        
        fill_color = taskvars['fill_color']
        r, c = unique_position
        
        # Keep the unique color cell in the output
        output_grid[r, c] = unique_color
        
        # Fill surrounding cells (including diagonals)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:  # Skip the center cell
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    output_grid[nr, nc] = fill_color
        
        return output_grid
    
    def create_grids(self):
        taskvars = {
            'rows': random.randint(10, 30),
            'fill_color': random.choice(range(1, 10))
        }
        
        train_test_data = {
            'train': [],
            'test': []
        }
        
        num_train = random.randint(3, 6)
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_test_data['train'].append({'input': input_grid, 'output': output_grid})
        
        input_grid = self.create_input(taskvars, {})
        output_grid = self.transform_input(input_grid, taskvars)
        train_test_data['test'].append({'input': input_grid, 'output': output_grid})
        
        return taskvars, train_test_data

