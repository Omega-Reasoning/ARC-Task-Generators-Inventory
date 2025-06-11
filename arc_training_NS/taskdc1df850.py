from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List, Set

class Taskdc1df850(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each input grid contains a number of main cells, which are single-colored cells of color {color('object_color')}. The number of main cells varies, but there is at least one in every input grid.",
            "Additionally, the grid contains random cells, which are single-colored cells of various colors different from {color('object_color')}.",
            "The number of random cells also varies, but it is always less than or equal to the number of main cells.",
            "Some random cells may share the same color; however, no color among the random cells appears as frequently as the main cell color.",
            "Each main cell is fully enclosed by a border of empty cells wherever the border lies within the grid, meaning it may be placed at an edge or corner as long as all neighboring cells inside the grid are empty.",
            "The border around each main cell is unique and non-overlapping with any other main cell — no two main cells share any border cell.",
            "The remaining cells are all empty."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "All single-colored cells are identified.",
            "The color that appears most frequently among all single-colored cells is determined. The cells with this color are identified as the main cells, and their color is {color('object_color')}.",
            "For each main cell, its surrounding border of empty cells is filled with the color {color('border_color')} in the output grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        taskvars = {
            'object_color': random.randint(1, 9),
            'border_color': random.randint(1, 9)
        }
        
        # Ensure object and border colors are different
        while taskvars['border_color'] == taskvars['object_color']:
            taskvars['border_color'] = random.randint(1, 9)
        
        # Generate 3-6 train examples and 1 test example
        num_train = random.randint(3, 6)
        
        train_examples = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        test_examples = []
        input_grid = self.create_input(taskvars, {})
        output_grid = self.transform_input(input_grid, taskvars)
        test_examples.append({'input': input_grid, 'output': output_grid})
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        object_color = taskvars['object_color']
        
        # Random grid size between 5 and 30
        height = random.randint(5, 30)
        width = random.randint(5, 30)
        
        grid = np.zeros((height, width), dtype=int)
        
        # Calculate maximum allowed colored cells (15% of total)
        max_colored_cells = int(0.15 * height * width)
        
        # Place main cells with proper borders
        num_main_cells = random.randint(1, min(8, max_colored_cells // 2))
        main_cell_positions = []
        occupied_positions = set()  # Track all cells that are occupied or reserved as borders
        
        attempts = 0
        while len(main_cell_positions) < num_main_cells and attempts < 1000:
            attempts += 1
            
            # Try to place a main cell
            r = random.randint(0, height - 1)
            c = random.randint(0, width - 1)
            
            if (r, c) in occupied_positions:
                continue
                
            # Check if we can create a border around this cell
            # Get all border positions (8-way connectivity)
            border_positions = []
            cell_and_border = [(r, c)]  # Include the cell itself
            
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue  # Skip the center cell
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        border_positions.append((nr, nc))
                        cell_and_border.append((nr, nc))
            
            # Check if any of these positions are already occupied
            if any(pos in occupied_positions for pos in cell_and_border):
                continue
            
            # Place the main cell
            grid[r, c] = object_color
            main_cell_positions.append((r, c))
            
            # Mark all positions (cell + border) as occupied
            occupied_positions.update(cell_and_border)
        
        # Place random cells
        remaining_cells = max_colored_cells - len(main_cell_positions)
        
        # Determine number of random cells
        if num_main_cells == 1:
            # If only one main cell, random cells are optional (0 to num_main_cells)
            num_random_cells = random.randint(0, min(remaining_cells, len(main_cell_positions)))
        else:
            # If multiple main cells, at least one random cell must exist
            min_random = 1
            max_random = min(remaining_cells, len(main_cell_positions))
            if max_random >= min_random:
                num_random_cells = random.randint(min_random, max_random)
            else:
                num_random_cells = 0  # Fallback if not enough space
        
        if num_random_cells > 0:
            # Get available positions for random cells
            available_positions = [(r, c) for r in range(height) for c in range(width) 
                                 if (r, c) not in occupied_positions]
            
            if len(available_positions) >= num_random_cells:
                random_positions = random.sample(available_positions, num_random_cells)
                
                # Assign colors to random cells, ensuring no color appears as frequently as main color
                available_colors = [i for i in range(1, 10) if i != object_color]
                
                # Distribute colors such that no single color reaches the frequency of main cells
                max_per_color = len(main_cell_positions) - 1
                if max_per_color <= 0:
                    max_per_color = 1
                
                color_counts = {}
                for pos in random_positions:
                    # Choose a color that hasn't reached its maximum
                    valid_colors = [c for c in available_colors 
                                  if color_counts.get(c, 0) < max_per_color]
                    if not valid_colors:
                        valid_colors = available_colors
                    
                    color = random.choice(valid_colors)
                    color_counts[color] = color_counts.get(color, 0) + 1
                    grid[pos[0], pos[1]] = color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        object_color = taskvars['object_color']
        border_color = taskvars['border_color']
        
        # Copy the input grid
        output_grid = grid.copy()
        
        # Find all single-colored cells and count frequencies
        color_counts = {}
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] != 0:
                    color_counts[grid[r, c]] = color_counts.get(grid[r, c], 0) + 1
        
        # Find the most frequent color (should be object_color)
        if color_counts:
            most_frequent_color = max(color_counts.keys(), key=lambda k: color_counts[k])
            
            # Find all cells of the most frequent color
            main_cells = []
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    if grid[r, c] == most_frequent_color:
                        main_cells.append((r, c))
            
            # For each main cell, fill its surrounding empty border
            for r, c in main_cells:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue  # Skip the center cell
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and 
                            grid[nr, nc] == 0):  # Only fill empty cells
                            output_grid[nr, nc] = border_color
        
        return output_grid
