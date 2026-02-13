from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from collections import Counter
from typing import Dict, List, Any, Tuple

class Task5582e5caGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain multi-colored (1-9) and several empty (0) cells.",
            "Most grids are completely filled with multi-colored (1-9) cells, while one grid may also contain several empty (0) cells.",
            "Each grid always has one color that appears more frequently than all other colors and empty (0) cells, with no ties to resolve.",
            "The color that appears most frequently should vary across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is initialized as a zero-filled grid with the same size as the input grid.",
            "Next, identify the most frequent cell color (1-9) in the input grid.",
            "Once identified, fill the entire output grid with this most frequent color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Define task variables
        taskvars = {
            'grid_size': random.randint(4, 30)  # Random size between 4x4 and 10x10
        }
        
        # Track used colors to ensure diversity
        used_frequent_colors = set()
        
        # Generate 3-4 train examples
        num_train_examples = random.randint(3, 4)
        train_examples = []
        
        # Decide which example will have empty cells (if any)
        example_with_empty = random.randint(0, num_train_examples)
        
        for i in range(num_train_examples):
            # Decide if this grid should have empty cells
            has_empty = (i == example_with_empty)
            
            # Choose a frequent color that hasn't been used yet
            available_colors = list(set(range(1, 10)) - used_frequent_colors)
            if not available_colors:  # If all colors used, reset
                available_colors = list(range(1, 10))
            
            frequent_color = random.choice(available_colors)
            used_frequent_colors.add(frequent_color)
            
            # Create gridvars to specify this grid's properties
            gridvars = {
                'has_empty': has_empty,
                'frequent_color': frequent_color
            }
            
            # Create the input/output pair
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate one test example with a different most frequent color
        available_colors = list(set(range(1, 10)) - used_frequent_colors)
        if not available_colors:  # If all colors used, choose any
            available_colors = list(range(1, 10))
        
        test_frequent_color = random.choice(available_colors)
        
        # Randomly decide if test has empty cells
        test_has_empty = random.choice([True, False])
        
        test_gridvars = {
            'has_empty': test_has_empty,
            'frequent_color': test_frequent_color
        }
        
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        has_empty = gridvars['has_empty']
        frequent_color = gridvars['frequent_color']
        
        # Create an empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Define how many cells should have the frequent color
        # The frequent color should appear in 30-50% of cells
        frequent_count = random.randint(int(grid_size * grid_size * 0.3), 
                                        int(grid_size * grid_size * 0.5))
        
        # If we have empty cells, decide how many (5-15% of cells)
        empty_count = 0
        if has_empty:
            empty_count = random.randint(int(grid_size * grid_size * 0.05),
                                        int(grid_size * grid_size * 0.15))
        
        # Calculate how many cells remain for other colors
        remaining_cells = grid_size * grid_size - frequent_count - empty_count
        
        # Choose other colors (excluding the frequent color)
        other_colors = [c for c in range(1, 10) if c != frequent_color]
        
        # Ensure no other color appears more than the frequent color
        # Do this by calculating max cells per other color
        max_other_color_count = frequent_count - 1
        min_colors_to_use = min(2, len(other_colors))  # Use at least 2 other colors if possible
        
        # Distribute remaining cells among other colors
        other_color_counts = {}
        remaining = remaining_cells
        
        # First, ensure each required color gets at least one cell
        for i in range(min_colors_to_use):
            other_color_counts[other_colors[i]] = 1
            remaining -= 1
        
        # Now distribute the rest, ensuring no color gets more than max_per_color
        colors_to_use = list(other_colors)  # Make a copy to modify
        while remaining > 0 and colors_to_use:
            # Choose a random color to add a cell to
            eligible_colors = [c for c in colors_to_use if other_color_counts.get(c, 0) < max_other_color_count]
            if not eligible_colors:
                break
                
            color = random.choice(eligible_colors)
            other_color_counts[color] = other_color_counts.get(color, 0) + 1
            remaining -= 1
            
            # If this color reached max, remove it from options
            if other_color_counts[color] >= max_other_color_count:
                colors_to_use.remove(color)
        
        # Now fill the grid
        cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]
        random.shuffle(cells)
        
        cell_index = 0
        
        # First, place frequent color cells
        for _ in range(frequent_count):
            r, c = cells[cell_index]
            grid[r, c] = frequent_color
            cell_index += 1
        
        # Then, place other colored cells
        for color, count in other_color_counts.items():
            for _ in range(count):
                r, c = cells[cell_index]
                grid[r, c] = color
                cell_index += 1
        
        # Empty cells are already zero, so no need to explicitly set them
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Count occurrences of each color (excluding 0)
        color_counts = {}
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                color = grid[r, c]
                if color > 0:  # Only count non-zero colors
                    color_counts[color] = color_counts.get(color, 0) + 1
        
        # Find most frequent color
        most_frequent_color = max(color_counts.items(), key=lambda x: x[1])[0]
        
        # Create output grid filled with the most frequent color
        output_grid = np.full(grid.shape, most_frequent_color, dtype=int)
        
        return output_grid

