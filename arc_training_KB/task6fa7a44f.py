from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, random_cell_coloring, retry
from transformation_library import find_connected_objects, GridObject, GridObjects

class Task6fa7a44fGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain a completely filled grid with colored objects (1-9), where each object is made of 4-way connected cells of the same color.",
            "Some of the objects can also be 1x1 blocks.",
            "The color, size, and shape of objects vary across examples."
        ]
        
        transformation_reasoning_chain = [
            "Output grids are of size {2*vars['grid_size']}x{vars['grid_size']}.",
            "The output grid is constructed by copying the input grid and pasting it to the top half of the output grid.",
            "Once copied, reflect the top-half vertically below to fill the bottom half with the reflected copy of the input grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        
        # Get the style from gridvars - either 'single_cells' or 'connected_objects'
        style = gridvars.get('style', 'connected_objects')
        
        # Always use 3 to 4 colors
        num_colors = gridvars.get('num_colors', random.choice([3, 4]))
        colors = random.sample(range(1, 10), num_colors)  # Choose from colors 1-9
        
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        if style == 'single_cells':
            # Fill with 1x1 blocks (random colors)
            for r in range(grid_size):
                for c in range(grid_size):
                    grid[r, c] = random.choice(colors)
        else:
            # Generate connected objects
            remaining_cells = set((r, c) for r in range(grid_size) for c in range(grid_size))
            
            # First, determine how many cells will be 1x1 blocks (between 10-30% of grid)
            single_cell_ratio = gridvars.get('single_cell_ratio', random.uniform(0.1, 0.3))
            total_cells = grid_size * grid_size
            num_single_cells = int(total_cells * single_cell_ratio)
            
            # Create counters to track color usage
            color_usage = {color: 0 for color in colors}
            
            # Place single cells randomly
            for _ in range(num_single_cells):
                if not remaining_cells:
                    break
                    
                r, c = random.choice(list(remaining_cells))
                color = random.choice(colors)
                grid[r, c] = color
                color_usage[color] += 1
                remaining_cells.remove((r, c))
            
            # Fill the rest with connected objects
            while remaining_cells:
                if len(remaining_cells) < 3:  # If only a few cells left, make them 1x1
                    for r, c in remaining_cells:
                        color = random.choice(colors)
                        grid[r, c] = color
                        color_usage[color] += 1
                    break
                
                # Choose a random start point and color
                start_r, start_c = random.choice(list(remaining_cells))
                color = random.choice(colors)
                
                # Determine object size (3-15 cells or until we run out)
                obj_size = min(random.randint(3, 15), len(remaining_cells))
                
                # Start the object
                grid[start_r, start_c] = color
                color_usage[color] += 1
                remaining_cells.remove((start_r, start_c))
                obj_cells = {(start_r, start_c)}
                
                # Grow the object
                for _ in range(obj_size - 1):
                    # Find valid neighbors of current object cells
                    neighbors = []
                    for r, c in obj_cells:
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 4-way connectivity
                            nr, nc = r + dr, c + dc
                            if (nr, nc) in remaining_cells:
                                neighbors.append((nr, nc))
                    
                    if not neighbors:
                        break
                        
                    # Add a random neighbor to the object
                    next_r, next_c = random.choice(neighbors)
                    grid[next_r, next_c] = color
                    color_usage[color] += 1
                    remaining_cells.remove((next_r, next_c))
                    obj_cells.add((next_r, next_c))
            
            # Ensure all colors are used at least once
            unused_colors = [c for c, count in color_usage.items() if count == 0]
            if unused_colors:
                # Replace some cells with the unused colors
                for unused_color in unused_colors:
                    # Find cells to replace
                    replaceable_cells = []
                    for r in range(grid_size):
                        for c in range(grid_size):
                            # Can replace any 1x1 block or any cell that won't break connectivity
                            replaceable_cells.append((r, c))
                    
                    if replaceable_cells:
                        r, c = random.choice(replaceable_cells)
                        grid[r, c] = unused_color
        
        # Verify we have at least 3 colors in the grid
        unique_colors = set(grid.flatten()) - {0}
        if len(unique_colors) < 3:
            # Add missing colors
            needed = 3 - len(unique_colors)
            missing_colors = [c for c in colors if c not in unique_colors][:needed]
            
            for color in missing_colors:
                r, c = random.randrange(grid_size), random.randrange(grid_size)
                grid[r, c] = color
        
        return grid
    
    def transform_input(self, grid, taskvars):
        # Get the actual grid dimensions directly from the input grid
        rows, cols = grid.shape
        
        # Create output grid with twice the height
        output = np.zeros((2 * rows, cols), dtype=int)
        
        # Copy input grid to top half
        output[:rows, :] = grid
        
        # Reflect top half to bottom half
        for r in range(rows):
            output[2 * rows - 1 - r, :] = output[r, :]
        
        return output
    
    def create_grids(self):
        # Set up task variables
        taskvars = {
            'grid_size': random.randint(5, 30)  # Choose a grid size between 5 and 15
        }
        
        # Create 3 train pairs and 1 test pair
        train_pairs = []
        
        # First train example: grid with 1x1 blocks
        input_grid = self.create_input(taskvars, {'style': 'single_cells', 'num_colors': random.choice([3, 4])})
        output_grid = self.transform_input(input_grid, taskvars)
        train_pairs.append({'input': input_grid, 'output': output_grid})
        
        # Second and third train examples: grids with connected objects
        for _ in range(2):
            # Vary the ratio of single cells but ensure 3-4 colors
            gridvars = {
                'style': 'connected_objects',
                'num_colors': random.choice([3, 4]),
                'single_cell_ratio': random.uniform(0.1, 0.3)
            }
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append({'input': input_grid, 'output': output_grid})
        
        # Test example: another grid with connected objects
        test_gridvars = {
            'style': 'connected_objects',
            'num_colors': random.choice([3, 4]),
            'single_cell_ratio': random.uniform(0.1, 0.3)
        }
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        return taskvars, {
            'train': train_pairs,
            'test': [{'input': test_input, 'output': test_output}]
        }

# Test the task generator
if __name__ == "__main__":
    generator = VerticalReflectionTaskGenerator()
    _, train_test_data = generator.create_grids()
    ARCTaskGenerator.visualize_train_test_data(train_test_data)