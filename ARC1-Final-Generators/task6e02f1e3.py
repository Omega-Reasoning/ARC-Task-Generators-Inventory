from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, GridObject, GridObjects

class Task6e02f1e3Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']} x {vars['grid_size']}.",
            "The grids are completely filled with colored objects, where each object consists of 4-way connected cells of the same color.",
            "The possible grid colors are: {color('object_color1')}, {color('object_color2')}, and {color('object_color3')}.",
            "The grid may be filled with one single color or a combination of two or three colors.",
            "The shapes of the objects should vary across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by identifying the number of different colors used in the input grid.",
            "If exactly one color is used, the first row of the output is filled with {color('object_color4')} color.",
            "If two colors are used, the main diagonal (top-left to bottom-right) of the output is filled with {color('object_color4')} color.",
            "If three colors are used, the inverse diagonal (top-right to bottom-left) of the output is filled with {color('object_color4')} color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Initialize task variables
        taskvars = {
            'grid_size': random.randint(5, 30),  # Keep reasonable size for testing
            'object_color1': random.randint(1, 9),
            'object_color2': 0,
            'object_color3': 0,
            'object_color4': 0
        }
        
        # Ensure all colors are different
        colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        colors.remove(taskvars['object_color1'])
        taskvars['object_color2'] = random.choice(colors)
        colors.remove(taskvars['object_color2'])
        taskvars['object_color3'] = random.choice(colors)
        colors.remove(taskvars['object_color3'])
        taskvars['object_color4'] = random.choice(colors)
        
        # Generate train and test data
        train_data = []
        test_data = []
        
        # Ensure we have required variations
        single_color_train = False
        two_color_train = False
        three_color_train = False
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        for i in range(num_train_examples):
            if i == 0 or not single_color_train:
                # One color example
                gridvars = {'num_colors': 1}
                single_color_train = True
            elif i == 1 or not two_color_train:
                # Two color example
                gridvars = {'num_colors': 2}
                two_color_train = True
            elif i == 2 or not three_color_train:
                # Three color example
                gridvars = {'num_colors': 3}
                three_color_train = True
            else:
                # Random example for remaining cases
                gridvars = {'num_colors': random.randint(1, 3)}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})
        
        # Generate 3 test examples (one with 1 color, one with 2 colors, one with 3 colors)
        for num_colors in [1, 2, 3]:
            gridvars = {'num_colors': num_colors}
            # For the single color test, use a different color than the single color train example
            if num_colors == 1:
                # Get color used in the single-color train example
                train_single_color = None
                for pair in train_data:
                    colors_used = set(pair['input'].flatten()) - {0}
                    if len(colors_used) == 1:
                        train_single_color = list(colors_used)[0]
                        break
                
                gridvars['single_color'] = train_single_color
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            test_data.append({'input': input_grid, 'output': output_grid})
        
        return taskvars, {'train': train_data, 'test': test_data}
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        num_colors = gridvars.get('num_colors', random.randint(1, 3))
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Get available colors
        available_colors = [taskvars['object_color1'], taskvars['object_color2'], taskvars['object_color3']]
        
        # For single color test examples, ensure a different color than train
        if num_colors == 1 and 'single_color' in gridvars and gridvars['single_color'] is not None:
            available_colors = [c for c in available_colors if c != gridvars['single_color']]
        
        # Shuffle colors to increase diversity
        random.shuffle(available_colors)
        colors = available_colors[:num_colors]
        
        # For single color case, fill entire grid
        if num_colors == 1:
            grid.fill(colors[0])
            return grid
        
        # For multiple colors, create connected regions
        remaining_coords = set((r, c) for r in range(grid_size) for c in range(grid_size))
        
        for color_idx in range(num_colors - 1):  # Reserve space for the last color
            color = colors[color_idx]
            
            # Choose random seed point for growing region
            if not remaining_coords:
                break
                
            seed = random.choice(list(remaining_coords))
            remaining_coords.remove(seed)
            region = {seed}
            grid[seed] = color
            
            # Grow region randomly
            frontier = list(region)
            target_size = len(remaining_coords) // (num_colors - color_idx)
            
            while frontier and len(region) < target_size:
                current = frontier.pop(random.randint(0, len(frontier)-1))
                r, c = current
                
                # Try to add neighbors
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < grid_size and 0 <= nc < grid_size and 
                        (nr, nc) in remaining_coords):
                        grid[nr, nc] = color
                        remaining_coords.remove((nr, nc))
                        region.add((nr, nc))
                        frontier.append((nr, nc))
        
        # Fill remaining cells with the last color
        last_color = colors[-1]
        for r, c in remaining_coords:
            grid[r, c] = last_color
            
        # Verify that we have the correct number of connected objects
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)
        if len(objects) != num_colors:
            # If we don't have the right number of objects, retry
            return self.create_input(taskvars, gridvars)
            
        return grid
    
    def transform_input(self, grid, taskvars):
        grid_size = taskvars['grid_size']
        output_grid = np.zeros((grid_size, grid_size), dtype=int)
        highlight_color = taskvars['object_color4']
        
        # Count unique colors in the input grid (excluding background)
        unique_colors = set(grid.flatten()) - {0}
        num_colors = len(unique_colors)
        
        # Apply the transformation based on number of colors
        if num_colors == 1:
            # Fill first row
            output_grid[0, :] = highlight_color
        elif num_colors == 2:
            # Fill main diagonal
            for i in range(grid_size):
                output_grid[i, i] = highlight_color
        elif num_colors == 3:
            # Fill inverse diagonal
            for i in range(grid_size):
                output_grid[i, grid_size - 1 - i] = highlight_color
                
        return output_grid

