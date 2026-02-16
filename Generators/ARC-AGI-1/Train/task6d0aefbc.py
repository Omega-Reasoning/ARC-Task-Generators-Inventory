from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import create_object, Contiguity, retry, random_cell_coloring
from Framework.transformation_library import find_connected_objects

class Task6d0aefbcGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain a completely filled grid with at least two different colored objects using the colors {color('color1')}, {color('color2')}, and {color('color3')}.",
            "Each object is made of 4-way connected cells of the same color.",
            "The shapes and positions of the objects vary across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are of size {vars['grid_size']}x{2*vars['grid_size']}.",
            "They are constructed by copying the input grid and pasting it to the left half of the output grid.",
            "Then, the left half is reflected horizontally to fill the right half, creating a mirrored version of the input grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Initialize task variables
        grid_size = random.randint(5, 15)  # Keeping it smaller for better visualization
        
        # Choose 3 different colors between 1 and 9
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        color1, color2, color3 = available_colors[:3]
        
        taskvars = {
            'grid_size': grid_size,
            'color1': color1,
            'color2': color2,
            'color3': color3
        }
        
        # Create 3-4 train examples
        num_train = random.randint(3, 4)
        train_examples = []
        
        # Ensure we have one example with 2 colors and one with 3 colors
        colors_used_in_examples = []
        
        # Randomize which of the first three training examples is the two-color case
        # and ensure one of the first three is a three-color example as well.
        first_three_indices = [0, 1, 2]
        two_color_index = random.choice(first_three_indices)
        # pick a different index for the three-color example
        three_color_candidates = [i for i in first_three_indices if i != two_color_index]
        three_color_index = random.choice(three_color_candidates)

        for i in range(num_train):
            if i == two_color_index:
                colors_used = [color1, color2]
            elif i == three_color_index:
                colors_used = [color1, color2, color3]
            else:  # Other examples use 2 or 3 colors randomly (preserve original behavior)
                colors_used = [color1, color2]
                if random.choice([True, False]):
                    colors_used.append(color3)
            
            colors_used_in_examples.append(colors_used)
            
            gridvars = {'colors_used': colors_used}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create test example
        # For test, randomly choose 2 or 3 colors
        test_colors = [color1, color2]
        if random.choice([True, False]):
            test_colors.append(color3)
        
        gridvars = {'colors_used': test_colors}
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        return taskvars, {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        colors_used = gridvars['colors_used']
        
        # Initialize an empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Determine how many objects to create (2-5)
        num_objects = random.randint(2, 5)
        
        # Create random connected objects with different colors
        for i in range(num_objects):
            color = random.choice(colors_used)
            
            # Create object size (a fraction of the grid)
            obj_height = random.randint(2, max(3, grid_size // 2))
            obj_width = random.randint(2, max(3, grid_size // 2))
            
            # Create a connected object
            obj = create_object(
                height=obj_height,
                width=obj_width,
                color_palette=color,
                contiguity=Contiguity.FOUR,
                background=0
            )
            
            # Find a random position to place the object
            r_pos = random.randint(0, grid_size - obj_height)
            c_pos = random.randint(0, grid_size - obj_width)
            
            # Paste the object onto the grid
            for r in range(obj_height):
                for c in range(obj_width):
                    if obj[r, c] != 0:
                        grid[r_pos + r, c_pos + c] = obj[r, c]
        
        # Ensure all colors are used - fill any remaining space
        for color in colors_used:
            if color not in grid:
                # Find an empty space
                emptys = np.where(grid == 0)
                if len(emptys[0]) > 0:
                    idx = random.randint(0, len(emptys[0]) - 1)
                    r, c = emptys[0][idx], emptys[1][idx]
                    grid[r, c] = color
        
        # Fill any remaining empty spaces randomly
        empty_spaces = np.where(grid == 0)
        for i in range(len(empty_spaces[0])):
            r, c = empty_spaces[0][i], empty_spaces[1][i]
            grid[r, c] = random.choice(colors_used)
        
        # Ensure objects are 4-way connected by color
        # This step is important to make sure we have proper objects according to the problem statement
        fixed_grid = np.zeros_like(grid)
        for color in colors_used:
            color_mask = (grid == color)
            objects = find_connected_objects(color_mask.astype(int), diagonal_connectivity=False)
            for obj in objects:
                for r, c, _ in obj:
                    fixed_grid[r, c] = color
        
        # If fixed grid has empty cells, fill them
        if 0 in fixed_grid:
            random_cell_coloring(fixed_grid, colors_used, density=1.0)
        
        return fixed_grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        
        # Create an output grid of size grid_size x (2*grid_size)
        output = np.zeros((grid_size, 2 * grid_size), dtype=int)
        
        # Copy the input grid to the left half
        output[:, :grid_size] = grid
        
        # Reflect the left half horizontally to the right half
        for r in range(grid_size):
            for c in range(grid_size):
                output[r, 2*grid_size - 1 - c] = grid[r, c]
        
        return output

