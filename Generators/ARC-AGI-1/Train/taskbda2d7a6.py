from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Taskbda2d7a6Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square grids of size {vars['grid_height']}x{vars['grid_width']}.",
            "The grid consists of concentric regions with different structures:",
            "Some regions are thick borders, some are thin rings",
            "Some regions may be filled with colors, some may be empty (black)",
            "The structure can vary - borders do not have to be 1 pixel thick",
            "Each input grid uses different color combinations."
        ]
        
        transformation_reasoning_chain = [
            "The output grid recreates the EXACT same structure as input.",
            "Only colors rotate through all regions.",
            "Each region gets the color from the region immediately inside it",
            "The innermost region color goes to the outermost region",
            "Empty regions (black) also participate in this rotation",
            "The structure pattern stays completely identical."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create a grid with concentric regions of different colors."""
        size = taskvars["grid_height"]  # Use size from taskvars
        colors = [taskvars["color1"], taskvars["color2"], taskvars["color3"]]
        
        grid = np.zeros((size, size), dtype=int)
        center = size // 2
        
        # Create the exact pattern from examples:
        # Outer border -> middle area -> center block
        
        # Fill outer border (can be thick)
        border_thickness = 1  # Keep it simple like examples
        for i in range(border_thickness):
            grid[i, :] = colors[0]
            grid[-(i+1), :] = colors[0]
            grid[:, i] = colors[0]
            grid[:, -(i+1)] = colors[0]
        
        # Fill middle area (everything between border and center)
        middle_start = border_thickness
        middle_end = size - border_thickness
        
        # Sometimes fill middle with a color, sometimes leave empty
        if random.choice([True, False]):
            grid[middle_start:middle_end, middle_start:middle_end] = colors[1]
        # If False, middle stays 0 (empty)
        
        # Fill center block (3x3)
        center_start = center - 1
        center_end = center + 2
        grid[center_start:center_end, center_start:center_end] = colors[2]
        
        return grid

    def transform_input(self, grid, taskvars):
        """Transform input by rotating colors through regions."""
        size = grid.shape[0]
        center = size // 2
        
        # Get colors from the three regions
        outer_color = grid[0, 0]  # Outer border
        center_color = grid[center, center]  # Center
        
        # Get middle color (sample from middle area)
        middle_color = grid[1, 1]  # Middle area
        
        # Create output grid by copying input structure
        output_grid = grid.copy()
        
        # Apply color rotation: center->outer, outer->middle, middle->center
        
        # Replace all instances of each color with the rotated color
        output_grid[grid == outer_color] = center_color  # outer becomes center color
        output_grid[grid == middle_color] = outer_color  # middle becomes outer color  
        output_grid[grid == center_color] = middle_color  # center becomes middle color
        
        return output_grid

    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        # Choose consistent grid size for all examples
        size = random.choice([7, 9])
        
        # Set up general task variables that apply to all examples
        general_taskvars = {
            "grid_height": size,
            "grid_width": size,
        }
        
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        used_color_combinations = set()
        available_colors = list(range(1, 10))
        
        # Generate training examples with different color combinations
        for i in range(num_train_examples):
            attempts = 0
            while attempts < 50:
                colors = random.sample(available_colors, 3)
                color1, color2, color3 = colors
                color_combo = tuple(sorted([color1, color2, color3]))
                
                if color_combo not in used_color_combinations:
                    used_color_combinations.add(color_combo)
                    break
                attempts += 1
            else:
                colors = random.sample(available_colors, 3)
                color1, color2, color3 = colors
            
            # Combine general task vars with example-specific colors
            taskvars = general_taskvars.copy()
            taskvars.update({
                "color1": color1,
                "color2": color2,
                "color3": color3
            })
            
            gridvars = {}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example with unique color combination
        attempts = 0
        while attempts < 50:
            colors = random.sample(available_colors, 3)
            test_color1, test_color2, test_color3 = colors
            test_color_combo = tuple(sorted([test_color1, test_color2, test_color3]))
            
            if test_color_combo not in used_color_combinations:
                break
            attempts += 1
        else:
            colors = random.sample(available_colors, 3)
            test_color1, test_color2, test_color3 = colors
        
        # Combine general task vars with test-specific colors
        test_taskvars = general_taskvars.copy()
        test_taskvars.update({
            "color1": test_color1,
            "color2": test_color2, 
            "color3": test_color3
        })
        
        test_gridvars = {}
        test_input = self.create_input(test_taskvars, test_gridvars)
        test_output = self.transform_input(test_input, test_taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return general_taskvars, {
            'train': train_examples,
            'test': test_examples
        }

