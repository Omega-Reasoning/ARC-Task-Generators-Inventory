from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Task85c4e7cdGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square grids of size {vars['grid_height']}x{vars['grid_width']}.",
            "The grid consists of concentric regions with different structures:",
            "Each boundary/ring is exactly one cell thick",
            "All regions are filled with colors, no empty (black) regions",
            "Each concentric layer has a different color",
            "Each input grid uses different color combinations."
        ]
        
        transformation_reasoning_chain = [
            "The output grid recreates the EXACT same structure as input.",
            "Only colors rotate through all regions.",
            "Each region gets the color from the region immediately inside it",
            "The innermost region color goes to the outermost region",
            "All filled regions participate in this rotation",
            "The structure pattern stays completely identical."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create a grid with concentric regions of different colors."""
        size = taskvars["grid_height"]
        
        grid = np.zeros((size, size), dtype=int)
        center = size // 2
        
        # Calculate how many concentric layers we can fit
        max_layers = center + 1
        
        # Get the colors for this example
        colors = []
        for i in range(max_layers):
            color_key = f"color{i+1}"
            if color_key in taskvars:
                colors.append(taskvars[color_key])
        
        # Create concentric layers from outside to inside
        for layer in range(max_layers):
            if layer < len(colors):
                # Calculate the bounds for this layer
                start = layer
                end = size - layer
                
                # Only fill the border of this layer (not the interior)
                if start < end:
                    # Fill top and bottom rows of this layer
                    if start < size and end > 0:
                        grid[start, start:end] = colors[layer]
                        if end - 1 != start:  # Don't duplicate if it's the same row
                            grid[end-1, start:end] = colors[layer]
                        
                        # Fill left and right columns of this layer
                        grid[start:end, start] = colors[layer]
                        if end - 1 != start:  # Don't duplicate if it's the same column
                            grid[start:end, end-1] = colors[layer]
        
        # Fill the center pixel if grid size is odd
        if size % 2 == 1 and len(colors) > 0:
            grid[center, center] = colors[-1]
        
        return grid

    def extract_layer_colors_from_grid(self, grid):
        """Extract the colors of each concentric layer from the grid."""
        size = grid.shape[0]
        center = size // 2
        max_layers = center + 1
        
        layer_colors = []
        
        for layer in range(max_layers):
            start = layer
            end = size - layer
            
            if start < end:
                # Get color from the top-left corner of this layer
                if start < size:
                    color = grid[start, start]
                    if color != 0:  # Skip black/empty cells
                        layer_colors.append(color)
                    else:
                        break  # No more layers
                else:
                    break
            else:
                break
        
        return layer_colors

    def transform_input(self, grid, taskvars):
        """Transform input by rotating colors through regions."""
        # Extract colors from the grid itself
        layer_colors = self.extract_layer_colors_from_grid(grid)
        
        if len(layer_colors) < 2:
            return grid.copy()
        
        # Create output grid by copying input structure
        output_grid = grid.copy()
        
        # Apply color rotation: each layer gets the color from the layer inside it
        # The innermost layer gets the color from the outermost layer
        for i in range(len(layer_colors)):
            current_color = layer_colors[i]
            # Get the color from the next inner layer (or outermost if we're at innermost)
            next_color = layer_colors[(i + 1) % len(layer_colors)]
            output_grid[grid == current_color] = next_color
        
        return output_grid

    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        # Choose smaller grid sizes to accommodate color limitations
        size = random.choice([5, 7, 9])
        max_layers = (size // 2) + 1
        
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
                # Select colors for all layers
                colors = random.sample(available_colors, min(max_layers, len(available_colors)))
                color_combo = tuple(sorted(colors))
                
                if color_combo not in used_color_combinations:
                    used_color_combinations.add(color_combo)
                    break
                attempts += 1
            else:
                colors = random.sample(available_colors, min(max_layers, len(available_colors)))
            
            # Combine general task vars with example-specific colors
            taskvars = general_taskvars.copy()
            for j, color in enumerate(colors):
                taskvars[f"color{j+1}"] = color
            
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
            test_colors = random.sample(available_colors, min(max_layers, len(available_colors)))
            test_color_combo = tuple(sorted(test_colors))
            
            if test_color_combo not in used_color_combinations:
                break
            attempts += 1
        else:
            test_colors = random.sample(available_colors, min(max_layers, len(available_colors)))
        
        # Combine general task vars with test-specific colors
        test_taskvars = general_taskvars.copy()
        for j, color in enumerate(test_colors):
            test_taskvars[f"color{j+1}"] = color
        
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

