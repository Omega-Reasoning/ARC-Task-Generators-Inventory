from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, List, Any, Tuple
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, GridObject, GridObjects

class Task3c9b0459Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain a completely filled grid with 8-way connected colored objects, with one object always being L-shaped in the form [[L, 0], [L, 0], [L, L]].",
            "Each input grid contains between two and four different colors, which vary across examples.",
            "The L-shaped object can vary in size and orientation, including rotations.",
            "Each input grid contains multiple colored objects with different shapes, though some objects may share the same color."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by simply rotating the input grids 180 degrees.",
        ]
        
        taskvars_definitions = {
            "grid_size": lambda: random.randint(5, 10)  # Grid size between 5 and 10
        }
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
        
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        taskvars = {"grid_size": random.randint(5, 30)}
        
        # Generate 3-5 train examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        for _ in range(num_train_examples):
            # Use between 2 and 4 colors for each training example
            # Randomly select colors (1-9) without replacement
            num_colors = random.randint(2, 4)
            colors = random.sample(range(1, 10), num_colors)
            
            # Create grid variables specific to this training example
            gridvars = {"colors": colors}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        # For test example, allow 2-4 colors to add variety
        num_colors_test = random.randint(2, 4)
        test_colors = random.sample(range(1, 10), num_colors_test)
        test_gridvars = {"colors": test_colors}

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
        size = taskvars['grid_size']
        colors = gridvars['colors']
        
        # Ensure we have between 2 and 4 distinct colors in the grid
        if len(colors) < 2:
            num_colors = random.randint(2, 4)
            colors = random.sample(range(1, 10), num_colors)
        # Trim or enforce an upper bound of 4 colors if input was larger
        if len(colors) > 4:
            colors = colors[:4]
        
        # Step 1: Create an empty grid
        grid = np.zeros((size, size), dtype=int)
        
        # Step 2: Create a distinct L-shaped object
        # Choose a color for the L-shape
        l_color = random.choice(colors)
        
        # Ensure other objects will use different colors initially
        other_colors = [c for c in colors if c != l_color]
        
        # Determine the size of the L shape (make it significant)
        # Vertical arm length should be at least 3 and at most 2/3 of grid size
        v_length = random.randint(3, max(3, int(2 * size // 3)))
        
        # Horizontal arm length should also be significant
        h_length = random.randint(3, max(3, int(2 * size // 3)))
        
        # Width of the arms (1 to 2 cells)
        arm_width = random.randint(1, min(2, size // 4))
        
        # Decide orientation of L-shape (0-3 for rotations)
        orientation = random.randint(0, 3)
        
        # Create initial L shape with specified dimensions
        # First create a template with zeros
        template_height = max(v_length, arm_width)
        template_width = max(h_length, arm_width)
        l_template = np.zeros((template_height, template_width), dtype=int)
        
        # Fill in the L shape
        # Vertical arm
        for i in range(v_length):
            for j in range(arm_width):
                l_template[i, j] = 1
                
        # Horizontal arm
        for i in range(v_length - arm_width, v_length):
            for j in range(arm_width, h_length):
                l_template[i, j] = 1
        
        # Rotate L shape according to orientation
        if orientation == 1:  # 90 degrees
            l_shape = np.rot90(l_template)
        elif orientation == 2:  # 180 degrees
            l_shape = np.rot90(l_template, 2)
        elif orientation == 3:  # 270 degrees
            l_shape = np.rot90(l_template, 3)
        else:  # 0 degrees (no rotation)
            l_shape = l_template
        
        l_height, l_width = l_shape.shape
        
        # Make sure the L shape fits in the grid
        if l_height > size or l_width > size:
            # Resize if needed (this should rarely happen with reasonable grid sizes)
            l_height = min(l_height, size)
            l_width = min(l_width, size)
            l_shape = l_shape[:l_height, :l_width]
        
        # Choose a corner to place the L shape
        corner = random.randint(0, 3)
        
        if corner == 0:  # Top-left
            r_start, c_start = 0, 0
        elif corner == 1:  # Top-right
            r_start, c_start = 0, size - l_width
        elif corner == 2:  # Bottom-left
            r_start, c_start = size - l_height, 0
        else:  # Bottom-right
            r_start, c_start = size - l_height, size - l_width
        
        # Place L shape on grid with the chosen color
        for r in range(l_height):
            for c in range(l_width):
                if l_shape[r, c] == 1:
                    grid[r + r_start, c + c_start] = l_color
        
        # Keep track of L shape positions
        l_positions = {(r + r_start, c + c_start) for r in range(l_height) for c in range(l_width) if l_shape[r, c] == 1}
        
        # Step 3: Fill the remaining space with the other colors (1..n)
        remaining_cells = [(r, c) for r in range(size) for c in range(size) if (r, c) not in l_positions]
        random.shuffle(remaining_cells)

        # Distribute remaining cells roughly evenly among the other colors
        n_other = len(other_colors)
        if n_other == 0:
            # If there are no other colors (shouldn't happen since min colors=2), fill with l_color
            for r, c in remaining_cells:
                grid[r, c] = l_color
        else:
            base = len(remaining_cells) // n_other
            remainder = len(remaining_cells) % n_other
            idx = 0
            for k in range(n_other):
                count = base + (1 if k < remainder else 0)
                for _ in range(count):
                    if idx >= len(remaining_cells):
                        break
                    r, c = remaining_cells[idx]
                    grid[r, c] = other_colors[k]
                    idx += 1
        
        # Step 4: Ensure each color is represented in at least one continuous object
        # Verify each color has a connected component
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)
        
        # Check if all specified colors are present in at least one object
        color_presence = {color: False for color in colors}
        
        for obj in objects:
            for _, _, color in obj.cells:
                color_presence[color] = True
        
        # If any color is missing, create an object with that color
        for color, present in color_presence.items():
            if not present:
                # Find a suitable location to place this color
                for r in range(size):
                    for c in range(size):
                        if grid[r, c] != l_color:  # Don't modify the L shape
                            grid[r, c] = color
                            # Just color one cell to ensure minimal representation
                            break
                    if not present:
                        break
        
        # Final verification to ensure all specified colors are used
        used_colors = set(grid.flatten())
        if len(used_colors.intersection(set(colors))) != len(colors):
            # This is a safety check - at this point all colors in 'colors' should be used
            # If not, force them to be used
            for color in colors:
                if color not in used_colors:
                    for r in range(size):
                        for c in range(size):
                            if grid[r, c] != l_color:  # Don't change the L shape
                                grid[r, c] = color
                                break
                        else:
                            continue
                        break
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # First reflect vertically (flip up-down)
        grid_vert_flipped = np.flipud(grid.copy())
        
        # Then reflect horizontally (flip left-right)
        grid_both_flipped = np.fliplr(grid_vert_flipped)
        
        return grid_both_flipped

