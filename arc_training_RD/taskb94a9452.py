from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import retry
from transformation_library import find_connected_objects, GridObject

class Taskb94a9452Generator(ARCTaskGenerator):
    def __init__(self):
        # Initialize reasoning chains
        input_reasoning_chain = [
            "Input grids are almost square shaped and of size {vars['grid_height']} x {vars['grid_width']}.", 
            "The grid consists of exactly one square block.", 
            "The square block is usually greater than or equal to 3x3.",
            "These blocks form two parts: an outer boundary region and inner block region always a square.",
            "Each input grid uses different color combinations for the boundary and inner regions."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is only the size of the square block from the input grid.",
            "The boundary region and inner block regions are copied from the input grid, but there is a color swap.",
            "The boundary region gets the color that was originally the inner block color.",
            "The inner block region gets the color that was originally the boundary color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create an almost square grid with a square block containing boundary and inner regions."""
        # Extract colors and dimensions from taskvars
        bound_color = taskvars["bound_color"]
        block_color = taskvars["block_color"]
        grid_height = taskvars["grid_height"]
        grid_width = taskvars["grid_width"]
        
        # Randomly determine block size (at least 3x3 but smaller than grid)
        block_size = random.randint(3, min(grid_height-2, grid_width-2, 10))
        
        # Create empty grid
        grid = np.zeros((grid_height, grid_width), dtype=int)
        
        # Randomly place the block within the grid
        max_row = grid_height - block_size
        max_col = grid_width - block_size
        start_row = random.randint(0, max_row)
        start_col = random.randint(0, max_col)
        
        # Fill the entire block with the boundary color
        for r in range(start_row, start_row + block_size):
            for c in range(start_col, start_col + block_size):
                grid[r, c] = bound_color
        
        # Calculate inner square dimensions and position
        inner_size = block_size - 2  # Inner square is boundary-2
        inner_start_row = start_row + 1
        inner_start_col = start_col + 1
        
        # Fill the inner square with the block color (only if inner_size > 0)
        if inner_size > 0:
            for r in range(inner_start_row, inner_start_row + inner_size):
                for c in range(inner_start_col, inner_start_col + inner_size):
                    grid[r, c] = block_color
        
        return grid
    
    def transform_input(self, grid, taskvars):
        """Transform input by extracting the block and swapping boundary/inner colors."""
        # Find the block in the input grid (connected non-zero region)
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=False)
        
        if len(objects) != 1:
            raise ValueError(f"Expected exactly one object, found {len(objects)}")
        
        block_object = objects[0]
        
        # Extract the block bounds
        block_rows, block_cols = block_object.bounding_box
        block_height = block_rows.stop - block_rows.start
        block_width = block_cols.stop - block_cols.start
        
        # Extract the block region from input
        block_region = grid[block_rows.start:block_rows.stop, block_cols.start:block_cols.stop]
        
        # Detect the colors dynamically from the block
        unique_colors = np.unique(block_region[block_region != 0])
        
        if len(unique_colors) == 1:
            # Only one color (boundary only, no inner region)
            bound_color = unique_colors[0]
            block_color = 0  # No inner color
        elif len(unique_colors) == 2:
            # Two colors - determine which is boundary and which is inner
            # The boundary color appears on the edges, inner color appears in center
            edge_colors = set()
            # Check top and bottom edges
            edge_colors.update(block_region[0, :][block_region[0, :] != 0])
            edge_colors.update(block_region[-1, :][block_region[-1, :] != 0])
            # Check left and right edges  
            edge_colors.update(block_region[:, 0][block_region[:, 0] != 0])
            edge_colors.update(block_region[:, -1][block_region[:, -1] != 0])
            
            # Boundary color appears on edges
            bound_color = list(edge_colors)[0] if len(edge_colors) == 1 else unique_colors[0]
            # Inner color is the other one
            block_color = unique_colors[1] if bound_color == unique_colors[0] else unique_colors[0]
        else:
            # Fallback - shouldn't happen with our generation logic
            bound_color = unique_colors[0]
            block_color = unique_colors[1] if len(unique_colors) > 1 else 0
        
        # Create output grid of the same size as the block
        output_grid = np.zeros((block_height, block_width), dtype=int)
        
        # Swap colors in the extracted block
        for r in range(block_height):
            for c in range(block_width):
                if block_region[r, c] == bound_color:
                    output_grid[r, c] = block_color
                elif block_region[r, c] == block_color:
                    output_grid[r, c] = bound_color
        
        return output_grid
    
    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        # Generate almost square dimensions (like 12x13, 13x12, 11x12, etc.)
        base_size = random.randint(11, 14)  # Base size between 11-14
        
        # Create "almost square" by making width and height differ by 0, 1, or 2
        height_offset = random.choice([-2, -1, 0, 1, 2])
        width_offset = random.choice([-2, -1, 0, 1, 2])
        
        grid_height = base_size + height_offset
        grid_width = base_size + width_offset
        
        # Ensure minimum size
        grid_height = max(grid_height, 8)
        grid_width = max(grid_width, 8)
        
        # Set up general task variables (only grid dimensions, no colors)
        general_taskvars = {
            "grid_height": grid_height,
            "grid_width": grid_width,
        }
        
        # Generate 3-5 training examples with different colors for each
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Keep track of used color combinations to ensure variety
        used_color_combinations = set()
        available_colors = list(range(1, 10))
        
        for i in range(num_train_examples):
            # Select distinct random colors for bound and block
            attempts = 0
            while attempts < 50:  # Prevent infinite loop
                colors = random.sample(available_colors, 2)
                bound_color, block_color = colors
                color_combo = tuple(sorted([bound_color, block_color]))
                
                if color_combo not in used_color_combinations:
                    used_color_combinations.add(color_combo)
                    break
                attempts += 1
            else:
                # If we can't find unused combination, just use random colors
                colors = random.sample(available_colors, 2)
                bound_color, block_color = colors
            
            # Create taskvars for this specific example
            example_taskvars = general_taskvars.copy()
            example_taskvars.update({
                "bound_color": bound_color,
                "block_color": block_color
            })
            
            gridvars = {}
            input_grid = self.create_input(example_taskvars, gridvars)
            output_grid = self.transform_input(input_grid, general_taskvars)  # Use general_taskvars for transform
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example with different colors
        attempts = 0
        while attempts < 50:
            colors = random.sample(available_colors, 2)
            test_bound_color, test_block_color = colors
            test_color_combo = tuple(sorted([test_bound_color, test_block_color]))
            
            if test_color_combo not in used_color_combinations:
                break
            attempts += 1
        else:
            # If we can't find unused combination, just use random colors
            colors = random.sample(available_colors, 2)
            test_bound_color, test_block_color = colors
        
        test_taskvars = general_taskvars.copy()
        test_taskvars.update({
            "bound_color": test_bound_color,
            "block_color": test_block_color
        })
        
        test_gridvars = {}
        test_input = self.create_input(test_taskvars, test_gridvars)
        test_output = self.transform_input(test_input, general_taskvars)  # Use general_taskvars for transform
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
             
        return general_taskvars, {
            'train': train_examples,
            'test': test_examples
        }

# Test code
if __name__ == "__main__":
    generator = Taskb94a9452Generator()
    taskvars, train_test_data = generator.create_grids()
    
    print("Task Variables:", taskvars)
    ARCTaskGenerator.visualize_train_test_data(train_test_data)