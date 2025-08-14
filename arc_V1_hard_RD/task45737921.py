from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import create_object, random_cell_coloring, retry
import numpy as np
import random

class Task45737921Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of different sizes.",
            "Each input grid contains many sub-grids, and those sub-grids are {vars['rows']}, {vars['columns']}in size.",
            "These sub-grids have small patterns (The patterns are usually the O‑piece (2 × 2), I‑piece (2 × 3), L‑piece (2 × 2), J‑piece (2 × 2), T‑piece (2 × 3), S‑piece (2 × 3), Z‑piece (2 × 3), plus/cross (3 × 3), and stair‑triangle (3 × 3), and can be rotated, inverted or mirrored, This example dimensions are with respect to a 3x3 sub-grid, Please change the dimensions according to the sub-grids dimensions) within them.",
            "The empty space inside the sub-grid is filled with another color.",
            "The sub-grids cannot have the same colors within them, but may or may or have the same colors, across the sub-grids."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.",
            "The sub-grids are also copied from the input grid, but the only difference here is the color transformation.",
            "Within each sub-grid, now the pattern gets the empty spacing color and the space gets the pattern color, You just need to interchange the colors within each sub-grids."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def _get_pattern_templates(self, sub_size):
        """Get pattern templates based on sub-grid size."""
        if sub_size == 2:
            patterns = [
                # L-piece variants
                [[1, 0],
                 [1, 1]],
                [[0, 1],
                 [1, 1]],
                [[1, 1],
                 [1, 0]],
                [[1, 1],
                 [0, 1]],
                # Diagonal
                [[1, 0],
                 [0, 1]],
                [[0, 1],
                 [1, 0]],
            ]
        elif sub_size == 3:
            patterns = [
                # Plus/cross
                [[0, 1, 0],
                 [1, 1, 1],
                 [0, 1, 0]],
                # L-piece variants
                [[1, 0, 0],
                 [1, 0, 0],
                 [1, 1, 1]],
                [[0, 0, 1],
                 [0, 0, 1],
                 [1, 1, 1]],
                # T-piece variants
                [[1, 1, 1],
                 [0, 1, 0],
                 [0, 0, 0]],
                [[0, 1, 0],
                 [1, 1, 1],
                 [0, 1, 0]],
                # Corner patterns
                [[1, 1, 0],
                 [1, 0, 0],
                 [0, 0, 0]],
                [[0, 1, 1],
                 [0, 0, 1],
                 [0, 0, 0]],
            ]
        else:  # 4x4
            patterns = [
                # Plus pattern
                [[0, 0, 1, 0],
                 [0, 1, 1, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 0]],
                # L-piece
                [[0, 1, 0, 0],
                 [0, 1, 0, 0],
                 [0, 1, 1, 0],
                 [0, 0, 0, 0]],
                # T-piece
                [[0, 1, 1, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],
            ]
        
        return patterns

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create input grid with scattered sub-grids containing patterns."""
        sub_rows = taskvars['rows']
        sub_cols = taskvars['columns']
        grid_size = gridvars['grid_size']
        
        # Create empty grid (background = 0)
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Calculate how many sub-grids we want (not filling the entire grid)
        # Ensure we have space for at least a few sub-grids
        theoretical_max = (grid_size // sub_rows) * (grid_size // sub_cols) // 3  # More conservative
        max_sub_grids = max(2, min(6, theoretical_max))  # Ensure at least 2, max 6
        
        # Ensure valid range for random.randint
        min_sub_grids = min(2, max_sub_grids)
        num_sub_grids = random.randint(min_sub_grids, max_sub_grids)
        
        # Get pattern templates
        patterns = self._get_pattern_templates(min(sub_rows, sub_cols))
        
        placed_positions = set()
        successful_placements = 0
        
        # Place sub-grids randomly with some spacing
        for _ in range(num_sub_grids * 3):  # Try more times than needed
            if successful_placements >= num_sub_grids:
                break
                
            attempts = 0
            while attempts < 20:  # Reduced attempts per placement
                # Random position for sub-grid
                if grid_size <= sub_rows or grid_size <= sub_cols:
                    break  # Grid too small
                    
                start_row = random.randint(0, grid_size - sub_rows)
                start_col = random.randint(0, grid_size - sub_cols)
                
                # Check if this position overlaps with existing sub-grids (with buffer)
                buffer = 1
                overlap = False
                for r in range(max(0, start_row - buffer), min(grid_size, start_row + sub_rows + buffer)):
                    for c in range(max(0, start_col - buffer), min(grid_size, start_col + sub_cols + buffer)):
                        if (r, c) in placed_positions:
                            overlap = True
                            break
                    if overlap:
                        break
                
                if not overlap:
                    # Choose random colors for this sub-grid
                    available_colors = list(range(1, 10))
                    random.shuffle(available_colors)
                    pattern_color = available_colors[0]
                    background_color = available_colors[1]
                    
                    # Choose and transform pattern
                    pattern = random.choice(patterns)
                    pattern = np.array(pattern)
                    
                    # Random transformations
                    if random.choice([True, False]):
                        pattern = np.rot90(pattern, random.randint(1, 3))
                    if random.choice([True, False]):
                        pattern = np.fliplr(pattern)
                    if random.choice([True, False]):
                        pattern = np.flipud(pattern)
                    
                    # Resize pattern if needed
                    if pattern.shape != (sub_rows, sub_cols):
                        new_pattern = np.zeros((sub_rows, sub_cols), dtype=int)
                        min_rows = min(pattern.shape[0], sub_rows)
                        min_cols = min(pattern.shape[1], sub_cols)
                        new_pattern[:min_rows, :min_cols] = pattern[:min_rows, :min_cols]
                        pattern = new_pattern
                    
                    # Place sub-grid
                    end_row = start_row + sub_rows
                    end_col = start_col + sub_cols
                    
                    # Fill background first
                    grid[start_row:end_row, start_col:end_col] = background_color
                    
                    # Apply pattern
                    for i in range(sub_rows):
                        for j in range(sub_cols):
                            if pattern[i, j] == 1:
                                grid[start_row + i, start_col + j] = pattern_color
                            # Mark position as occupied
                            placed_positions.add((start_row + i, start_col + j))
                    
                    successful_placements += 1
                    break
                
                attempts += 1
        
        # If we couldn't place enough sub-grids, place at least one
        if successful_placements == 0 and grid_size >= sub_rows and grid_size >= sub_cols:
            # Place one sub-grid in the center
            start_row = (grid_size - sub_rows) // 2
            start_col = (grid_size - sub_cols) // 2
            
            pattern = random.choice(patterns)
            pattern = np.array(pattern)
            
            # Resize if needed
            if pattern.shape != (sub_rows, sub_cols):
                new_pattern = np.zeros((sub_rows, sub_cols), dtype=int)
                min_rows = min(pattern.shape[0], sub_rows)
                min_cols = min(pattern.shape[1], sub_cols)
                new_pattern[:min_rows, :min_cols] = pattern[:min_rows, :min_cols]
                pattern = new_pattern
            
            # Random colors
            available_colors = list(range(1, 10))
            random.shuffle(available_colors)
            pattern_color = available_colors[0]
            background_color = available_colors[1]
            
            # Place the sub-grid
            grid[start_row:start_row+sub_rows, start_col:start_col+sub_cols] = background_color
            for i in range(sub_rows):
                for j in range(sub_cols):
                    if pattern[i, j] == 1:
                        grid[start_row + i, start_col + j] = pattern_color
        
        return grid

    def transform_input(self, grid: np.ndarray) -> np.ndarray:
        """Transform input by swapping colors within each sub-grid."""
        # Detect sub-grid size by analyzing the grid structure
        # This is a simplified approach - in practice, we'd need more sophisticated detection
        rows, cols = grid.shape
        
        # Try different sub-grid sizes to find the right one
        for sub_size in [2, 3, 4]:
            if self._detect_sub_grids_of_size(grid, sub_size):
                sub_rows = sub_cols = sub_size
                break
        else:
            # Default fallback
            sub_rows = sub_cols = 2
        
        # Copy input grid
        output_grid = grid.copy()
        
        # Find all sub-grids by detecting connected regions
        visited = np.zeros_like(grid, dtype=bool)
        
        for r in range(rows - sub_rows + 1):
            for c in range(cols - sub_cols + 1):
                if not visited[r, c] and grid[r, c] != 0:
                    # Check if this could be the start of a sub-grid
                    sub_region = grid[r:r+sub_rows, c:c+sub_cols]
                    
                    # Check if this region has exactly 2 colors (excluding 0)
                    unique_colors = np.unique(sub_region)
                    non_zero_colors = unique_colors[unique_colors != 0]
                    
                    if len(non_zero_colors) == 2:
                        # Check if any part of this region was already processed
                        if np.any(visited[r:r+sub_rows, c:c+sub_cols]):
                            continue
                        
                        # This looks like a sub-grid, swap its colors
                        color1, color2 = non_zero_colors[0], non_zero_colors[1]
                        
                        # Swap colors in this sub-grid
                        mask1 = sub_region == color1
                        mask2 = sub_region == color2
                        
                        output_grid[r:r+sub_rows, c:c+sub_cols][mask1] = color2
                        output_grid[r:r+sub_rows, c:c+sub_cols][mask2] = color1
                        
                        # Mark this region as visited
                        visited[r:r+sub_rows, c:c+sub_cols] = True
        
        return output_grid

    def _detect_sub_grids_of_size(self, grid: np.ndarray, sub_size: int) -> bool:
        """Helper method to detect if grid contains sub-grids of given size."""
        rows, cols = grid.shape
        found_sub_grids = 0
        
        for r in range(rows - sub_size + 1):
            for c in range(cols - sub_size + 1):
                sub_region = grid[r:r+sub_size, c:c+sub_size]
                unique_colors = np.unique(sub_region)
                non_zero_colors = unique_colors[unique_colors != 0]
                
                if len(non_zero_colors) == 2:
                    found_sub_grids += 1
                    if found_sub_grids >= 1:  # Found at least one sub-grid
                        return True
        
        return False

    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        """Create train and test grids with consistent variables."""
        # Generate sub-grid dimensions - start with smaller sizes to ensure fit
        sub_size_options = [2, 3]  # Reduced to avoid issues with larger grids
        sub_size = random.choice(sub_size_options)
        
        # Store task variables
        taskvars = {
            'rows': sub_size,
            'columns': sub_size,
        }
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Generate grid sizes - ensure they're large enough for sub-grids
        min_size = max(8, sub_size * 3)  # At least 3x the sub-grid size
        max_size = min(20, 30 // sub_size * sub_size)  # Stay within ARC limits
        all_sizes = [random.randint(min_size, max_size) for _ in range(num_train_examples + 1)]
        
        # Create training examples
        for i in range(num_train_examples):
            gridvars = {'grid_size': all_sizes[i]}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {'grid_size': all_sizes[-1]}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }

# Test code
if __name__ == "__main__":
    generator = SubGridPatternSwapTaskGenerator()
    taskvars, train_test_data = generator.create_grids()
    
    print("Task Variables:", taskvars)
    print(f"Number of train examples: {len(train_test_data['train'])}")
    print(f"Number of test examples: {len(train_test_data['test'])}")
    
    # Print grid information
    for i, example in enumerate(train_test_data['train']):
        input_shape = example['input'].shape
        unique_colors = len(np.unique(example['input']))
        print(f"Train example {i+1}: {input_shape[0]}x{input_shape[1]}, {unique_colors} unique colors")
    
    test_shape = train_test_data['test'][0]['input'].shape
    test_unique_colors = len(np.unique(train_test_data['test'][0]['input']))
    print(f"Test example: {test_shape[0]}x{test_shape[1]}, {test_unique_colors} unique colors")
    
    # Visualize if the visualization method is available
    try:
        ARCTaskGenerator.visualize_train_test_data(train_test_data)
    except:
        print("Visualization not available, but grids created successfully!")