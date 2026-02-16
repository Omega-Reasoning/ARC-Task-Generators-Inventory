from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import create_object, Contiguity, retry
from Framework.transformation_library import find_connected_objects, BorderBehavior

class Taska5f85a15Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are squares and can have different sizes.",
            "They contain one or more same-colored diagonal lines, with the remaining cells being empty (0).",
            "The diagonal lines are in the direction which is parallel to the main diagonal (top-left to bottom-right).",
            "The diagonal lines may not necessarily be on the main diagonal but parallel to it.",
            "Incase of more than one diagonal line, all diagonal lines must be completely separated from each other by having at least one empty (0) diagonal line between two consecutive colored diagonal lines.",
            "If there are more than one diagonal lines, the color of each line should be the same within the grid but must vary across grids."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying the colored diagonal lines.",
            "Once identified, for each diagonal line, start from the second cell and change its color to {color('fill_color')}.", 
            "Repeat this process by changing the color of all alternating cells along the diagonal line, continuing until no more cells can be recolored."
        ]
        
        # Initialize task variables
        taskvars_definitions = {"fill_color": random.randint(1, 9)}
        
        # Call parent constructor with correct arguments
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Get the fill color from task variables (already defined in __init__)
        fill_color = random.randint(1, 9)
        
        # Ensure we have different grid sizes and line colors
        grid_sizes = random.sample(range(15, 31), 4)  # For 3-5 training grids + 1 test grid
        line_colors = []
        
        # Ensure line colors are different from fill_color
        for _ in range(4):
            color = random.randint(1, 9)
            while color == fill_color or color in line_colors:
                color = random.randint(1, 9)
            line_colors.append(color)
        
        # Make a map of grid configurations we need to ensure
        grid_configs = [
            {"size": grid_sizes[0], "line_color": line_colors[0], "has_main_diag": True, "num_diags": 1},
            {"size": grid_sizes[1], "line_color": line_colors[1], "has_main_diag": False, "num_diags": 2},
            {"size": grid_sizes[2], "line_color": line_colors[2], "has_main_diag": False, "num_diags": random.randint(1, 3)},
        ]
        
        # Add one more training grid if we want 4 training examples
        if random.choice([True, False]):
            grid_configs.append({"size": grid_sizes[3], "line_color": line_colors[3], "has_main_diag": random.choice([True, False]), "num_diags": random.randint(1, 3)})
        
        # Test grid configuration
        # Ensure test num_diags is strictly different from all training grids
        # and the test line color is different from all training grid colors.
        training_num_diags = {cfg["num_diags"] for cfg in grid_configs}

        # pick a num_diags for test that is not used in training; prefer 1..3
        possible_num_diags = [n for n in range(1, 4) if n not in training_num_diags]
        if possible_num_diags:
            test_num_diags = random.choice(possible_num_diags)
        else:
            # if all 1..3 are present in training (rare), use 4 to keep it distinct
            test_num_diags = 4

        # Determine colors actually used in training (grid_configs may be shorter than line_colors)
        used_training_colors = {cfg["line_color"] for cfg in grid_configs}

        # pick a test color not used in training and not equal to fill_color
        available_colors = [c for c in range(1, 10) if c not in used_training_colors and c != fill_color]
        if not available_colors:
            # As a safe fallback (extremely unlikely given color range), choose any color not equal to fill_color
            available_colors = [c for c in range(1, 10) if c != fill_color]

        test_color = random.choice(available_colors)

        test_config = {
            "size": random.randint(15, 30),
            "line_color": test_color,
            "has_main_diag": random.choice([True, False]),
            "num_diags": test_num_diags,
        }

        # Generate train grids
        train_pairs = []
        for config in grid_configs:
            gridvars = {
                "size": config["size"],
                "line_color": config["line_color"],
                "has_main_diag": config["has_main_diag"],
                "num_diags": config["num_diags"]
            }
            
            input_grid = self.create_input({"fill_color": fill_color}, gridvars)
            output_grid = self.transform_input(input_grid, {"fill_color": fill_color})
            train_pairs.append({"input": input_grid, "output": output_grid})
        
        # Generate test grid
        test_gridvars = {
            "size": test_config["size"],
            "line_color": test_config["line_color"],
            "has_main_diag": test_config["has_main_diag"],
            "num_diags": test_config["num_diags"]
        }
        
        test_input = self.create_input({"fill_color": fill_color}, test_gridvars)
        test_output = self.transform_input(test_input, {"fill_color": fill_color})
        
        return {"fill_color": fill_color}, {
            "train": train_pairs,
            "test": [{"input": test_input, "output": test_output}]
        }
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        # Extract variables
        grid_size = gridvars["size"]
        line_color = gridvars["line_color"]
        has_main_diag = gridvars["has_main_diag"]
        num_diags = gridvars["num_diags"]
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Find valid diagonal offsets that would give at least 3 cells on each diagonal
        # An offset of k means cells at positions (i, i+k) or (i-k, i)
        # Length of diagonal with offset k is (grid_size - abs(k))
        valid_offsets = [k for k in range(-grid_size+3, grid_size-2) if abs(k) <= grid_size-3]
        
        # If we need the main diagonal, make sure 0 is in the selected offsets
        if has_main_diag:
            offsets = [0]
            valid_offsets.remove(0)
            num_diags -= 1
        else:
            offsets = []
        
        # Ensure diagonal lines are separated by at least one empty diagonal
        if num_diags > 0:
            # If we already have the main diagonal, exclude adjacent diagonals
            if has_main_diag:
                if 1 in valid_offsets:
                    valid_offsets.remove(1)
                if -1 in valid_offsets:
                    valid_offsets.remove(-1)
            
            # Select remaining offsets ensuring separation
            while num_diags > 0 and valid_offsets:
                offset = random.choice(valid_offsets)
                offsets.append(offset)
                valid_offsets.remove(offset)
                
                # Remove adjacent diagonals to ensure separation
                if offset+1 in valid_offsets:
                    valid_offsets.remove(offset+1)
                if offset-1 in valid_offsets:
                    valid_offsets.remove(offset-1)
                
                num_diags -= 1
        
        # Draw the diagonal lines
        for offset in offsets:
            if offset >= 0:
                # Diagonals below or on the main diagonal
                for i in range(grid_size - offset):
                    grid[i, i + offset] = line_color
            else:
                # Diagonals above the main diagonal
                for i in range(grid_size + offset):
                    grid[i - offset, i] = line_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        # Create a copy of the input grid
        output = grid.copy()
        fill_color = taskvars["fill_color"]
        
        # Find all diagonal lines
        # We'll use diagonal detection by iterating through all possible diagonals
        grid_size = grid.shape[0]
        
        # Process each diagonal (both main and parallel)
        for offset in range(-grid_size+1, grid_size):
            if offset >= 0:
                # Diagonals below or on the main diagonal (from top-left to bottom-right)
                cells = [(i, i+offset) for i in range(grid_size-offset)]
            else:
                # Diagonals above the main diagonal (from top-left to bottom-right)
                cells = [(i-offset, i) for i in range(grid_size+offset)]
            
            # Filter valid cells and check if this is a colored diagonal
            valid_cells = [(r, c) for r, c in cells if 0 <= r < grid_size and 0 <= c < grid_size]
            diagonal_values = [grid[r, c] for r, c in valid_cells]
            
            # Skip empty diagonals or diagonals with mixed colors
            if not diagonal_values or all(val == 0 for val in diagonal_values):
                continue
                
            # Identify non-zero cells and check if they all have the same color
            non_zero_cells = [(r, c) for r, c in valid_cells if grid[r, c] != 0]
            non_zero_colors = {grid[r, c] for r, c in non_zero_cells}
            
            if len(non_zero_colors) > 1:
                continue
                
            # Ensure the diagonal has at least 3 cells before applying transformation
            if len(non_zero_cells) >= 3:
                # Apply the alternating color pattern: change every second cell to fill_color
                for i in range(1, len(non_zero_cells), 2):
                    r, c = non_zero_cells[i]
                    output[r, c] = fill_color
                
        return output

