from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import retry
from Framework.transformation_library import find_connected_objects

class Task05269061Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain three completely filled single-colored diagonal lines, along with empty (0) cells.",
            "Each line is colored differently, and the colors vary across grids.",
            "The diagonal lines are parallel to the inverse diagonal direction (top-right to bottom-left).",
            "To construct the input grid, choose three different colors and fill the grid completely with diagonal lines from top-right to bottom-left.",
            "Arrange the three chosen colors so that each color repeats after every three diagonal lines from top-right to bottom-left.",
            "Once the grid is fully filled with diagonal lines, remove {2*vars['grid_size']-4} of them, ensuring that the remaining three diagonal lines are each a different color.",
            "All other cells must remain empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and identifying the three differently colored diagonal lines.",
            "Once the colors are identified, completely fill the output grid with single-colored diagonal lines in the same direction.",
            "The lines should be positioned such that their colors repeat every three diagonal lines.",
            "Ensure that the lines from the input grid are preserved exactly, while new diagonal lines are added and positioned according to the above rule."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        """Create train and test grids according to the specifications."""
        # Initialize random grid size between 5 and 15
        grid_size = random.randint(5, 15)
        taskvars = {'grid_size': grid_size}
        
        # Create train examples
        train_examples = []
        
        # Two examples with consecutive diagonal lines
        for _ in range(2):
            colors = self.get_random_colors(3)
            gridvars = {'colors': colors, 'consecutive': True}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # One example with two consecutive lines and one separated
        colors = self.get_random_colors(3)
        gridvars = {'colors': colors, 'consecutive': False, 'partially_consecutive': True}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Test example with all lines separated (no consecutive diagonals)
        colors = self.get_random_colors(3)
        gridvars = {'colors': colors, 'consecutive': False, 'partially_consecutive': False, 'ensure_no_consecutive': True}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        test_examples = [{'input': input_grid, 'output': output_grid}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def get_random_colors(self, count):
        """Get a list of random unique colors (1-9)."""
        return random.sample(range(1, 10), count)
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        """Create an input grid with three diagonal lines according to the specified pattern."""
        grid_size = taskvars['grid_size']
        colors = gridvars['colors']
        consecutive = gridvars.get('consecutive', False)
        partially_consecutive = gridvars.get('partially_consecutive', False)
        ensure_no_consecutive = gridvars.get('ensure_no_consecutive', False)
        
        # Step 1: Initialize the grid and fill it completely with the pattern of diagonal lines
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Create all possible diagonal positions (top-right to bottom-left)
        all_diagonals = []
        for offset in range(2*grid_size - 1):
            diagonal = []
            for i in range(grid_size):
                j = grid_size - 1 - offset + i
                if 0 <= j < grid_size:
                    diagonal.append((i, j))
            if diagonal:
                all_diagonals.append(diagonal)
        
        # Fill all diagonals with repeating color pattern
        for i, diagonal in enumerate(all_diagonals):
            color_index = i % 3
            color = colors[color_index]
            for r, c in diagonal:
                grid[r, c] = color
        
        # Step 2: Choose which 3 diagonals to keep based on constraints, remove all others
        temp_grid = np.copy(grid)  # Keep a copy of the fully filled grid
        grid = np.zeros((grid_size, grid_size), dtype=int)  # Reset to empty
        
        # Choose diagonals to keep according to the pattern constraints
        if consecutive:
            # Choose a random starting position for three consecutive diagonals
            start_pos = random.randint(0, len(all_diagonals) - 3)
            keep_diagonals = all_diagonals[start_pos:start_pos + 3]
        elif partially_consecutive:
            # Two consecutive, one separated
            start_pos = random.randint(0, len(all_diagonals) - 2)
            keep_diagonals = all_diagonals[start_pos:start_pos + 2]
            
            # Find a non-consecutive position
            remaining_positions = list(range(0, start_pos)) + list(range(start_pos + 2, len(all_diagonals)))
            if remaining_positions:
                third_pos = random.choice(remaining_positions)
                keep_diagonals.append(all_diagonals[third_pos])
            else:
                # Fallback if no non-consecutive positions are available
                third_pos = start_pos + 2 if start_pos + 2 < len(all_diagonals) else 0
                keep_diagonals.append(all_diagonals[third_pos])
        elif ensure_no_consecutive:
            # For test case: Ensure NO consecutive diagonals at all
            total_diagonals = len(all_diagonals)
            
            # Ensure we have enough diagonals to place 3 with no consecutives
            if total_diagonals < 5:
                # For very small grids, we might not be able to avoid consecutive diagonals
                # Choose positions as far apart as possible
                positions = [0, total_diagonals//2, total_diagonals-1]
            else:
                # Try to find three positions with no consecutive diagonals
                # We'll keep randomly selecting until we find a valid arrangement
                while True:
                    positions = sorted(random.sample(range(total_diagonals), 3))
                    # Check if any two positions are consecutive
                    consecutive_found = False
                    for i in range(len(positions) - 1):
                        if positions[i+1] == positions[i] + 1:
                            consecutive_found = True
                            break
                    if not consecutive_found:
                        break
            
            keep_diagonals = [all_diagonals[pos] for pos in positions]
        else:
            # All three diagonals separated (but might be consecutive)
            positions = sorted(random.sample(range(len(all_diagonals)), 3))
            keep_diagonals = [all_diagonals[pos] for pos in positions]
        
        # Copy only the selected diagonals from the original filled grid
        for diagonal in keep_diagonals:
            for r, c in diagonal:
                grid[r, c] = temp_grid[r, c]
        
        # Verify that we have exactly 3 different colors in the final grid
        unique_colors = np.unique(grid[grid > 0])
        if len(unique_colors) != 3:
            # If we don't have 3 colors, retry with different positions
            return self.create_input(taskvars, gridvars)
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        """Transform the input grid by completing the diagonal pattern, 
        preserving existing colored cells exactly."""
        grid_size = grid.shape[0]
        output_grid = np.copy(grid)
        
        # Find the colors of the existing diagonal lines
        unique_colors = np.unique(grid[grid > 0])
        if len(unique_colors) != 3:
            raise ValueError(f"Expected 3 different colors, found {len(unique_colors)}")
        
        # Identify all diagonal positions in the grid
        all_diagonals = []
        for offset in range(2*grid_size - 1):
            diagonal = []
            for i in range(grid_size):
                j = grid_size - 1 - offset + i
                if 0 <= j < grid_size:
                    diagonal.append((i, j))
            if diagonal:
                all_diagonals.append(diagonal)
        
        # Find the position and color of each non-empty diagonal
        color_mapping = {}
        for i, diagonal in enumerate(all_diagonals):
            # Check if this diagonal has any colored cells
            diagonal_colors = set()
            for r, c in diagonal:
                if grid[r, c] > 0:
                    diagonal_colors.add(grid[r, c])
            
            # If we found a colored diagonal, record its color
            if len(diagonal_colors) == 1:  # Should only be one color per diagonal
                color_mapping[i] = next(iter(diagonal_colors))
        
        # Determine the pattern positions (0, 1, 2) for each color
        color_pattern = [None, None, None]
        for pos, color in color_mapping.items():
            pattern_pos = pos % 3
            color_pattern[pattern_pos] = color
        
        # If any pattern position is still None, we need to infer it
        missing_positions = [i for i, color in enumerate(color_pattern) if color is None]
        missing_colors = [color for color in unique_colors if color not in color_pattern]
        
        # Assign missing colors to missing positions
        for i, pos in enumerate(missing_positions):
            if i < len(missing_colors):
                color_pattern[pos] = missing_colors[i]
        
        # Fill the output grid according to the pattern, preserving existing cells
        for i, diagonal in enumerate(all_diagonals):
            pattern_pos = i % 3
            color = color_pattern[pattern_pos]
            
            # Only fill empty cells in this diagonal
            for r, c in diagonal:
                if grid[r, c] == 0:  # Only fill cells that are empty in the input
                    output_grid[r, c] = color
        
        return output_grid


