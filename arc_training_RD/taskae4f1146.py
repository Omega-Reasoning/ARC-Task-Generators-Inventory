from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Taskae4f1146Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are 9x9 fixed size.",
            "The grid consists of exactly 4 sub-grids of 3x3 spread across the main grid with proper spacing between them, these sub-grids may or may not have patterns within them. These patterns can be just single cells or scattered cells, or an actual pattern.",
            "These sub-grids consist of two colors namely {color('base_color')} color which contributes to the base color of the sub-grid and {color('pattern_color')} color which contributes to the formation of patterns within the sub-grid."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is 3x3 fixed size.",
            "The output grid is formed by identifying that one sub-grid from the input grid which has the maximum patterns."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create a 9x9 input grid with exactly 4 sub-grids (3x3 each) placed with proper spacing."""
        # Create a 9x9 grid filled with zeros
        grid = np.zeros((9, 9), dtype=int)
        
        # Get colors from taskvars
        base_color = taskvars['base_color']
        pattern_color = taskvars['pattern_color']
        
        # Define valid positions for 3x3 sub-grids with spacing
        # Positions that ensure at least 1 cell spacing between sub-grids
        valid_positions = [
            (0, 0),   # top-left corner
            (0, 6),   # top-right corner  
            (6, 0),   # bottom-left corner
            (6, 6),   # bottom-right corner
            (0, 3),   # top-center
            (3, 0),   # middle-left
            (3, 6),   # middle-right
            (6, 3),   # bottom-center
        ]
        
        # Function to check if two positions have proper spacing
        def has_proper_spacing(pos1, pos2):
            r1, c1 = pos1
            r2, c2 = pos2
            # Check if sub-grids don't overlap and have at least some spacing
            # Two 3x3 grids need at least 1 cell gap between them
            return (abs(r1 - r2) >= 4 or abs(c1 - c2) >= 4 or 
                    (abs(r1 - r2) >= 3 and abs(c1 - c2) >= 3))
        
        # Select exactly 4 positions with proper spacing
        selected_positions = []
        max_attempts = 100
        
        for attempt in range(max_attempts):
            # Try to select 4 positions
            candidate_positions = random.sample(valid_positions, min(4, len(valid_positions)))
            
            # Check if all pairs have proper spacing
            valid_selection = True
            for i in range(len(candidate_positions)):
                for j in range(i + 1, len(candidate_positions)):
                    if not has_proper_spacing(candidate_positions[i], candidate_positions[j]):
                        valid_selection = False
                        break
                if not valid_selection:
                    break
            
            if valid_selection:
                selected_positions = candidate_positions
                break
        
        # If we couldn't find 4 positions with proper spacing, use the 4 corners
        if len(selected_positions) != 4:
            selected_positions = [(0, 0), (0, 6), (6, 0), (6, 6)]
        
        # Create sub-grids at selected positions
        for start_i, start_j in selected_positions:
            # Fill the 3x3 subgrid with base_color first
            for i in range(3):
                for j in range(3):
                    grid[start_i + i, start_j + j] = base_color
            
            # Randomly decide density of pattern cells in this sub-grid
            pattern_density = random.uniform(0.1, 0.8)
            
            # Apply patterns to this 3x3 sub-grid
            for i in range(3):
                for j in range(3):
                    if random.random() < pattern_density:
                        grid[start_i + i, start_j + j] = pattern_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """Extract the 3x3 sub-grid with the maximum number of pattern cells."""
        base_color = taskvars['base_color']
        pattern_color = taskvars['pattern_color']
        
        # All possible 3x3 starting positions in a 9x9 grid
        possible_positions = []
        for r in range(7):  # 0 to 6
            for c in range(7):  # 0 to 6
                possible_positions.append((r, c))
        
        # Find all valid sub-grids (those that contain our colors)
        valid_subgrids = []
        
        for start_r, start_c in possible_positions:
            subgrid = grid[start_r:start_r+3, start_c:start_c+3]
            
            # Check if this subgrid contains our colors and is likely a placed sub-grid
            has_base = np.any(subgrid == base_color)
            has_zeros = np.any(subgrid == 0)
            
            # A valid sub-grid should have base color and minimal zeros
            # (since we fill the entire 3x3 area with base color first)
            if has_base and np.sum(subgrid == 0) <= 1:  # Allow minimal zeros due to edge effects
                pattern_count = np.sum(subgrid == pattern_color)
                valid_subgrids.append({
                    'position': (start_r, start_c),
                    'pattern_count': pattern_count,
                    'subgrid': subgrid.copy()
                })
        
        # Remove overlapping detections - keep only non-overlapping sub-grids
        filtered_subgrids = []
        for current in valid_subgrids:
            curr_r, curr_c = current['position']
            is_unique = True
            
            for existing in filtered_subgrids:
                exist_r, exist_c = existing['position']
                # Check if they overlap significantly
                if abs(curr_r - exist_r) < 3 and abs(curr_c - exist_c) < 3:
                    # They overlap, keep the one with more patterns
                    if current['pattern_count'] > existing['pattern_count']:
                        filtered_subgrids.remove(existing)
                    else:
                        is_unique = False
                    break
            
            if is_unique:
                filtered_subgrids.append(current)
        
        # Find the sub-grid with maximum patterns
        if filtered_subgrids:
            max_subgrid_info = max(filtered_subgrids, key=lambda x: x['pattern_count'])
            return max_subgrid_info['subgrid']
        
        # Fallback: if no valid sub-grid found, create a default one
        default_grid = np.full((3, 3), base_color, dtype=int)
        default_grid[0, 0] = pattern_color  # Add at least one pattern
        return default_grid

    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        # Randomly select colors ensuring they are all different
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)

        # Store task variables
        taskvars = {
            'base_color': available_colors[0],
            'pattern_color': available_colors[1],
        }
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # All grids are 9x9 for this task
        for _ in range(num_train_examples):
            gridvars = {}  # No grid-specific variables needed
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {}
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