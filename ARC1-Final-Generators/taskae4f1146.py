from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Taskae4f1146Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are squares and are of size {vars['grid_size']} x {vars['grid_size']}.",
            "The grid consists of exactly 4 sub-grids of 3x3 spread across the main grid with proper spacing between them, these sub-grids may or may not have patterns within them. These patterns can be just single cells or scattered cells, or an actual pattern.",
            "These sub-grids consist of two colors namely {color('base_color')} color which contributes to the base color of the sub-grid and {color('pattern_color')} color which contributes to the formation of patterns within the sub-grid."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is 3x3 fixed size.",
            "The output grid is formed by identifying that one sub-grid from the input grid which has the maximum number of {color('target_color')} colored cells.",
            "The output grid is constructed by identifying the 3x3 sub-grid that contains the highest number of {color('target_color')} cells."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def check_spacing(self, pos1, pos2, min_gap=1):
        """Check if two 3x3 sub-grids have proper spacing (at least min_gap cells apart)."""
        r1, c1 = pos1
        r2, c2 = pos2
        
        # Check if they overlap or are too close
        if (abs(r1 - r2) < 3 + min_gap) and (abs(c1 - c2) < 3 + min_gap):
            return False  # Too close or overlapping
        
        return True
    
    def find_valid_positions(self, grid_size, num_subgrids=4, min_gap=1):
        """Find valid positions for sub-grids with proper spacing."""
        # Generate all possible positions for 3x3 sub-grids
        all_positions = []
        for r in range(grid_size - 2):
            for c in range(grid_size - 2):
                all_positions.append((r, c))
        
        # Try to find a valid combination of positions
        max_attempts = 1000
        
        for attempt in range(max_attempts):
            if len(all_positions) < num_subgrids:
                break
                
            # Randomly select positions
            candidate_positions = random.sample(all_positions, num_subgrids)
            
            # Check if all pairs have proper spacing
            valid_combination = True
            for i in range(len(candidate_positions)):
                for j in range(i + 1, len(candidate_positions)):
                    if not self.check_spacing(candidate_positions[i], candidate_positions[j], min_gap):
                        valid_combination = False
                        break
                if not valid_combination:
                    break
            
            if valid_combination:
                return candidate_positions
        
        # Fallback: place sub-grids in corners if grid is large enough
        if grid_size >= 8:
            return [(0, 0), (0, grid_size-3), (grid_size-3, 0), (grid_size-3, grid_size-3)]
        elif grid_size >= 6:
            return [(0, 0), (0, grid_size-3), (grid_size-3, 0)]
        else:
            return [(0, 0)]
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create a variable-sized input grid with exactly 4 sub-grids (3x3 each) placed with proper spacing."""
        # Get grid dimensions from taskvars
        grid_size = taskvars['grid_size']
        
        # Create a grid filled with zeros
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Get colors from taskvars
        base_color = taskvars['base_color']
        pattern_color = taskvars['pattern_color']
        target_color = taskvars['target_color']
        
        # Find valid positions with proper spacing
        selected_positions = self.find_valid_positions(grid_size, num_subgrids=4, min_gap=1)
        
        if not selected_positions:
            # If no valid positions found, return empty grid (shouldn't happen with proper grid sizing)
            return grid
        
        # Create clear pattern: ensure one sub-grid has significantly more target_color cells
        num_subgrids = len(selected_positions)
        
        # Assign target color counts: [1, 2, 3, max] where max is clearly the highest
        if num_subgrids >= 4:
            target_counts = [1, 2, 3, 6]  # Last one has clearly the most
        elif num_subgrids == 3:
            target_counts = [1, 2, 5]     # Last one has clearly the most  
        elif num_subgrids == 2:
            target_counts = [2, 6]        # Last one has clearly the most
        else:
            target_counts = [6]           # Only one sub-grid
        
        # Shuffle positions but keep counts ordered (so we know which has most)
        random.shuffle(selected_positions)
        
        # Create sub-grids at selected positions
        for idx, (start_i, start_j) in enumerate(selected_positions):
            desired_target_count = target_counts[idx] if idx < len(target_counts) else 1
            
            # Fill the 3x3 subgrid with base_color first
            for i in range(3):
                for j in range(3):
                    if start_i + i < grid_size and start_j + j < grid_size:
                        grid[start_i + i, start_j + j] = base_color
            
            # Now place the exact number of target_color cells
            if target_color == pattern_color:
                # We want specific number of pattern_color cells
                subgrid_positions = []
                for i in range(3):
                    for j in range(3):
                        if start_i + i < grid_size and start_j + j < grid_size:
                            subgrid_positions.append((start_i + i, start_j + j))
                
                # Select random positions for target color
                target_positions = random.sample(subgrid_positions, min(desired_target_count, len(subgrid_positions)))
                
                for pos_r, pos_c in target_positions:
                    grid[pos_r, pos_c] = pattern_color
                    
            elif target_color == base_color:
                # We want specific number of base_color cells
                # Fill some positions with pattern_color, leaving desired_target_count as base_color
                subgrid_positions = []
                for i in range(3):
                    for j in range(3):
                        if start_i + i < grid_size and start_j + j < grid_size:
                            subgrid_positions.append((start_i + i, start_j + j))
                
                total_cells = len(subgrid_positions)
                pattern_cells_count = max(0, total_cells - desired_target_count)
                
                if pattern_cells_count > 0:
                    pattern_positions = random.sample(subgrid_positions, pattern_cells_count)
                    
                    for pos_r, pos_c in pattern_positions:
                        grid[pos_r, pos_c] = pattern_color
                # The remaining cells already have base_color
        
        return grid
    
    def transform_input(self, grid, taskvars):
        """Extract the 3x3 sub-grid with the maximum number of target_color cells."""
        target_color = taskvars['target_color']
        
        # Find the sub-grid with maximum target_color count
        best_subgrid = None
        max_count = -1
        
        # Find all possible 3x3 sub-grids
        for start_r in range(grid.shape[0] - 2):
            for start_c in range(grid.shape[1] - 2):
                subgrid = grid[start_r:start_r+3, start_c:start_c+3]
                
                # Only consider sub-grids that have some colored cells (not all zeros)
                if np.any(subgrid != 0):
                    # Count target_color cells in this sub-grid
                    target_count = np.sum(subgrid == target_color)
                    
                    if target_count > max_count:
                        max_count = target_count
                        best_subgrid = subgrid.copy()
        
        # Return the sub-grid with maximum target_color count
        if best_subgrid is not None:
            return best_subgrid
        
        # Fallback: return empty 3x3 grid
        return np.zeros((3, 3), dtype=int)

    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        # Randomly select colors ensuring they are all different
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)

        # Generate variable grid size (must be large enough to fit 4 sub-grids with spacing)
        min_size = 10  # Increased to ensure better spacing
        max_size = 15  # Reasonable maximum
        
        grid_size = random.randint(min_size, max_size)

        # Choose base and pattern colors
        base_color = available_colors[0]
        pattern_color = available_colors[1]
        
        # Always use pattern_color as target for consistency
        target_color = pattern_color

        # Store task variables
        taskvars = {
            'grid_size': grid_size,
            'base_color': base_color,
            'pattern_color': pattern_color,
            'target_color': target_color,
        }
        
        # Generate 3-5 training examples with same task variables
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        for _ in range(num_train_examples):
            gridvars = {}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example with same task variables
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

# Test code
if __name__ == "__main__":
    generator = Taskae4f1146Generator()
    taskvars, train_test_data = generator.create_grids()
    
    print("Task Variables:", taskvars)
    ARCTaskGenerator.visualize_train_test_data(train_test_data)