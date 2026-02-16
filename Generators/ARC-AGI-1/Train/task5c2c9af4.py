from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, List, Any, Tuple, Optional
from Framework.transformation_library import find_connected_objects, GridObject

class Task5c2c9af4Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain exactly three same-colored cells, placed equidistantly in a diagonal line.",
            "This diagonal line can be parallel to the main diagonal (top-left to bottom-right) or the inverse diagonal (top-right to bottom-left).",
            "The grid color and the positions of the three same-colored cells vary across examples.",
            "The colored cells are spaced such that there is at least one and at most three empty cells between any two colored cells, depending on the grid size.",
            "All remaining cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying the three same-colored cells forming a diagonal line.",
            "These cells guide the creation of one-cell wide rectangular frames in the output grid.",
            "The middle cell out of the three same-colored cells remains unchanged, while the first and last colored cells define two opposite corners of the innermost rectangular frame.",
            "This frame is drawn by connecting the corners, forming a one-cell wide rectangular frame.",
            "After the first frame is created, additional frames are added around it—each separated by the same number of empty cells that match the spacing between the colored cells in the input.",
            "New frames continue to be added until the grid edges are reached.",
            "If a frame cannot be fully drawn due to grid boundaries, any sides that can be added are still included."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        
        grid_size = random.randint(9, 30)  # Increased minimum size to ensure enough space
        
        # Define task variables
        taskvars = {
            'grid_size': grid_size
        }
        
        # Generate 3-6 training examples
        num_train_examples = random.randint(3, 6)
        train_examples = []
        
        used_colors = set()

        # --- Enforce at least one main-diagonal and one inverse-diagonal example in TRAIN ---
        # We'll force the first two training examples to be one of each.
        forced_diagonals = [True, False]  # True = main, False = inverse
        random.shuffle(forced_diagonals)  # randomize which comes first
        
        for i in range(num_train_examples):
            # Choose a color that hasn't been used yet
            color = random.randint(1, 9)
            while color in used_colors:
                color = random.randint(1, 9)
            used_colors.add(color)
            
            # Force diagonal type for first two examples; random afterwards
            if i < 2:
                is_main_diagonal = forced_diagonals[i]
            else:
                is_main_diagonal = random.choice([True, False])
            
            # Create an input grid with the chosen color + enforced diagonal
            gridvars = {'color': color, 'is_main_diagonal': is_main_diagonal}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example with a different color
        test_color = random.randint(1, 9)
        while test_color in used_colors:
            test_color = random.randint(1, 9)
        
        # Test diagonal can remain random (or you can force it too if you want)
        test_gridvars = {'color': test_color, 'is_main_diagonal': random.choice([True, False])}
        test_input = self.create_input(taskvars, test_gridvars)
        
        test_examples = [{
            'input': test_input,
            'output': self.transform_input(test_input, taskvars)
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        color = gridvars['color']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Determine diagonal type (main or inverse)
        # If provided (forced), use it; otherwise pick randomly.
        is_main_diagonal = gridvars.get('is_main_diagonal', random.choice([True, False]))
        
        # Calculate spacing - ensure at least 2 (so there's at least 1 empty cell between points)
        min_spacing = 2  # Minimum spacing to ensure at least one empty cell between points
        max_spacing = min(4, (grid_size - 6) // 2)  # Ensure we don't go out of bounds
        
        # Handle case where min_spacing equals max_spacing
        if min_spacing > max_spacing:
            # If the constraints are impossible, increase grid size or adjust spacing
            spacing = 2  # Fallback to minimum spacing
        elif min_spacing == max_spacing:
            spacing = min_spacing  # Only one possible value
        else:
            spacing = random.randint(min_spacing, max_spacing)
        
        # Calculate safe boundaries to place the three points
        safe_margin = 2
        
        # Ensure we have enough space for the pattern
        max_start_row = grid_size - (2 * spacing) - safe_margin
        max_start_col = grid_size - (2 * spacing) - safe_margin
        
        # Make sure we have valid starting positions
        if max_start_row <= safe_margin:
            max_start_row = safe_margin + 1
        if max_start_col <= safe_margin:
            max_start_col = safe_margin + 1
            
        # Choose starting point
        start_row = random.randint(safe_margin, max_start_row)
        start_col = random.randint(safe_margin, max_start_col)
        
        # Place three colored cells along the chosen diagonal
        if is_main_diagonal:
            # Main diagonal (top-left to bottom-right)
            for i in range(3):
                r = start_row + i * spacing
                c = start_col + i * spacing
                grid[r, c] = color
        else:
            # Inverse diagonal (top-right to bottom-left)
            for i in range(3):
                r = start_row + i * spacing
                c = start_col + (2 - i) * spacing  # Decreasing column
                grid[r, c] = color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Create a copy of the input grid
        output_grid = grid.copy()
        
        # Find the three colored cells
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        if len(objects) == 0:
            return output_grid  # Safety check
        
        # Get coordinates of the colored cells
        cells = []
        for obj in objects:
            for r, c, color in obj.cells:
                cells.append((r, c, color))
        
        # Sort cells - we need to determine which is the middle point
        if len(cells) != 3:
            return output_grid  # Safety check
        
        # Sort by row first to get them in sequence
        cells.sort(key=lambda x: x[0])
        
        # The middle cell remains unchanged
        middle_cell = cells[1]
        middle_r, middle_c, color = middle_cell
        
        # Determine if this is a main diagonal or inverse diagonal
        is_main_diagonal = (cells[2][1] > cells[0][1])
        
        # First and last cells define corners of the innermost frame
        min_r = min(cells[0][0], cells[2][0])
        max_r = max(cells[0][0], cells[2][0])
        
        min_c = min(cells[0][1], cells[2][1])
        max_c = max(cells[0][1], cells[2][1])
        
        # Calculate spacing between cells
        spacing = (max_r - min_r) // 2
        
        # Draw frames until we reach grid boundaries
        grid_size = grid.shape[0]
        current_min_r, current_min_c = min_r, min_c
        current_max_r, current_max_c = max_r, max_c
        
        while True:
            # Draw current frame (horizontal and vertical lines)
            # Draw horizontal lines if within bounds
            for c in range(max(0, current_min_c), min(grid_size, current_max_c + 1)):
                if 0 <= current_min_r < grid_size:
                    output_grid[current_min_r, c] = color
                if 0 <= current_max_r < grid_size:
                    output_grid[current_max_r, c] = color
            
            # Draw vertical lines if within bounds
            for r in range(max(0, current_min_r + 1), min(grid_size, current_max_r)):
                if 0 <= current_min_c < grid_size:
                    output_grid[r, current_min_c] = color
                if 0 <= current_max_c < grid_size:
                    output_grid[r, current_max_c] = color
            
            # Calculate next frame position
            current_min_r -= spacing
            current_min_c -= spacing
            current_max_r += spacing
            current_max_c += spacing
            
            # Check if next frame would be out of bounds completely
            if (current_min_r < 0 and current_max_r >= grid_size) or \
               (current_min_c < 0 and current_max_c >= grid_size):
                break
        
        return output_grid
